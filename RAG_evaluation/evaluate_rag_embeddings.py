#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG Evaluation (Relevance & Accuracy) with RAGAS
- Supports black-box RAG (no retrieved docs) via RubricsScore
- Supports retrieval-aware scoring via ResponseRelevancy + answer_correctness
- Outputs per-query scores + aggregated stats + bootstrap CIs
"""

import os, json, re, random, math
import numpy as np
import pandas as pd
import atexit
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ResponseRelevancy
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tqdm import tqdm
import asyncio
from datasets import Dataset

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ---------- Config ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

INPUT_CSV = "testset.csv"           # change to your path
OUTPUT_RESULTS_CSV = "rag_eval_results.csv"
OUTPUT_SUMMARY_CSV = "rag_eval_summary.csv"
USE_RETRIEVAL_IF_AVAILABLE = False   # set False to force black-box mode

# ---------- Text normalization ----------
try:
    from unidecode import unidecode
    USE_UNIDECODE = True
except:
    USE_UNIDECODE = False

def normalize_text(s) -> str:
    # Handle None/NaN/float
    if s is None:
        return ""
    try:
        # Pandas NaN is a float and not equal to itself
        if isinstance(s, float) and math.isnan(s):
            return ""
    except Exception:
        pass

    # Coerce everything to string
    s = str(s)

    s = s.strip()
    if USE_UNIDECODE:
        s = unidecode(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    # keep punctuation for LLM evaluators
    return s

from pathlib import Path

def load_dataset(path: str) -> pd.DataFrame:
    # Read as strings; keep empty strings (no NaN)
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,   # prevents empty -> NaN
        na_filter=False,         # speeds up & keeps raw strings
        encoding="utf-8-sig"     # handles BOM if present
    )

    required = {"query", "grounded_answer", "rag_output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Ensure required cols are strings with no NaN
    for col in ["query", "grounded_answer", "rag_output"]:
        df[col] = df[col].astype(str).fillna("")

    # Optional column: retrieved_docs as JSON string
    if "retrieved_docs" not in df.columns:
        df["retrieved_docs"] = "[]"
    else:
        df["retrieved_docs"] = df["retrieved_docs"].astype(str).fillna("")
        # Normalize to JSON list string
        def _sanitize_docs(x: str) -> str:
            x = (x or "").strip()
            if not x:
                return "[]"
            if (x.startswith("[") and x.endswith("]")) or (x.startswith("{") and x.endswith("}")):
                return x
            # treat as a single snippet string
            import json as _json
            return _json.dumps([x], ensure_ascii=False)
        df["retrieved_docs"] = df["retrieved_docs"].apply(_sanitize_docs)

    return df


# ---------- RAGAS imports ----------
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import answer_correctness
from langchain_community.vectorstores import FAISS

# Some RAGAS versions name this metric/class slightly differently.
# For rubric scoring (1–5), we’ll use RubricsScore:
from ragas.metrics import RubricsScore



# ---------- Evaluator LLM wiring ----------
# Choose ONE: OpenAI / LiteLLM / HF / your in-house
# Here is a generic OpenAI example; replace with your own evaluator LLM.
EVALUATOR_LLM = None
EVALUATOR_EMBEDDINGS = None
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

def setup_openai_evaluator(model_name: str = "gpt-4o"):
    # ---- Option A: LangChain LLM style (most compatible with RAGAS) ----
    try:
        from langchain_openai import ChatOpenAI
        chat = ChatOpenAI(model=model_name, temperature=0)
        return LangchainLLMWrapper(chat)
    except Exception:
        pass

    # Fallback: raise to tell the user to install langchain_openai
    raise RuntimeError(
        "Please `pip install langchain_openai` or swap in your evaluator LLM that RAGAS accepts."
    )

def setup_google_embeddings():
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        ge = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Wrap for RAGAS
        return LangchainEmbeddingsWrapper(ge)
        
    except Exception:
        return None  



def init_evaluator():
    global EVALUATOR_LLM, EVALUATOR_EMBEDDINGS
    EVALUATOR_LLM = setup_openai_evaluator()
    EVALUATOR_EMBEDDINGS = setup_google_embeddings()

# ---------- Rubric templates (you can edit or load from file) ----------
RUBRIC_RELEVANCE = {
    "score1_description": "The response is entirely off-topic or irrelevant to the user query.",
    "score2_description": "Mostly off-topic; only tangentially related to the query.",
    "score3_description": "Addresses the general topic but not the specific question or includes substantial irrelevant information.",
    "score4_description": "Mostly focused on the query with minor digressions or unnecessary details.",
    "score5_description": "Fully relevant and focused on the user query.",
}

RUBRIC_ACCURACY = {
    "score1_description": "Completely incorrect, contradicts the reference, or fabricates facts.",
    "score2_description": "Partially accurate but with major errors or omissions that change meaning.",
    "score3_description": "Mostly accurate but missing key details or containing minor errors.",
    "score4_description": "Accurate with only minor omissions or slight inaccuracies.",
    "score5_description": "Fully accurate, precise, and consistent with the reference.",
}

# ---------- Helpers ----------
def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return (math.nan, math.nan)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    low = np.percentile(means, 100 * (alpha / 2))
    high = np.percentile(means, 100 * (1 - alpha / 2))
    return (float(low), float(high))

def _shutdown_grpc_aio():
    try:
        from grpc import aio as grpc_aio
        grpc_aio.shutdown_grpc_aio()
    except Exception:
        pass
atexit.register(_shutdown_grpc_aio)

def to_hf_dataset_for_correctness(df: pd.DataFrame) -> Dataset:
    # RAGAS answer_correctness expects: question, answer, ground_truth
    data = {
        "question": df["query"].tolist(),
        "answer": df["rag_output"].tolist(),
        "ground_truth": df["grounded_answer"].tolist(),
    }
    return Dataset.from_dict(data)

# ---------- Main evaluation ----------
def main():
    print("Loading data…")
    df = load_dataset(INPUT_CSV)

    # Keep originals for rubric prompts; also build normalized variants
    df["query_norm"] = df["query"].apply(normalize_text)
    df["grounded_norm"] = df["grounded_answer"].apply(normalize_text)
    df["rag_norm"] = df["rag_output"].apply(normalize_text)

    print("Initializing evaluator LLM…")
    init_evaluator()

    per_row_records = []

    # ---- Metrics path 1: Black-box compatible (Rubrics 1–5) ----
    # Relevance via rubric (needs only query & response)
    relevance_rubric_scorer = RubricsScore(rubrics=RUBRIC_RELEVANCE, llm=EVALUATOR_LLM)
    # Accuracy via rubric (needs response & reference)
    accuracy_rubric_scorer = RubricsScore(rubrics=RUBRIC_ACCURACY, llm=EVALUATOR_LLM)

    # ---- Metrics path 2: Example-based numeric scores (0–1) ----
    # Correctness: built-in RAGAS metric
    hf_dataset = to_hf_dataset_for_correctness(df)
    correctness_scores = evaluate(hf_dataset, metrics=[answer_correctness]).to_pandas()
    # The dataframe will contain a column like 'answer_correctness'
    correctness_list = correctness_scores["answer_correctness"].tolist()

    # Relevance (0–1) with retrieval if available; otherwise skip and rely on rubric relevance
    
    
    # ---- Response relevancy scorer (query-only; embeddings required by RAGAS) ----
    response_relevancy_scorer = None
    try:
        response_relevancy_scorer = ResponseRelevancy(
            llm=EVALUATOR_LLM,
            embeddings=EVALUATOR_EMBEDDINGS  
        )
        # Optional: quick sanity print
        print("[DEBUG] ResponseRelevancy ready:",
            type(EVALUATOR_LLM).__name__, type(EVALUATOR_EMBEDDINGS).__name__)
    except Exception as e:
        print("[WARN] ResponseRelevancy init failed:", repr(e))
        raise RuntimeError(f"ResponseRelevancy init failed: {e!r}")




    print("Scoring per query…")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        query = row["query"]
        reference = row["grounded_answer"]
        response = row["rag_output"]

        # Build SingleTurnSample for rubric scorers
        # For accuracy rubric: we use 'reference' (gold) and 'response'
        acc_sample = SingleTurnSample(response=response, reference=reference)

        # For relevance rubric: we use 'query' as "user_input" and 'response'
        # Some RAGAS versions accept 'user_input' or 'query'; SingleTurnSample supports 'user_input' field.
        rel_sample = SingleTurnSample(user_input=query, response=response)

        # 1) Rubrics-based relevance (1–5)
        rel_rubric = relevance_rubric_scorer.single_turn_ascore(rel_sample)
        if hasattr(rel_rubric, "__await__"):  # async in some versions
            rel_rubric = asyncio.get_event_loop().run_until_complete(rel_rubric)

        # 2) Rubrics-based accuracy (1–5)
        acc_rubric = accuracy_rubric_scorer.single_turn_ascore(acc_sample)
        if hasattr(acc_rubric, "__await__"):
            acc_rubric = asyncio.get_event_loop().run_until_complete(acc_rubric)

        # 3) Answer correctness (0–1) from vector (already computed)
        ans_correctness = correctness_list[i]

        # 4) Response relevancy (0–1) with retrieval, if we have contexts
        resp_relevancy = None
        if response_relevancy_scorer is not None:
            rel_q = SingleTurnSample(user_input=query, response=response)
            rr = response_relevancy_scorer.single_turn_ascore(rel_q)
            if hasattr(rr, "__await__"):
                rr = asyncio.get_event_loop().run_until_complete(rr)
            resp_relevancy = float(rr)

        per_row_records.append({
            "idx": i,
            "query": query,
            "gold": reference,
            "response": response,
            "relevance_rubric_1to5": float(rel_rubric),
            "accuracy_rubric_1to5": float(acc_rubric),
            "answer_correctness_0to1": float(ans_correctness),
            "response_relevancy_0to1": resp_relevancy,
        })

        # Write per-query CSV
    df_scores = pd.DataFrame(per_row_records)
    df_scores.to_csv(OUTPUT_RESULTS_CSV, index=False)

        # Aggregation
    df_scores = pd.DataFrame(per_row_records)

    def agg(col):
        vals = df_scores[col].dropna().tolist()
        avg = float(np.mean(vals)) if vals else float("nan")
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        low, high = bootstrap_ci(vals) if vals else (float("nan"), float("nan"))
        return avg, std, low, high

    summary = {}
    for metric in ["relevance_rubric_1to5", "accuracy_rubric_1to5",
                    "answer_correctness_0to1", "response_relevancy_0to1"]:
        avg, std, low, high = agg(metric)
        summary[metric] = {
            "n": int(df_scores[metric].notna().sum()),
            "mean": avg, "std": std,
            "ci95_low": low, "ci95_high": high
        }

    summary_rows = []
    for metric, stats in summary.items():
        row = {"metric": metric}
        row.update(stats)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

    print("\n=== Aggregated Results ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved per-query results to {OUTPUT_RESULTS_CSV} and summary to {OUTPUT_SUMMARY_CSV}\n")


if __name__ == "__main__":
    # Some RAGAS versions require asyncio run for async scorers;
    # this script uses sync calls; if your version is fully async,
    # we can switch to asyncio.run(main_async()).
    main()
