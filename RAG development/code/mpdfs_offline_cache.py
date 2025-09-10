import os
from dotenv import load_dotenv
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseLanguageModel
import hashlib
import numpy as np
import time
import random
import tempfile
import pymupdf as fitz
from PIL import Image
import pytesseract
import io
import pandas as pd
import filetype
import re
import csv
import uuid
import pathlib
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

# --------------------
# Offline setup
# --------------------
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 4096
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3:8b")  # local Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TEMPERATURE = 0.7

# Local LLM (Ollama)
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    num_ctx=8192,
    num_predict=4096,
    base_url=OLLAMA_BASE_URL
)

# Local embeddings (offline HF model)
EMBED_MODEL_PATH = r"C:\models\GIST-Embedding-v0"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_PATH,
    model_kwargs={"device": "cpu", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": True},
)

# --------------------
# CSV logging setup
# --------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_ID = uuid.uuid4().hex[:8]
LOG_PATH = LOG_DIR / f"session-{SESSION_ID}.csv"
print(f"[LOG] Session log: {LOG_PATH}")

def append_log_entry_csv(log_path: pathlib.Path, entry: dict):
    """Append an entry to a CSV file, create header if new file."""
    file_exists = log_path.exists()
    try:
        with log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)
    except Exception as e:
        print(f"[LOG][ERROR] Could not write to {log_path}: {e}")

# --------------------
# In-memory session cache
# --------------------
INDEX_CACHE: Dict[str, Any] = {
    "signatures": None,        # list of (path, size, mtime)
    "hybrid_retriever": None,  # EnsembleRetriever
    "dense_retriever": None,   # FAISS retriever
    "sparse_retriever": None,  # BM25 retriever
    "images": None,            # keep for logging consistency
    "tables": None,
    "texts_count": 0,
}

def file_signatures(pdf_file_objs):
    """
    Return content-based signatures for uploaded files so Gradio temp paths
    don't force a rebuild each submit. Signature = (sha256, size).
    """
    sigs = []
    for f in (pdf_file_objs or []):
        p = f.name
        h = hashlib.sha256()
        try:
            with open(p, "rb") as rf:
                # read in chunks to avoid big memory usage
                for chunk in iter(lambda: rf.read(1 << 20), b""):
                    h.update(chunk)
            st = os.stat(p)
            sigs.append((h.hexdigest(), st.st_size))
        except Exception:
            sigs.append(("ERROR_HASH", 0))
    sigs.sort(key=lambda x: x[0])  # stable order
    return sigs


def signatures_changed(a, b) -> bool:
    return a != b

# --------------------
# Prompt
# --------------------
PROMPT = PromptTemplate(
    template="""Context: {context}

Question: {question}

You are a search bot that forms a coherent answer to a user query based on search results that are provided to you.
If the search results are irrelevant to the question, respond with "I do not have enough information to answer this question."
Do not base your response on information or knowledge that is not in the search results.
Make sure your response is answering the query asked.
Consider that each search result is a partial segment from a bigger text, and may be incomplete.

If the question is about images or tables, refer to them specifically in your answer.""",
    input_variables=["context", "question"]
)

# --------------------
# PDF processing helpers
# --------------------
def process_single_pdf_text(pdf_path):
    """Extract text from a single PDF using PyMuPDF and split into chunks."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def extract_images_and_tables_single_pdf(pdf_path):
    """Extract images and tables from a single PDF."""
    doc = fitz.open(pdf_path)
    images = []
    tables = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append((f"PDF: {os.path.basename(pdf_path)}, Page {page_num + 1}, Image {img_index + 1}", image))

        tables_on_page = page.find_tables()
        for table_index, table in enumerate(tables_on_page):
            df = pd.DataFrame(table.extract())
            tables.append((f"PDF: {os.path.basename(pdf_path)}, Page {page_num + 1}, Table {table_index + 1}", df))

    return images, tables

# --------------------
# Build or reuse index (cache)
# --------------------
def build_or_reuse_index(pdf_files_list):
    """Build the index if PDFs changed; otherwise reuse from cache. Returns a log string."""
    sigs = file_signatures(pdf_files_list)

    if INDEX_CACHE["hybrid_retriever"] is None or signatures_changed(sigs, INDEX_CACHE["signatures"]):
        processing_log = "--- PDF Processing Log ---\n"
        all_texts, all_images, all_tables = [], [], []

        if not pdf_files_list:
            INDEX_CACHE["signatures"] = None
            INDEX_CACHE["hybrid_retriever"] = None
            return processing_log + "No PDFs uploaded.\n"

        for pdf_file_obj in pdf_files_list:
            pdf_path = pdf_file_obj.name
            actual_mime_type = "UNKNOWN"
            try:
                actual_mime_type = filetype.guess_mime(pdf_path)
                print(f"DEBUG: Detected MIME type for {os.path.basename(pdf_path)}: {actual_mime_type}")
            except Exception as e:
                processing_log += f"  - WARNING: Could not determine MIME type for {os.path.basename(pdf_path)}: {e}\n"

            if actual_mime_type != "application/pdf":
                processing_log += f"  - SKIPPING: '{os.path.basename(pdf_path)}' is not a PDF (detected as {actual_mime_type}).\n"
                continue

            processing_log += f"Processing PDF: {os.path.basename(pdf_path)}\n"
            try:
                texts_from_pdf = process_single_pdf_text(pdf_path)
                images_from_pdf, tables_from_pdf = extract_images_and_tables_single_pdf(pdf_path)
                all_texts.extend(texts_from_pdf)
                all_images.extend(images_from_pdf)
                all_tables.extend(tables_from_pdf)
                processing_log += f"  - Chunks extracted: {len(texts_from_pdf)}\n"
                processing_log += f"  - Images extracted: {len(images_from_pdf)}\n"
                processing_log += f"  - Tables extracted: {len(tables_from_pdf)}\n"
            except Exception as e:
                processing_log += f"  - ERROR processing {os.path.basename(pdf_path)}: {e}\n"
                continue

        if not all_texts:
            INDEX_CACHE["signatures"] = None
            INDEX_CACHE["hybrid_retriever"] = None
            return processing_log + "No text content could be extracted from any of the provided PDFs.\n"

        processing_log += f"\n--- Combined Processing Summary ---\n"
        processing_log += f"Total number of text chunks from all PDFs: {len(all_texts)}\n"
        processing_log += f"Total number of images extracted from all PDFs: {len(all_images)}\n"
        processing_log += f"Total number of tables extracted from all PDFs: {len(all_tables)}\n\n"

        # Build retrievers ONCE using your offline embeddings
        vectorstore = FAISS.from_texts(all_texts, embeddings)
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        sparse_retriever = BM25Retriever.from_texts(all_texts)
        sparse_retriever.k = 4

        hybrid = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )

        INDEX_CACHE["signatures"] = sigs
        INDEX_CACHE["hybrid_retriever"] = hybrid
        INDEX_CACHE["dense_retriever"] = dense_retriever
        INDEX_CACHE["sparse_retriever"] = sparse_retriever
        INDEX_CACHE["images"] = all_images
        INDEX_CACHE["tables"] = all_tables
        INDEX_CACHE["texts_count"] = len(all_texts)

        processing_log += "(Index built)\n"
        return processing_log

    return "(Reusing cached index — PDFs unchanged)\n"

# --------------------
# Query expansion + RAG pipeline
# --------------------
def expand_query(query: str, llm_: BaseLanguageModel) -> str:
    prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Given the following query, generate 3-5 related terms or phrases that could be relevant to the query. "
            "Separate the terms with commas.\n\n"
            "Query: {query}\n\nRelated terms:"
        ),
    )
    chain = LLMChain(llm=llm_, prompt=prompt)
    response = chain.invoke({"query": query})
    expanded_terms = [term.strip() for term in response["text"].split(",") if term.strip()]
    expanded_query = f"{query} {' '.join(expanded_terms)}"
    return expanded_query

def rag_pipeline(query, qa_chain, retriever, images, tables, llm_instance):
    expanded_query = expand_query(query, llm_instance)
    relevant_docs = retriever.get_relevant_documents(expanded_query)

    log = "Query Expansion:\n"
    log += f"Original query: {query}\n"
    log += f"Expanded query: {expanded_query}\n\n"
    log += "Relevant chunks (Hybrid):\n"

    retrieved_chunks = []
    for i, doc in enumerate(relevant_docs, 1):
        retrieved_chunks.append(doc.page_content)
        log += f"Chunk {i} sample: {doc.page_content[:200]}...\n\n"

    log += f"Number of images in all PDFs: {len(images)}\n"
    log += f"Number of tables in all PDFs: {len(tables)}\n\n"

    response = qa_chain.invoke({"query": query})
    return response["result"], log, retrieved_chunks

# --------------------
# Orchestrator (uses cache)
# --------------------
def process_pdfs_and_query(pdf_files_list, query):
    """Ensure index exists/reused, then answer the query."""
    processing_log = build_or_reuse_index(pdf_files_list)

    if INDEX_CACHE["hybrid_retriever"] is None:
        return "No index available. Please upload valid PDFs.", processing_log, []

    hybrid_retriever = INDEX_CACHE["hybrid_retriever"]
    all_images = INDEX_CACHE["images"] or []
    all_tables = INDEX_CACHE["tables"] or []

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result, chunks_log, retrieved_chunks = rag_pipeline(
        query, qa, hybrid_retriever, all_images, all_tables, llm
    )
    final_log = processing_log + chunks_log
    return result, final_log, retrieved_chunks

# --------------------
# Gradio interface
# --------------------
def gradio_interface(pdf_files_list_from_gradio, query):
    t_start = time.perf_counter()

    if not pdf_files_list_from_gradio:
        return "Please upload at least one PDF file.", "No PDFs uploaded."

    result, full_log, retrieved_chunks = process_pdfs_and_query(pdf_files_list_from_gradio, query)

    total_time = time.perf_counter() - t_start
    full_log += f"\n=== RAG runtime ===\nTotal: {total_time:.3f} seconds\n"
    result = f"(Generated in {total_time:.2f}s)\n\n{result}"

    # Append a row to CSV after every query
    append_log_entry_csv(LOG_PATH, {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": result,
        "processing_log": full_log,
        "retrieved_chunks": " || ".join(retrieved_chunks)
    })

    return result, full_log

def main():
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.File(label="Upload PDFs", file_count="multiple"),
            gr.Textbox(label="Enter your question"),
        ],
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Textbox(label="Processing Log"),
        ],
        title="RAG Pretest (Multiple PDFs) — Offline",
        description="LangChain + Ollama (llama3:8b) + local HF embeddings. Index cached per session.",
    )
    iface.launch()

if __name__ == "__main__":
    main()
