import os
import pathlib
import uuid

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


class Config:
    """Configuration class for the RAG application."""
    load_dotenv()

    BASE_DIR = pathlib.Path(__file__).resolve().parent
    LOG_DIR = BASE_DIR / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    SESSION_ID = uuid.uuid4().hex[:8]
    LOG_PATH = LOG_DIR / f"session-{SESSION_ID}.csv"

    print(f"[LOG] Session log: {LOG_PATH}")

    # Constants
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150
    MAX_TOKENS = 4096
    MODEL_NAME = "gpt-4o"
    TEMPERATURE = 0.7

    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("Please set GOOGLE_API_KEY in your .env file")

    EMBEDDING_CACHE_PATH = BASE_DIR / "embeddings_cache.db"

    MQR_NUM = 4

    PROMPT = PromptTemplate(
        template="""Context: {context}

Question: {question}

You are a search bot that forms a coherent answer to a user query based on search results that are provided to you.
If the search results are irrelevant to the question, respond with "I do not have enough information to answer this question."
Do not base your response on information or knowledge that is not in the search results.
Make sure your response is answering the query asked
Consider that each search result is a partial segment from a bigger text, and may be incomplete.

If the question is about images or tables, refer to them specifically in your answer.""",
        input_variables=["context", "question"]
    )

    MQR_PROMPT = PromptTemplate(
        input_variables=["question", "n"],
        template=(
            "You are a domain-savvy query rewriter. Generate {n} diverse reformulations of the question "
            "to capture synonyms, abbreviations, and likely headings found in building certification PDFs "
            "(e.g., WELL, Fitwel). Keep each on its own line, no numbering or extra text.\n\n"
            "Question: {question}\n\n"
            "Rewrites:"
        )
    )