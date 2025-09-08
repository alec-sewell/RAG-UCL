import os
from dotenv import load_dotenv
import json
from datetime import datetime, timezone
import uuid
import pathlib
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever 
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document 

import numpy as np
import time
import random
import csv
import tempfile
# from langchain_community.document_loaders import BSHTMLLoader # This was for web scraping HTML, not needed for local PDF
from langchain.memory import ConversationBufferMemory

import pymupdf as fitz
import pytesseract # Make sure pytesseract is installed and Tesseract OCR is configured if you're using it for images
from PIL import Image
import io
import re
import pandas as pd
import filetype # <--- NEW: Import filetype

from typing import List, Tuple, Dict, Any








# ---- In-memory session cache (lives while this Python process runs) ----
INDEX_CACHE: Dict[str, Any] = {
    "signatures": None,        # list of (path, size, mtime) for current PDFs
    "hybrid_retriever": None,  # EnsembleRetriever
    "dense_retriever": None,   # FAISS retriever
    "sparse_retriever": None,  # BM25 retriever
    "images": None,            # to keep your log consistent
    "tables": None,
    "texts_count": 0,
}

def file_signatures(pdf_file_objs) -> List[Tuple[str, int, float]]:
    """Return a cheap signature for uploaded files: (path, size, mtime)."""
    sigs = []
    for f in (pdf_file_objs or []):
        p = f.name
        try:
            st = os.stat(p)
            sigs.append((p, st.st_size, st.st_mtime))
        except Exception:
            sigs.append((p, 0, 0.0))
    sigs.sort(key=lambda x: x[0])  # stable order
    return sigs

def signatures_changed(a, b) -> bool:
    return a != b




# Load environment variables
load_dotenv()

BASE_DIR = pathlib.Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = uuid.uuid4().hex[:8]
LOG_PATH = LOG_DIR / f"session-{SESSION_ID}.csv"

print(f"[LOG] Session log: {LOG_PATH}")



# Constants
CHUNK_SIZE = 1000 # Adjusted for better balance with multiple docs
CHUNK_OVERLAP = 200
MAX_TOKENS = 4096
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.7



# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Initialize LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

# [ALEC] Changed the QA prompt
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



# --- Modified for single PDF processing (called in a loop for multiple PDFs) ---
def process_pdfs_and_query(pdf_files_list, query):
    """Ensure index exists/reused, then answer the query."""
    processing_log = build_or_reuse_index(pdf_files_list)

    if INDEX_CACHE["hybrid_retriever"] is None:
        return "No index available. Please upload valid PDFs.", processing_log, []

    hybrid_retriever = INDEX_CACHE["hybrid_retriever"]
    all_images = INDEX_CACHE["images"] or []
    all_tables = INDEX_CACHE["tables"] or []

    # Build QA chain on demand (cheap), reusing the cached retriever
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=hybrid_retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    print("\nRAG Pipeline initialized. Running query...")
    result, chunks_log, retrieved_chunks = rag_pipeline(query, qa, hybrid_retriever, all_images, all_tables, llm)

    final_log = processing_log + chunks_log
    return result, final_log, retrieved_chunks




# --- Modified for single PDF processing (called in a loop for multiple PDFs) ---
def extract_images_and_tables_single_pdf(pdf_path): # Renamed function
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
            # Include PDF name in metadata for clarity in multi-doc scenario
            images.append((f"PDF: {os.path.basename(pdf_path)}, Page {page_num + 1}, Image {img_index + 1}", image))
        
        tables_on_page = page.find_tables()
        for table_index, table in enumerate(tables_on_page):
            df = pd.DataFrame(table.extract())
            # Include PDF name in metadata for clarity in multi-doc scenario
            tables.append((f"PDF: {os.path.basename(pdf_path)}, Page {page_num + 1}, Table {table_index + 1}", df))
    
    return images, tables


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001") 
def create_retrievers(all_texts): 
    """Create embeddings and vector store from text chunks."""
    vectorstore = FAISS.from_texts(all_texts, embeddings)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    sparse_retriever = BM25Retriever.from_texts(all_texts)
    sparse_retriever.k = 4
    return dense_retriever, sparse_retriever

# Multi-query rewrite prompt (generates diverse alternatives)
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
MQR_NUM = 4  




# --- Modified to accept llm_instance for expand_query ---
def rag_pipeline(query, qa_chain, retriever, images, tables, llm_instance):
    # ...
    relevant_docs = retriever.get_relevant_documents(query)
    generated_alts = []
    
    try:
        if hasattr(retriever, "llm_chain"):
            alt_text = retriever.llm_chain.invoke({"question": query})
            if isinstance(alt_text, dict) and "text" in alt_text:
                alt_text = alt_text["text"]
            if isinstance(alt_text, str):
                generated_alts = [line.strip() for line in alt_text.splitlines() if line.strip()]
    except Exception:
        pass  


    # Build log
    log = "Multi-Query Expansion:\n"
    log += f"Original query: {query}\n"
    if generated_alts:
        log += "Generated alternates:\n" + "\n".join(f"- {q}" for q in generated_alts) + "\n"
    log += "\nRelevant chunks (Hybrid + MQR):\n"


    retrieved_chunks = []
    context = ""
    for i, doc in enumerate(relevant_docs, 1):
        retrieved_chunks.append(doc.page_content)
        context += doc.page_content + "\n\n"
        log += f"Chunk {i} sample: {doc.page_content[:200]}...\n\n"

    log += f"Number of images in all PDFs: {len(images)}\n"
    log += f"Number of tables in all PDFs: {len(tables)}\n\n"

    # Run QA
    response = qa_chain.invoke({"query": query})
    return response['result'], log, retrieved_chunks

def safe_slug(text: str, max_len: int = 60) -> str:
    """Make a filesystem-safe short slug from text (for filenames)."""
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_.-]", "", text)
    return text[:max_len] if text else "query"


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


def process_single_pdf_text(pdf_path):
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

        # Build retrievers ONCE using your existing embeddings
        vectorstore = FAISS.from_texts(all_texts, embeddings)
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        sparse_retriever = BM25Retriever.from_texts(all_texts)
        sparse_retriever.k = 12

        hybrid = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )
        
        mqr = MultiQueryRetriever.from_llm(
            retriever=hybrid,
            llm=llm,
            prompt=MQR_PROMPT.partial(n=str(MQR_NUM)),
            include_original=True,   
        )

        INDEX_CACHE["signatures"] = sigs
        INDEX_CACHE["hybrid_retriever"] = mqr
        INDEX_CACHE["dense_retriever"] = dense_retriever
        INDEX_CACHE["sparse_retriever"] = sparse_retriever
        INDEX_CACHE["images"] = all_images
        INDEX_CACHE["tables"] = all_tables
        INDEX_CACHE["texts_count"] = len(all_texts)

        processing_log += "(Index built)\n"
        return processing_log

    return "(Reusing cached index — PDFs unchanged)\n"




# --- Modified Gradio interface function ---
def gradio_interface(pdf_files_list_from_gradio, query): 
    """Gradio interface function."""
    t_start = time.perf_counter()  # Start timer

    
    if not pdf_files_list_from_gradio: 
        return "Please upload at least one PDF file.", "No PDFs uploaded."

    # Pass the list of file objects to the processing function
    result, full_log, retrieved_chunks = process_pdfs_and_query(pdf_files_list_from_gradio, query)
    
    total_time = time.perf_counter() - t_start  # End timer
    full_log += f"\n=== RAG runtime ===\nTotal: {total_time:.3f} seconds\n"

    
    result = f"(Generated in {total_time:.2f}s)\n\n{result}"
    append_log_entry_csv(LOG_PATH, {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "query": query,
    "answer": result,
    "processing_log": full_log,
    "retrieved_chunks": " || ".join(retrieved_chunks)  
    })

    return result, full_log


  
def main():
    """Main function to launch the Gradio interface."""
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.File(label="Upload PDFs", file_count="multiple"), 
            gr.Textbox(label="Enter your question")
        ],
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Textbox(label="Processing Log")
        ],
        title="RAG Pretest (Multiple PDFs)",
        description="Langchain+Chatgpt4omini Pretest UI for WELL Certification with multi-PDF support."
    )
    
    iface.launch()

if __name__ == "__main__":
    main()
