import os

import filetype
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI

from .DiskCacheEmbeddings import DiskCacheEmbeddings
from .RateLimitedEmbeddings import RateLimitedEmbeddings
from .Config import Config
from .PDFProcessor import PDFProcessor
from .TextProcessor import TextProcessor


class RAGManager:
    """Manages the RAG pipeline, including indexing and querying."""

    def __init__(self, config: Config):
        self.config = config
        self.pdf_processor = PDFProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        self.embeddings = self._init_embeddings()

        # In-memory session cache
        self.signatures = None
        self.hybrid_retriever = None
        self.dense_retriever = None
        self.sparse_retriever = None
        self.images = None
        self.tables = None
        self.texts_count = 0

    def _init_embeddings(self):
        base_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        rate_limited_embeddings = RateLimitedEmbeddings(inner=base_embeddings, batch_size=64)
        embeddings = DiskCacheEmbeddings(inner=rate_limited_embeddings, db_path=self.config.EMBEDDING_CACHE_PATH)
        print(f"[CACHE] Using persistent disk cache for embeddings at: {self.config.EMBEDDING_CACHE_PATH}")
        return embeddings

    def _signatures_changed(self, new_sigs) -> bool:
        return self.signatures != new_sigs

    def build_or_reuse_index(self, pdf_files_list):
        sigs = self.pdf_processor.get_file_signatures(pdf_files_list)
        if self.hybrid_retriever is None or self._signatures_changed(sigs):
            processing_log = "--- PDF Processing Log ---\n"
            per_file_docs = []
            all_images, all_tables = [], []

            if not pdf_files_list:
                self.signatures = None
                self.hybrid_retriever = None
                return processing_log + "No PDFs uploaded.\n"

            for f in pdf_files_list:
                pdf_path = f.name
                try:
                    mime = filetype.guess_mime(pdf_path)
                except Exception as e:
                    mime = None
                    processing_log += f"  - WARNING: Could not determine MIME type for {os.path.basename(pdf_path)}: {e}\n"

                if mime != "application/pdf":
                    processing_log += f"  - SKIPPING: '{os.path.basename(pdf_path)}' is not a PDF (detected as {mime}).\n"
                    continue

                processing_log += f"Processing PDF: {os.path.basename(pdf_path)}\n"
                try:
                    file_docs = self.pdf_processor.chunk_pdf_with_metadata(pdf_path)
                    file_docs = [d for d in file_docs if len(d.page_content.strip()) > 60]
                    images_from_pdf, tables_from_pdf = self.pdf_processor.extract_images_and_tables(pdf_path)

                    per_file_docs.extend(file_docs)
                    all_images.extend(images_from_pdf)
                    all_tables.extend(tables_from_pdf)

                    processing_log += (
                        f"  - Chunks (center) created: {len(file_docs)}\n"
                        f"  - Images extracted: {len(images_from_pdf)}\n"
                        f"  - Tables extracted: {len(tables_from_pdf)}\n"
                    )
                except Exception as e:
                    processing_log += f"  - ERROR processing {os.path.basename(pdf_path)}: {e}\n"
                    continue

            if not per_file_docs:
                self.signatures = None
                self.hybrid_retriever = None
                return processing_log + "No text content could be extracted from any of the provided PDFs.\n"

            ctx_docs = TextProcessor.contextualize_documents(per_file_docs, left=1, right=1)
            processing_log += f"\n--- Combined Processing Summary ---\n"
            processing_log += f"Context-augmented vectors: {len(ctx_docs)}\n"
            processing_log += f"Images total: {len(all_images)}\n"
            processing_log += f"Tables total: {len(all_tables)}\n\n"

            vectorstore = FAISS.from_documents(ctx_docs, self.embeddings)

            N = max(12, min(60, (len(ctx_docs) // 250) * 8))
            fetch_k = min(200, N * 6)

            dense_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": N, "fetch_k": fetch_k, "lambda_mult": 0.4}
            )
            sparse_retriever = BM25Retriever.from_documents(ctx_docs)
            sparse_retriever.k = min(60, N)

            hybrid = EnsembleRetriever(retrievers=[dense_retriever, sparse_retriever], weights=[0.6, 0.4])

            mqr = MultiQueryRetriever.from_llm(
                retriever=hybrid,
                llm=self.llm,
                prompt=self.config.MQR_PROMPT.partial(n=str(self.config.MQR_NUM)),
            )

            ce = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
            reranker = CrossEncoderReranker(model=ce, top_n=10)
            compression_retriever = ContextualCompressionRetriever(base_retriever=mqr, base_compressor=reranker)

            self.signatures = sigs
            self.hybrid_retriever = compression_retriever
            self.dense_retriever = dense_retriever
            self.sparse_retriever = sparse_retriever
            self.images = all_images
            self.tables = all_tables
            self.texts_count = len(ctx_docs)

            processing_log += "(Index built)\n"
            return processing_log

        return "(Reusing cached index — PDFs unchanged)\n"

    def _rag_pipeline(self, query: str):
        relevant_docs = self.hybrid_retriever.invoke(query)

        generated_alts = []
        try:
            base_r = getattr(self.hybrid_retriever, "base_retriever", None)
            mqr_like = base_r if base_r is not None else self.hybrid_retriever
            if hasattr(mqr_like, "llm_chain"):
                alt_text = mqr_like.llm_chain.invoke({"question": query})
                if isinstance(alt_text, dict) and "text" in alt_text:
                    alt_text = alt_text["text"]
                if isinstance(alt_text, str):
                    generated_alts = [line.strip() for line in alt_text.splitlines() if line.strip()]
        except Exception:
            pass

        log = "Multi-Query Expansion + Cross-Encoder Re-rank:\n"
        log += f"Original query: {query}\n"
        if generated_alts:
            log += "Generated alternates:\n" + "\n".join(f"- {q}" for q in generated_alts) + "\n"
        log += "\nRelevant chunks (after re-rank):\n"

        retrieved_chunks = []
        context = ""
        for i, doc in enumerate(relevant_docs, 1):
            md = getattr(doc, "metadata", {}) or {}
            center_preview = md.get("center_preview", doc.page_content[:200])
            retrieved_chunks.append(center_preview)

            context += md.get("center_preview", "") + "\n\n"

            score = md.get("score") or md.get("relevance_score")
            score_str = f" (score={score:.4f})" if isinstance(score, (int, float)) else ""
            log += (
                f"Chunk {i}{score_str} "
                f"[{md.get('source_file','?')} "
                f"center={md.get('center_index','?')} "
                f"win={md.get('window_left_index','?')}-{md.get('window_right_index','?')}] "
                f"center preview: {center_preview[:200]}...\n\n"
            )
        log += f"Number of images in all PDFs: {len(self.images)}\n"
        log += f"Number of tables in all PDFs: {len(self.tables)}\n\n"

        doc_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.config.PROMPT)
        raw = doc_chain.invoke({"context": relevant_docs, "question": query})
        answer = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        return answer, log, retrieved_chunks

    def process_and_query(self, pdf_files_list, query: str):
        """Ensure the index exists/reused, then answer the query."""
        processing_log = self.build_or_reuse_index(pdf_files_list)

        if self.hybrid_retriever is None:
            return "No index available. Please upload valid PDFs.", processing_log, []

        print("\nRAG Pipeline initialized. Running query...")
        result, chunks_log, retrieved_chunks = self._rag_pipeline(query)

        final_log = processing_log + chunks_log
        return result, final_log, retrieved_chunks