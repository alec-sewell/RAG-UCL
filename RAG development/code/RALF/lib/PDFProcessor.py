import io
import os
from typing import List, Tuple

import pandas as pd
import pymupdf as fitz
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFProcessor:
    """Handles processing of PDF files, including text, image, and table extraction."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def get_file_signatures(self, pdf_file_objs) -> List[Tuple[str, int, float]]:
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

    def read_pdf_pages(self, pdf_path: str):
        """Yield (page_number, text) for each page using PyMuPDF."""
        doc = fitz.open(pdf_path)
        try:
            for i, page in enumerate(doc, start=1):
                yield i, page.get_text()
        finally:
            doc.close()

    def chunk_pdf_with_metadata(self, pdf_path: str):
        """Split each page to chunks; return a list[Document] with page + file metadata."""
        docs = []
        file_name = os.path.basename(pdf_path)
        for page_num, page_text in self.read_pdf_pages(pdf_path):
            page_docs = self.text_splitter.create_documents(
                texts=[page_text],
                metadatas=[{
                    "source_file": file_name,
                    "page": page_num,
                }],
            )
            for idx, d in enumerate(page_docs):
                d.metadata["chunk"] = idx
            docs.extend(page_docs)
        return docs

    def extract_images_and_tables(self, pdf_path: str):
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