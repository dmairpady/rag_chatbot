# ingest.py
import os
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(file_path: str) -> str:
    _, ext = os.path.splitext(file_path.lower())
    if ext not in ['.pdf', '.docx', '.txt']:
        raise ValueError("Unsupported format. Use .pdf, .docx, or .txt")
    full_text = ""
    if ext == '.pdf':
        reader = PdfReader(file_path)
        for p in reader.pages:
            t = p.extract_text()
            if t:
                full_text += t + "\n"
    elif ext == '.docx':
        doc = Document(file_path)
        for para in doc.paragraphs:
            if para.text:
                full_text += para.text + "\n"
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
    return full_text

def split_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
