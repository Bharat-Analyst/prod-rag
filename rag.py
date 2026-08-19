"""Build the Chroma index from the PDFs in corpus/.

Run this once before using the CLI (rag_advanced.py), the Streamlit app
(app.py) or the evaluation harness (evaluate.py):

    uv run rag.py

Embeddings are computed locally with sentence-transformers/all-MiniLM-L6-v2,
so no API key is needed for this step.
"""

import os
import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_DIR = "corpus"
PERSIST_DIR = "./chroma-db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 120


def load_documents(corpus_dir=CORPUS_DIR):
    """Parse every PDF in corpus/ into per-page Documents."""
    pdfs = sorted(f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {corpus_dir}/ — add some papers first.")

    docs = []
    for name in pdfs:
        # Relative path keeps the `source` metadata short and portable.
        pages = PyMuPDFLoader(os.path.join(corpus_dir, name)).load()
        docs.extend(pages)
        print(f"  {name}: {len(pages)} pages")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_index(chunks, persist_dir=PERSIST_DIR, rebuild=False):
    if rebuild and os.path.exists(persist_dir):
        print(f"Removing existing index at {persist_dir}")
        shutil.rmtree(persist_dir)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )


def main():
    print(f"Loading PDFs from {CORPUS_DIR}/ ...")
    docs = load_documents()

    chunks = split_documents(docs)
    print(f"\n{len(docs)} pages -> {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"\nEmbedding with {EMBED_MODEL} (local, first run downloads the model)...")
    build_index(chunks, rebuild=True)
    print(f"\nDone. Index persisted to {PERSIST_DIR}")


if __name__ == "__main__":
    main()
