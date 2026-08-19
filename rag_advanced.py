"""Hybrid retrieval (dense + BM25) -> cross-encoder rerank -> Gemini answer.

Run the CLI with:

    uv run rag_advanced.py

Requires the Chroma index built by rag.py and GOOGLE_API_KEY in .env.
"""

import os

import dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

# LangChain 1.0 moved some of these into langchain_classic — try both.
try:
    from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

dotenv.load_dotenv()

GEN_MODEL = "gemini-3.6-flash"
PERSIST_DIR = "./chroma-db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

_client = None


def get_client():
    """Lazily create the Gemini client so retrieval-only tools (evaluate.py)
    don't require GOOGLE_API_KEY."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def get_store():
    # Assumes the index was already built by rag.py.
    if not os.path.exists(PERSIST_DIR):
        raise SystemExit(f"No index at {PERSIST_DIR} — run `uv run rag.py` first.")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)


def build_retriever(store, fetch_k=20, top_n=4):
    # 1. Dense (semantic) retriever — meaning-based, from Chroma.
    dense = store.as_retriever(search_kwargs={"k": fetch_k})

    # 2. Sparse (keyword/BM25) retriever — exact-term match.
    #    Rebuild it from the chunks already stored in Chroma (no re-parsing PDFs).
    data = store.get()  # returns documents + metadatas
    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(data["documents"], data["metadatas"])
    ]
    sparse = BM25Retriever.from_documents(docs)
    sparse.k = fetch_k

    # 3. Hybrid = fuse both ranked lists (Reciprocal Rank Fusion).
    hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])

    # 4. Rerank the fused candidates with a cross-encoder; keep the best top_n.
    cross_encoder = HuggingFaceCrossEncoder(model_name=RERANK_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)


def source_name(doc):
    """Filename of the PDF a chunk came from, e.g. '1706.03762.pdf'."""
    return os.path.basename(doc.metadata.get("source", "?"))


def page_number(doc):
    """1-based page number for display (PyMuPDF stores pages 0-based)."""
    page = doc.metadata.get("page")
    return page + 1 if isinstance(page, int) else "?"


def format_source(doc):
    return f"{source_name(doc)} p.{page_number(doc)}"


def build_prompt(query, hits):
    context = "\n\n---\n\n".join(
        f"[{format_source(d)}]\n{d.page_content}" for d in hits
    )
    return (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )


def answer(query, retriever):
    hits = retriever.invoke(query)
    resp = get_client().models.generate_content(
        model=GEN_MODEL, contents=build_prompt(query, hits)
    )
    return resp.text, hits


def stream_answer(query, retriever):
    """Yield (hits, chunk) — hits once up front, then answer text token-by-token.

    Used by the Streamlit app so sources are known before generation finishes.
    """
    hits = retriever.invoke(query)
    yield hits, ""
    for chunk in get_client().models.generate_content_stream(
        model=GEN_MODEL, contents=build_prompt(query, hits)
    ):
        if chunk.text:
            yield hits, chunk.text


if __name__ == "__main__":
    store = get_store()
    print("Building hybrid + reranking retriever (loads reranker model once)...")
    retriever = build_retriever(store)
    print("\nRAG ready. Ask a question (blank line to quit).\n")
    while True:
        q = input("> ").strip()
        if not q:
            break
        text, hits = answer(q, retriever)
        print("\n" + text + "\n\nSources:")
        for d in hits:
            print(f"  - {format_source(d)}")
        print()
