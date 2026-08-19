"""Streamlit UI over the hybrid + reranking RAG pipeline.

    uv run streamlit run app.py
"""

import streamlit as st

from rag_advanced import GEN_MODEL, build_retriever, format_source, get_store, stream_answer

st.set_page_config(page_title="Paper RAG", page_icon="📄")


@st.cache_resource(show_spinner="Loading index, BM25 and reranker (first run is slow)...")
def load_retriever(fetch_k: int, top_n: int):
    """Built once per (fetch_k, top_n) and reused across reruns."""
    return build_retriever(get_store(), fetch_k=fetch_k, top_n=top_n)


with st.sidebar:
    st.header("Retrieval settings")
    fetch_k = st.slider("Candidates per retriever (fetch_k)", 5, 50, 20, step=5)
    top_n = st.slider("Chunks kept after rerank (top_n)", 1, 10, 4)
    st.caption(f"Answers generated with `{GEN_MODEL}`.")

st.title("📄 Paper RAG")
st.caption("Hybrid dense + BM25 retrieval, cross-encoder reranking, Gemini answers.")

question = st.text_input(
    "Question",
    placeholder="e.g. What problem does batch normalization solve?",
)

if question:
    retriever = load_retriever(fetch_k, top_n)

    answer_box = st.empty()
    parts = []
    hits = []
    with st.spinner("Retrieving and generating..."):
        for hits, chunk in stream_answer(question, retriever):
            if chunk:
                parts.append(chunk)
                answer_box.markdown("".join(parts))

    if not parts:
        answer_box.warning("No answer returned.")

    st.divider()
    st.subheader("Sources")
    if hits:
        for doc in hits:
            with st.expander(format_source(doc)):
                st.write(doc.page_content)
    else:
        st.write("No sources retrieved.")
