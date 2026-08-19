# prod-rag

A retrieval-augmented generation system over a corpus of machine-learning papers.
Ask a question in plain English; the system finds the relevant passages across
the PDFs and has Gemini answer **only** from what it retrieved, citing the source
file and page.

It combines two retrieval strategies rather than relying on embeddings alone:
dense vector search catches paraphrases, BM25 catches exact terms (model names,
symbols, acronyms), and a cross-encoder reranks the fused candidates so the
strongest passages reach the model.

## Architecture

```
corpus/*.pdf
     |
     |  PyMuPDF — parse to per-page documents
     v
  chunk         RecursiveCharacterTextSplitter (512 chars, 120 overlap)
     |
     |  embed locally — sentence-transformers/all-MiniLM-L6-v2 (no API calls)
     v
  chroma-db/    persisted vector index                          [ rag.py ]
     |
     +---------------------------+
     |                           |
  dense retrieval            BM25 retrieval
  (semantic, top-k)          (keyword, top-k)
     |                           |
     +------------+--------------+
                  |  EnsembleRetriever — reciprocal rank fusion
                  v
              rerank            BAAI/bge-reranker-base cross-encoder
                  |             scores each candidate against the question,
                  |             keeps the best top_n
                  v
              Gemini            gemini-3.6-flash, answers from context only
                  |
                  v
          answer + sources (filename + page)      [ rag_advanced.py / app.py ]
```

Both the embedding model and the reranker run locally on your machine. The only
network call is the final generation step.

## Files

| File | Purpose |
| --- | --- |
| `rag.py` | Builds the Chroma index from the PDFs in `corpus/`. Run once. |
| `rag_advanced.py` | Hybrid retrieval + reranking + Gemini. Importable, plus a CLI. |
| `app.py` | Streamlit UI with streaming answers and source citations. |
| `evaluate.py` | Retrieval quality harness — recall@k against known sources. |

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it manages Python itself) and Python 3.12,
pinned in `.python-version` and downloaded automatically.

```bash
uv sync                 # create .venv and install dependencies
cp .env.example .env    # then add your GOOGLE_API_KEY
```

Get a Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
`.env` is gitignored — never commit it.

Not using uv? `pip install -r requirements.txt` against Python 3.11+ works too.

### Add your corpus

`corpus/` is **not** included in this repository — the papers may be copyrighted.
Create it and add your own PDFs:

```bash
mkdir -p corpus
# drop any PDFs in here (this project was developed against ~30 arXiv ML papers)
```

### Build the index

```bash
uv run rag.py
```

This parses every PDF, chunks it, embeds the chunks locally and persists the
result to `chroma-db/`. The first run downloads the embedding model (~90MB).
Roughly 30 papers produce about 4,800 chunks. `chroma-db/` is gitignored, so
each clone builds its own index.

## Running

### CLI

```bash
uv run rag_advanced.py
```

Loads the retriever once, then loops on questions. Blank line quits.

```
> What problem does batch normalization solve?

Internal covariate shift — the change in the distribution of network
activations caused by parameter updates during training...

Sources:
  - 1502.03167v3.pdf p.2
  - 1502.03167v3.pdf p.1
```

### Streamlit app

```bash
uv run streamlit run app.py
```

Opens at http://localhost:8501. The answer streams in token by token, with each
source passage listed underneath in an expander. The retriever is built once and
held in `@st.cache_resource`, so it is not rebuilt on every interaction — only
the first question pays the model-loading cost. `fetch_k` (candidates per
retriever) and `top_n` (chunks kept after reranking) are adjustable in the sidebar;
changing either rebuilds the cached retriever.

## Evaluation

```bash
uv run evaluate.py                  # recall@1, @3, @5, @10
uv run evaluate.py --k 5 --fetch-k 40
```

`evaluate.py` holds 16 question/expected-source pairs — one per paper — and
measures how often the correct PDF appears in the top-k retrieved chunks. It
reports the hybrid retriever and the reranked pipeline side by side, so the
reranker's contribution is visible. Retrieval only: no API key needed.

```
Recall@k        Hybrid   Hybrid + rerank
----------------------------------------
recall@1          0.94              0.94
recall@3          1.00              1.00
recall@5          1.00              1.00
recall@10         1.00              1.00
```

Read these numbers with the benchmark in mind: each question uses vocabulary
specific to its paper, and the corpus spans distinct topics, so source
attribution is close to saturated. It is a regression check that retrieval is
wired up correctly, not a hard benchmark. To make it discriminating, add
questions whose answers span several papers, or questions phrased without the
paper's own terminology.

## Configuration

Defaults live at the top of the modules:

| Setting | Where | Default |
| --- | --- | --- |
| Chunk size / overlap | `rag.py` | 512 / 120 |
| Embedding model | `rag.py`, `rag_advanced.py` | `all-MiniLM-L6-v2` |
| Reranker | `rag_advanced.py` | `BAAI/bge-reranker-base` |
| Generation model | `rag_advanced.py` | `gemini-3.6-flash` |
| Dense/BM25 fusion weights | `rag_advanced.py` | 0.5 / 0.5 |

Changing the chunking or embedding model requires rebuilding the index
(`uv run rag.py`).
