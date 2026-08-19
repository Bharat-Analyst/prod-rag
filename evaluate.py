"""Retrieval evaluation: recall@k against known source documents.

Each case pairs a question with the PDF that should answer it. A case is a
hit@k if that PDF appears among the top-k retrieved chunks.

    uv run evaluate.py                 # default k = 1, 3, 5, 10
    uv run evaluate.py --k 5 --fetch-k 40

Retrieval only — no Gemini calls, so no GOOGLE_API_KEY is required.
"""

import argparse

from rag_advanced import build_retriever, get_store, source_name

# (question, expected source PDF) — one per paper in corpus/.
EVAL_SET = [
    ("What is scaled dot-product attention and why is it divided by the square "
     "root of the key dimension?", "1706.03762.pdf"),
    ("How does the masked language model pre-training objective work?",
     "1810.04805.pdf"),
    ("How do residual connections address the degradation problem in very deep "
     "networks?", "1512.03385.pdf"),
    ("What is internal covariate shift and how does normalizing layer inputs "
     "reduce it?", "1502.03167v3.pdf"),
    ("How are bias-corrected first and second moment estimates computed in this "
     "stochastic optimizer?", "1412.6980.pdf"),
    ("Describe the minimax two-player game between the generator and the "
     "discriminator.", "1406.2661.pdf"),
    ("What is the reparameterization trick used to get a low-variance gradient "
     "estimator for the variational lower bound?", "1312.6114.pdf"),
    ("How is experience replay used to train a Q-network to play Atari games "
     "from raw pixels?", "1312.5602v1.pdf"),
    ("What are visual tokens and how are they used in masked image modeling "
     "pre-training?", "2106.08254v2.pdf"),
    ("Can a stack of feed-forward layers applied over image patches replace "
     "attention on ImageNet?", "2105.02723v1.pdf"),
    ("What are Sylvester normalizing flows and how do they generalize planar "
     "flows for variational inference?", "1803.05649v2.pdf"),
    ("How are quaternion convolutions used to reduce the parameters of a "
     "generative adversarial network?", "2104.09630v2.pdf"),
    ("How does MAP-Elites compare against proximal policy optimization on "
     "deterministic locomotion tasks?", "2009.08438v2.pdf"),
    ("How is kernel density estimation combined with a convolutional LSTM for "
     "probabilistic photovoltaic power forecasting?", "2107.01343v1.pdf"),
    ("What does direct numerical simulation reveal about fog formation in a "
     "moist stably stratified surface layer?", "2012.04128v1.pdf"),
    ("How is human advice incorporated into a policy gradient method for safe "
     "reinforcement learning?", "1808.04096v1.pdf"),
]


def first_rank(docs, expected):
    """1-based rank of the first chunk from `expected`, or None."""
    for i, doc in enumerate(docs, start=1):
        if source_name(doc) == expected:
            return i
    return None


def evaluate(retriever, ks):
    """Run every case through both the hybrid retriever and the reranked one."""
    hybrid = retriever.base_retriever  # the EnsembleRetriever underneath
    rows = []
    for question, expected in EVAL_SET:
        rows.append({
            "question": question,
            "expected": expected,
            "hybrid_rank": first_rank(hybrid.invoke(question), expected),
            "rerank_rank": first_rank(retriever.invoke(question), expected),
        })
    return rows


def recall_at(rows, key, k):
    hits = sum(1 for r in rows if r[key] is not None and r[key] <= k)
    return hits / len(rows)


def print_report(rows, ks):
    width = 58
    print(f"\n{'Question':<{width}}  {'Expected source':<18}  {'Hybrid':>6}  {'Rerank':>6}")
    print("-" * (width + 36))
    for r in rows:
        q = r["question"].replace("\n", " ")
        q = q if len(q) <= width else q[: width - 1] + "…"
        hy = r["hybrid_rank"] or "—"
        rr = r["rerank_rank"] or "—"
        print(f"{q:<{width}}  {r['expected']:<18}  {str(hy):>6}  {str(rr):>6}")
    print("-" * (width + 36))
    print("(numbers are the rank of the first chunk from the expected PDF; — = not retrieved)")

    print(f"\n{'Recall@k':<12}  {'Hybrid':>8}  {'Hybrid + rerank':>16}")
    print("-" * 40)
    for k in ks:
        print(f"{'recall@' + str(k):<12}  "
              f"{recall_at(rows, 'hybrid_rank', k):>8.2f}  "
              f"{recall_at(rows, 'rerank_rank', k):>16.2f}")
    print(f"\n{len(rows)} questions evaluated.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10],
                        help="cutoffs to report recall at (default: 1 3 5 10)")
    parser.add_argument("--fetch-k", type=int, default=30,
                        help="candidates each retriever fetches before fusion")
    args = parser.parse_args()

    ks = sorted(args.k)
    # top_n must cover the largest cutoff, or recall@k would be capped by it.
    print(f"Loading retriever (fetch_k={args.fetch_k}, top_n={max(ks)})...")
    retriever = build_retriever(get_store(), fetch_k=args.fetch_k, top_n=max(ks))

    print(f"Evaluating {len(EVAL_SET)} questions...")
    print_report(evaluate(retriever, ks), ks)


if __name__ == "__main__":
    main()
