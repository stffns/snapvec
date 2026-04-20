"""Minimal SnapIndex example.

Build a 4-bit scalar-quantized index over 1,000 random vectors and
measure top-1 recall for a handful of queries without any training.

Run with: python examples/quickstart.py
"""
from __future__ import annotations

import numpy as np

from snapvec import SnapIndex


def main() -> None:
    rng = np.random.default_rng(0)
    dim, n_corpus, n_queries = 128, 1000, 5

    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    idx = SnapIndex(dim=dim, bits=4, seed=0)
    idx.add_batch(list(range(n_corpus)), corpus)

    print(f"SnapIndex: n={len(idx)}, dim={dim}, bits=4")

    truth = (queries @ corpus.T).argmax(axis=1)
    hits = [idx.search(q, k=1)[0][0] for q in queries]

    recall_at_1 = sum(h == t for h, t in zip(hits, truth)) / n_queries
    print(f"top-1 recall over {n_queries} queries: {recall_at_1:.2f}")


if __name__ == "__main__":
    main()
