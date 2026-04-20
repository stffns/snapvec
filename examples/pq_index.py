"""PQSnapIndex example.

PQSnapIndex learns per-subspace k-means codebooks from a training sample,
then encodes each vector as M bytes.  Much higher recall than scalar
quantization at the same bytes/vec, but requires a one-time ``fit`` call.

Run with: python examples/pq_index.py
"""
from __future__ import annotations

import numpy as np

from snapvec import PQSnapIndex


def main() -> None:
    rng = np.random.default_rng(0)
    dim, n_corpus, n_queries = 128, 2000, 10

    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    idx = PQSnapIndex(dim=dim, M=16, K=256, normalized=True, seed=0)
    idx.fit(corpus[:1000])  # train codebooks on first half
    idx.add_batch(list(range(n_corpus)), corpus)

    print(f"PQSnapIndex: n={len(idx)}, dim={dim}, M=16, K=256")
    print(f"bytes/vec: {idx.M} (vs {dim * 4} for float32)")

    truth = (queries @ corpus.T).argmax(axis=1)
    hits = [idx.search(q, k=1)[0][0] for q in queries]
    recall_at_1 = sum(h == t for h, t in zip(hits, truth)) / n_queries
    print(f"top-1 recall over {n_queries} queries: {recall_at_1:.2f}")


if __name__ == "__main__":
    main()
