"""IVFPQSnapIndex with float16 rerank.

IVF-PQ partitions the corpus into ``nlist`` coarse clusters and only
scans ``nprobe`` of them per query (sub-linear search).  With
``keep_full_precision=True`` + ``rerank_candidates``, the top candidates
are rescored against the stored float16 vectors, which recovers almost
all the recall lost to PQ compression.

Run with: python examples/ivf_pq.py
"""
from __future__ import annotations

import numpy as np

from snapvec import IVFPQSnapIndex


def main() -> None:
    rng = np.random.default_rng(0)
    dim, n_corpus, n_queries = 128, 5000, 20

    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    queries = rng.standard_normal((n_queries, dim)).astype(np.float32)

    idx = IVFPQSnapIndex(
        dim=dim,
        nlist=64,
        M=16,
        K=256,
        normalized=True,
        keep_full_precision=True,
        seed=0,
    )
    idx.fit(corpus[:2500])
    idx.add_batch(list(range(n_corpus)), corpus)

    print(f"IVFPQSnapIndex: n={len(idx)}, nlist=64, M=16")

    truth = (queries @ corpus.T).argmax(axis=1)

    # Baseline: IVF-PQ only.
    pq_hits = [idx.search(q, k=1, nprobe=8)[0][0] for q in queries]
    pq_recall = sum(h == t for h, t in zip(pq_hits, truth)) / n_queries

    # With rerank: candidates pre-selected by PQ, rescored in fp16.
    rerank_hits = [
        idx.search(q, k=1, nprobe=8, rerank_candidates=50)[0][0] for q in queries
    ]
    rerank_recall = sum(h == t for h, t in zip(rerank_hits, truth)) / n_queries

    print(f"top-1 recall (PQ only):        {pq_recall:.2f}")
    print(f"top-1 recall (PQ + fp16 rerank): {rerank_recall:.2f}")


if __name__ == "__main__":
    main()
