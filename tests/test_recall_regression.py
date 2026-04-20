"""Recall regression smoke test.

Builds a small clustered synthetic corpus, indexes it with
``IVFPQSnapIndex`` at a realistic config, and asserts
``recall@10`` stays above a loose floor versus float32 brute-force.

Runs in CI on every commit.  Purpose is to catch recall cliffs
introduced by kernel rewrites or dtype changes that unit tests wouldn't
notice.  It is *not* a replacement for a real benchmark -- the corpus
is synthetic and tiny so the signal is qualitative.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from snapvec import IVFPQSnapIndex


def _clustered_corpus(
    n: int, dim: int, *, n_clusters: int, seed: int,
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3
    assign = rng.integers(0, n_clusters, size=n)
    jitter = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
    return centers[assign] + jitter


def _brute_topk(
    queries: NDArray[np.float32], corpus: NDArray[np.float32], k: int,
) -> NDArray[np.int64]:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    sub = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub, axis=1)
    return np.take_along_axis(idx, order, axis=1).astype(np.int64)


def _recall_at_k(
    pred: list[list[int]], truth: NDArray[np.int64], k: int,
) -> float:
    total = 0
    for p, t in zip(pred, truth):
        total += len(set(p[:k]) & set(t[:k].tolist()))
    return total / (len(pred) * k)


def test_ivfpq_recall_stays_above_floor() -> None:
    """IVF-PQ + fp16 rerank must keep recall@10 >= 0.9 on clustered data.

    Parameters mirror a real deployment shape on a small corpus:
    nlist = 4 * sqrt(N), M = dim / 8, K = 256, rerank_candidates = 100.
    If this asserts starts failing we introduced a recall regression
    that will also show up on real embeddings.
    """
    n, dim = 5_000, 128
    k = 10
    corpus = _clustered_corpus(n, dim, n_clusters=32, seed=17)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12

    rng = np.random.default_rng(73)
    queries = rng.standard_normal((50, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    idx = IVFPQSnapIndex(
        dim=dim,
        nlist=32,          # ~4 * sqrt(5000)
        M=16,              # dim / 8
        K=256,
        normalized=True,
        keep_full_precision=True,
        seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(n)), corpus)

    truth = _brute_topk(queries, corpus, k)
    hits = [
        [doc_id for doc_id, _score in idx.search(
            q, k=k, nprobe=8, rerank_candidates=100,
        )]
        for q in queries
    ]
    r = _recall_at_k(hits, truth, k)
    assert r >= 0.90, (
        f"IVFPQ recall@10 regressed to {r:.3f} on the synthetic "
        f"clustered corpus.  Expected >= 0.90."
    )


def test_ivfpq_pq_only_recall_above_floor() -> None:
    """Without rerank, IVF-PQ alone has a much lower recall ceiling on
    aggressive PQ configs.  Keep a *loose* floor (>= 0.45) so we only
    fire on a big regression; the rerank test above is the real recall
    guardrail.
    """
    n, dim = 5_000, 128
    k = 10
    corpus = _clustered_corpus(n, dim, n_clusters=32, seed=17)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12

    rng = np.random.default_rng(73)
    queries = rng.standard_normal((50, dim)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12

    idx = IVFPQSnapIndex(
        dim=dim, nlist=32, M=16, K=256, normalized=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(n)), corpus)

    truth = _brute_topk(queries, corpus, k)
    hits = [
        [doc_id for doc_id, _score in idx.search(q, k=k, nprobe=8)]
        for q in queries
    ]
    r = _recall_at_k(hits, truth, k)
    assert r >= 0.45, (
        f"IVFPQ PQ-only recall@10 regressed to {r:.3f}.  Expected >= 0.45."
    )
