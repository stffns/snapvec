"""Multi-round RHT benchmark on real embeddings.

Uses BGE-small-en-v1.5 (384d) over the SciFact corpus (5k scientific
abstracts) via fastembed.  Ground truth: exact float32 top-k cosine.

We hold out a subset of corpus docs as queries (classic recall@k setup
for an ANN index — we are *not* evaluating IR quality against qrels;
we are evaluating snapvec's approximation vs. exact search).

Run: ``python experiments/bench_multi_round_rht_real.py``
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from datasets import load_dataset  # type: ignore[import-untyped]
from fastembed import TextEmbedding  # type: ignore[import-untyped]

from snapvec import SnapIndex
from snapvec._rotation import padded_dim, rht


CACHE_PATH = Path("experiments/.cache_scifact_bge_small.npy")


# ──────────────────────────────────────────────────────────────────── #
# Data + embeddings                                                     #
# ──────────────────────────────────────────────────────────────────── #

def load_corpus(limit: int = 3000) -> list[str]:
    ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    # title + text, deduped/truncated for stability
    texts = []
    for row in ds.select(range(min(limit, len(ds)))):
        t = (row["title"] + ". " + row["text"]).strip()
        texts.append(t[:2000])
    return texts


def embed(texts: list[str]) -> NDArray[np.float32]:
    if CACHE_PATH.exists():
        arr = np.load(CACHE_PATH)
        if arr.shape[0] == len(texts):
            print(f"  (cache hit: {arr.shape})")
            return arr.astype(np.float32)
    print("  embedding with BGE-small-en-v1.5…")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    # L2-normalize (BGE is cosine-native)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    CACHE_PATH.parent.mkdir(exist_ok=True)
    np.save(CACHE_PATH, vecs)
    print(f"  embedded {vecs.shape}, cached to {CACHE_PATH}")
    return vecs


# ──────────────────────────────────────────────────────────────────── #
# Metrics                                                               #
# ──────────────────────────────────────────────────────────────────── #

def brute_force_topk(
    queries: NDArray[np.float32], corpus: NDArray[np.float32], k: int
) -> NDArray[np.int64]:
    sims = queries @ corpus.T
    return np.argsort(-sims, axis=1)[:, :k]


def recall_at_k(pred: list[list[int]], truth: NDArray[np.int64], k: int) -> float:
    hits = 0
    total = 0
    for p, t in zip(pred, truth):
        hits += len(set(p[:k]) & set(t[:k].tolist()))
        total += k
    return hits / total


def kurtosis_excess(flat: NDArray[np.float32]) -> float:
    m, s = flat.mean(), flat.std()
    if s == 0:
        return 0.0
    z = (flat - m) / s
    return float((z ** 4).mean() - 3.0)


# ──────────────────────────────────────────────────────────────────── #
# Main                                                                  #
# ──────────────────────────────────────────────────────────────────── #

def main() -> None:
    n_corpus = 3000
    n_queries = 300
    k = 10
    bits_list = [2, 3, 4]
    rounds_list = [1, 2, 3]

    print("Loading SciFact corpus…")
    texts = load_corpus(limit=n_corpus + n_queries)
    print(f"  {len(texts)} docs")
    vecs = embed(texts)
    corpus, queries = vecs[:n_corpus], vecs[n_corpus:n_corpus + n_queries]
    print(f"  corpus: {corpus.shape}, queries: {queries.shape}")

    print("\nGround truth (exact float32 top-k):")
    t0 = time.perf_counter()
    truth = brute_force_topk(queries, corpus, k)
    print(f"  computed in {time.perf_counter() - t0:.2f}s")

    print("\nPost-RHT coordinate excess-kurtosis on real embeddings:")
    pdim = padded_dim(corpus.shape[1])
    padded = np.zeros((len(corpus), pdim), dtype=np.float32)
    padded[:, :corpus.shape[1]] = corpus
    raw_kurt = kurtosis_excess(corpus.ravel())
    print(f"  pre-RHT (BGE-small raw):  excess_kurt = {raw_kurt:+.4f}")
    for r in rounds_list:
        rot = rht(padded, seed=0, rounds=r)
        print(f"  rounds={r}:  excess_kurt = {kurtosis_excess(rot.ravel()):+.4f}")

    print("\nrecall@10 vs exact float32 (BGE-small-en-v1.5, SciFact):")
    header = f"  {'bits':>4}  " + "  ".join(f"r={r:<2d}" for r in rounds_list)
    print(header)
    ids = list(range(n_corpus))
    for b in bits_list:
        row = f"  {b:>4d}  "
        for r in rounds_list:
            idx = SnapIndex(
                dim=corpus.shape[1], bits=b, rht_rounds=r,
                normalized=True, seed=0,
            )
            idx.add_batch(ids, corpus)
            pred = [[h[0] for h in idx.search(q, k=k)] for q in queries]
            row += f" {recall_at_k(pred, truth, k):>5.3f}"
        print(row)

    print("\nSearch latency per query (N=3000, d=384):")
    for b in [4]:
        for r in rounds_list:
            idx = SnapIndex(dim=corpus.shape[1], bits=b, rht_rounds=r,
                            normalized=True, seed=0)
            idx.add_batch(ids, corpus)
            _ = idx.search(queries[0], k=k)  # warm cache
            t0 = time.perf_counter()
            for q in queries[:100]:
                idx.search(q, k=k)
            dt = (time.perf_counter() - t0) / 100 * 1e3
            print(f"  bits={b}, rounds={r}: {dt:>5.2f} ms/query")


if __name__ == "__main__":
    main()
