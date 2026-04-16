"""Baseline benchmark: sqlite-vec brute-force at N=10K to N=1M.

Measures raw ANN latency of sqlite-vec to compare against snapvec
IVF-PQ at the same scales. Uses the same FIQA/BGE-small embeddings.
"""
from __future__ import annotations

import os
import sqlite3
import struct
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

try:
    import sqlite_vec
except ImportError:
    raise SystemExit("pip install sqlite-vec")

CORPUS_CANDIDATES = [
    Path("experiments/.cache_fiqa_corpus_bge_small.npy"),
    Path("experiments/.cache_fiqa_bge_small.npy"),
]
QUERIES_PATH = Path("experiments/.cache_fiqa_queries_bge_small.npy")

N_QUERIES = 200
KK = 10


def find_corpus() -> Path:
    for p in CORPUS_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("Missing FIQA corpus .npy")


def serialize(vec: NDArray[np.float32]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def augment_corpus(
    base: NDArray[np.float32], target_n: int, rng: np.random.Generator,
) -> NDArray[np.float32]:
    if target_n <= len(base):
        return base[:target_n]
    copies_needed = target_n - len(base)
    indices = rng.integers(0, len(base), size=copies_needed)
    jitter = rng.normal(0, 0.01, size=(copies_needed, base.shape[1])).astype(np.float32)
    augmented = base[indices] + jitter
    augmented /= np.linalg.norm(augmented, axis=1, keepdims=True) + 1e-12
    return np.vstack([base, augmented])


def brute_topk(
    queries: NDArray[np.float32], corpus: NDArray[np.float32],
) -> NDArray[np.int64]:
    sims = queries @ corpus.T
    idx = np.argpartition(-sims, KK - 1, axis=1)[:, :KK]
    sub = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-sub, axis=1)
    return np.take_along_axis(idx, order, axis=1)


def recall_at_k(pred_ids: list[list[int]], truth: NDArray[np.int64]) -> float:
    hits = sum(len(set(p[:KK]) & set(t.tolist())) for p, t in zip(pred_ids, truth))
    return hits / (len(pred_ids) * KK)


def bench_sqlite_vec(
    corpus: NDArray[np.float32],
    queries: NDArray[np.float32],
    truth: NDArray[np.int64],
) -> None:
    N = len(corpus)
    dim = corpus.shape[1]

    # Create temp database
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Create vec0 table
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_chunks USING vec0(embedding float[{dim}])"
    )

    # Insert vectors
    t0 = perf_counter()
    batch = [(i, serialize(corpus[i])) for i in range(N)]
    conn.executemany(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        batch,
    )
    conn.commit()
    insert_s = perf_counter() - t0

    # Disk size
    disk_mb = os.path.getsize(db_path) / 1024 / 1024

    print(f"  insert:  {insert_s:.1f}s")
    print(f"  disk:    {disk_mb:.1f} MB")

    # Warm up
    for q in queries[:3]:
        conn.execute(
            "SELECT rowid, distance FROM vec_chunks "
            "WHERE embedding MATCH ? AND k = ?",
            (serialize(q), KK),
        ).fetchall()

    # Search
    pred_ids: list[list[int]] = []
    t0 = perf_counter()
    for q in queries:
        rows = conn.execute(
            "SELECT rowid, distance FROM vec_chunks "
            "WHERE embedding MATCH ? AND k = ?",
            (serialize(q), KK),
        ).fetchall()
        pred_ids.append([r[0] for r in rows])
    total_s = perf_counter() - t0
    us_per_q = total_s / len(queries) * 1e6
    rec = recall_at_k(pred_ids, truth)

    print(f"  search:  {us_per_q:,.0f} us/query")
    print(f"  recall:  {rec:.3f}")

    conn.close()
    os.unlink(db_path)


def main() -> None:
    corpus_path = find_corpus()
    if not QUERIES_PATH.exists():
        raise SystemExit(f"Missing queries at {QUERIES_PATH}")

    base_corpus = np.load(corpus_path).astype(np.float32)
    base_corpus /= np.linalg.norm(base_corpus, axis=1, keepdims=True) + 1e-12
    queries_all = np.load(QUERIES_PATH).astype(np.float32)
    queries_all /= np.linalg.norm(queries_all, axis=1, keepdims=True) + 1e-12
    queries = queries_all[:N_QUERIES]

    rng = np.random.default_rng(0)

    print("sqlite-vec Baseline Benchmark")
    print(f"Base corpus: {base_corpus.shape[0]:,} x {base_corpus.shape[1]}")
    print(f"Queries: {len(queries)}")

    for target_n in [10_000, 57_638, 100_000, 200_000, 500_000]:
        print(f"\n{'='*50}")
        print(f"N = {target_n:,}")
        print(f"{'='*50}")

        corpus = augment_corpus(base_corpus, target_n, rng)

        print(f"  Brute-force ground truth...", flush=True)
        t0 = perf_counter()
        truth = brute_topk(queries, corpus)
        print(f"  done in {perf_counter() - t0:.1f}s")

        bench_sqlite_vec(corpus, queries, truth)


if __name__ == "__main__":
    main()
