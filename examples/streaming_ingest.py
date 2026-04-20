"""Streaming ingest: grow an index with many small batches.

SnapIndex is training-free so you can start searching after the first
``add_batch`` call and keep appending.  PQ-based indices need an initial
``fit(sample)`` before adding vectors.

Run with: python examples/streaming_ingest.py
"""
from __future__ import annotations

import numpy as np

from snapvec import SnapIndex


def main() -> None:
    rng = np.random.default_rng(0)
    dim, batch_size, n_batches = 64, 100, 10

    idx = SnapIndex(dim=dim, bits=4, seed=0)

    for batch in range(n_batches):
        vecs = rng.standard_normal((batch_size, dim)).astype(np.float32)
        base = batch * batch_size
        ids = list(range(base, base + batch_size))
        idx.add_batch(ids, vecs)
        print(f"after batch {batch + 1:2d}: n={len(idx):5d}")

    assert len(idx) == batch_size * n_batches

    query = rng.standard_normal(dim).astype(np.float32)
    hits = idx.search(query, k=3)
    print(f"\ntop-3 for a fresh query: {[h[0] for h in hits]}")


if __name__ == "__main__":
    main()
