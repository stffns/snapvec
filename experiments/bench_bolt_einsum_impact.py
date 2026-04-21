"""Measure the real-world impact of PR #59's einsum norm swap.

The PR replaced `np.linalg.norm(X, axis=1)` with
`np.sqrt(np.einsum('ij,ij->i', X, X))` in 7 sites across four index
types, citing a ~4x isolated speedup.  Micro-benchmarks show 2.4x
on (10000, 384) float32; the question is whether that shows up in
the end-to-end paths that call the norm.

This script toggles between the two implementations via a
monkey-patch and measures:

- `SnapIndex.add_batch`  (scalar, no IVF, 1 norm per batch)
- `PQSnapIndex.add_batch`  (scalar, no IVF, 1 norm + 1 rot-norm)
- `ResidualSnapIndex.add_batch`  (scalar, no IVF, 1 norm per batch)
- `IVFPQSnapIndex.add_batch`  (IVF, _preprocess called in chunks)
- `IVFPQSnapIndex.search_batch`  (Q norm once per batch)
"""
from __future__ import annotations

from time import perf_counter

import numpy as np

from snapvec import (
    IVFPQSnapIndex, PQSnapIndex, ResidualSnapIndex, SnapIndex,
)


def clustered(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((16, dim)).astype(np.float32) * 3
    assign = rng.integers(0, 16, size=n)
    jitter = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
    return centers[assign] + jitter


class LegacyNorm:
    """Monkey-patch context: temporarily restore np.linalg.norm for
    axis=1 calls by overriding the einsum version in-place.

    We don't have a clean hook into snapvec to swap algorithms, so
    we instead patch np.einsum to fall back to np.linalg.norm for
    the specific signature the PR uses.
    """

    def __init__(self) -> None:
        self._orig_einsum = np.einsum

    def _patched_einsum(self, *args, **kwargs):
        # Detect the (subscripts, X, X) call shape used by the PR
        if args and args[0] == "ij,ij->i" and len(args) == 3 and args[1] is args[2]:
            # Fall back to (X ** 2).sum(1) which is what np.linalg.norm uses internally
            return (args[1] ** 2).sum(1)
        return self._orig_einsum(*args, **kwargs)

    def __enter__(self) -> "LegacyNorm":
        np.einsum = self._patched_einsum  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        np.einsum = self._orig_einsum  # type: ignore[assignment]


def time_n(fn, n_reps: int = 5) -> float:
    # warm
    fn()
    fn()
    times = []
    for _ in range(n_reps):
        t0 = perf_counter()
        fn()
        times.append(perf_counter() - t0)
    return float(np.median(times) * 1e3)  # ms


def bench_snap_add(dim: int, n: int) -> tuple[float, float]:
    corpus = clustered(n, dim, seed=0)

    def run() -> None:
        idx = SnapIndex(dim=dim, bits=4, seed=0)
        idx.add_batch(list(range(n)), corpus)

    t_einsum = time_n(run)
    with LegacyNorm():
        t_legacy = time_n(run)
    return t_legacy, t_einsum


def bench_pq_add(dim: int, n: int) -> tuple[float, float]:
    corpus = clustered(n, dim, seed=0)

    def run() -> None:
        idx = PQSnapIndex(dim=dim, M=dim // 8, K=64, seed=0)
        idx.fit(corpus)
        idx.add_batch(list(range(n)), corpus)

    t_einsum = time_n(run)
    with LegacyNorm():
        t_legacy = time_n(run)
    return t_legacy, t_einsum


def bench_residual_add(dim: int, n: int) -> tuple[float, float]:
    corpus = clustered(n, dim, seed=0)

    def run() -> None:
        idx = ResidualSnapIndex(dim=dim, b1=3, b2=3, seed=0)
        idx.add_batch(list(range(n)), corpus)

    t_einsum = time_n(run)
    with LegacyNorm():
        t_legacy = time_n(run)
    return t_legacy, t_einsum


def bench_ivfpq_search_batch(dim: int, n: int, B: int) -> tuple[float, float]:
    corpus = clustered(n, dim, seed=0)
    queries = clustered(B, dim, seed=1)
    idx = IVFPQSnapIndex(
        dim=dim, nlist=32, M=dim // 8, K=64, normalized=False, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(n)), corpus)

    def run() -> None:
        idx.search_batch(queries, k=10, nprobe=8)

    t_einsum = time_n(run)
    with LegacyNorm():
        t_legacy = time_n(run)
    return t_legacy, t_einsum


def main() -> None:
    dim, n = 384, 10_000
    print(f"Impact of einsum-based row norm vs np.linalg.norm (dim={dim}, n={n}):")
    print()
    print(f"{'path':<40} {'legacy ms':>11} {'einsum ms':>11} {'delta':>8}")
    print("-" * 75)

    rows = [
        ("SnapIndex.add_batch", bench_snap_add(dim, n)),
        ("PQSnapIndex.add_batch (+fit)", bench_pq_add(dim, n)),
        ("ResidualSnapIndex.add_batch", bench_residual_add(dim, n)),
        ("IVFPQSnapIndex.search_batch B=256", bench_ivfpq_search_batch(dim, n, 256)),
    ]
    for label, (legacy_ms, einsum_ms) in rows:
        delta_pct = (einsum_ms - legacy_ms) / legacy_ms * 100
        print(
            f"{label:<40} {legacy_ms:>11.2f} {einsum_ms:>11.2f}  {delta_pct:+6.1f}%"
        )


if __name__ == "__main__":
    main()
