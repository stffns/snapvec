"""Shared k-means primitives for the PQ / IVF-PQ code paths.

Extracted from ``_pq.py`` and ``_ivfpq.py`` so both index types import
the same implementation and any fix (dead-cluster handling, assignment
routine) applies in one place.

Scope is intentionally narrow: k-means++ init + plain Lloyd with MSE
objective, plus an explicit squared-L2 assignment helper that both
indexes use at query time to match the metric used during training.
"""
from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def kmeans_pp_init(
    X: NDArray[np.float32], K: int, rng: np.random.Generator,
) -> NDArray[np.float32]:
    """K-means++ seeding on ``X`` with ``K`` centers.

    Falls back to uniform sampling when every point is already zero
    distance from the selected centres — e.g. duplicated points or
    fewer unique values than ``K``.  The guard prevents the
    ``rng.choice(..., p=[0, 0, ...])`` ValueError.
    """
    n = X.shape[0]
    centers = [X[int(rng.integers(n))]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, K):
        total = d2.sum()
        probs = d2 / total if total > 1e-12 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=probs))
        centers.append(X[nxt])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    return np.stack(centers).astype(np.float32)


def kmeans_mse(
    X: NDArray[np.float32], K: int, n_iters: int = 15, seed: int = 0,
) -> NDArray[np.float32]:
    """Plain Lloyd k-means under squared Euclidean distance.

    Dead-cluster handling: when multiple clusters become empty in the
    same iteration, each is reseeded to a *different* far-from-its-
    current-assignment point rather than all to the single worst-fit
    point — otherwise reseeded clusters collapse into duplicates.
    """
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(X, K, rng)
    x_sq = (X ** 2).sum(1, keepdims=True)
    for _ in range(n_iters):
        d2 = x_sq - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        asn = d2.argmin(1)
        newC = np.empty_like(C)
        dead_ks: list[int] = []
        for k in range(K):
            m = asn == k
            if m.any():
                newC[k] = X[m].mean(0)
            else:
                dead_ks.append(k)
        if dead_ks:
            # Pick distinct far-from-assignment points for each dead k.
            # argsort(-min_d2) gives the candidates in descending distance.
            min_d2 = d2.min(1)
            order = np.argsort(-min_d2)
            used: set[int] = set()
            picks: list[int] = []
            for idx in order:
                idx_int = int(idx)
                if idx_int not in used:
                    picks.append(idx_int)
                    used.add(idx_int)
                if len(picks) == len(dead_ks):
                    break
            for k, pick in zip(dead_ks, picks):
                newC[k] = X[pick]
        if np.allclose(newC, C, atol=1e-5):
            return newC
        C = newC
    return C


def assign_l2(
    X: NDArray[np.float32], C: NDArray[np.float32],
) -> NDArray[np.int64]:
    """Hard-assign every row in X to its nearest centroid (squared L2)."""
    d2 = (X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    return cast("NDArray[np.int64]", d2.argmin(1))


def probe_scores_l2_monotone(
    coarse: NDArray[np.float32], q: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Score coarse centroids so that ranking matches squared-L2 order.

    Squared L2 between query ``q`` and centroid ``c`` is
    ``‖q‖² − 2⟨q, c⟩ + ‖c‖²``.  Since ``‖q‖²`` is constant across
    centroids, minimising L2² is equivalent to *maximising*
    ``2⟨q, c⟩ − ‖c‖²`` — that's what we return.

    Using the plain dot product ``⟨q, c⟩`` here would be wrong because
    coarse centroids are means of unit vectors (so their norms vary),
    and the index was built with L2 assignment.  Mixing metrics at
    probe time gives slightly lower recall, especially with uneven
    cluster sizes.
    """
    # np.float32(2.0) guards against NEP 50-era numpy upcasting a
    # Python '2.0' scalar to float64 here; on numpy >= 2.0 this is a
    # no-op, on older numpy it keeps the return dtype matching the
    # annotation.
    return cast(
        "NDArray[np.float32]",
        np.float32(2.0) * (coarse @ q) - (coarse ** 2).sum(1),
    )


__all__ = [
    "kmeans_pp_init",
    "kmeans_mse",
    "assign_l2",
    "probe_scores_l2_monotone",
]
