"""Determinism tests.

Building an index twice with the same seed and the same inputs must
produce bit-identical outputs.  If this ever breaks, it is almost
always a non-deterministic code path that will also cause subtle
recall drift between runs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from snapvec import IVFPQSnapIndex, PQSnapIndex, ResidualSnapIndex, SnapIndex


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clustered(n: int, dim: int, *, seed: int, n_clusters: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3
    assign = rng.integers(0, n_clusters, size=n)
    jitter = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
    return centers[assign] + jitter


def test_snapindex_save_is_bitwise_deterministic(tmp_path: Path) -> None:
    """Same seed + same inputs => byte-identical .snpv file."""
    vecs = _clustered(200, 64, seed=42)

    def build(path: Path) -> None:
        idx = SnapIndex(dim=64, bits=4, seed=7)
        idx.add_batch(list(range(200)), vecs)
        idx.save(path)

    a = tmp_path / "a.snpv"
    b = tmp_path / "b.snpv"
    build(a)
    build(b)
    assert _sha256(a) == _sha256(b)


def test_pqsnapindex_save_is_bitwise_deterministic(tmp_path: Path) -> None:
    """PQSnapIndex.fit + add_batch with the same seed => same bytes."""
    vecs = _clustered(500, 64, seed=1)

    def build(path: Path) -> None:
        idx = PQSnapIndex(dim=64, M=8, K=16, seed=3)
        idx.fit(vecs)
        idx.add_batch(list(range(500)), vecs)
        idx.save(path)

    a = tmp_path / "a.snpq"
    b = tmp_path / "b.snpq"
    build(a)
    build(b)
    assert _sha256(a) == _sha256(b)


def test_residualsnapindex_save_is_bitwise_deterministic(tmp_path: Path) -> None:
    vecs = _clustered(200, 64, seed=9)

    def build(path: Path) -> None:
        idx = ResidualSnapIndex(dim=64, b1=3, b2=3, seed=11)
        idx.add_batch(list(range(200)), vecs)
        idx.save(path)

    a = tmp_path / "a.snpr"
    b = tmp_path / "b.snpr"
    build(a)
    build(b)
    assert _sha256(a) == _sha256(b)


def test_ivfpqsnapindex_save_is_bitwise_deterministic(tmp_path: Path) -> None:
    """IVF-PQ fit involves kmeans; same seed must still produce same bytes."""
    vecs = _clustered(500, 64, seed=21)

    def build(path: Path) -> None:
        idx = IVFPQSnapIndex(dim=64, nlist=8, M=8, K=16, seed=13)
        idx.fit(vecs)
        idx.add_batch(list(range(500)), vecs)
        idx.save(path)

    a = tmp_path / "a.snpi"
    b = tmp_path / "b.snpi"
    build(a)
    build(b)
    assert _sha256(a) == _sha256(b)


@pytest.mark.parametrize(
    ("index_cls", "ext", "build_kwargs"),
    [
        (SnapIndex, ".snpv", {"bits": 4}),
        (PQSnapIndex, ".snpq", {"M": 8, "K": 16}),
        (ResidualSnapIndex, ".snpr", {"b1": 3, "b2": 3}),
        (IVFPQSnapIndex, ".snpi", {"nlist": 8, "M": 8, "K": 16}),
    ],
)
def test_search_results_are_deterministic(
    tmp_path: Path,
    index_cls: type,
    ext: str,
    build_kwargs: dict[str, object],
) -> None:
    """search() returns identical (id, score) tuples across two builds."""
    vecs = _clustered(300, 32, seed=5)

    def build() -> object:
        idx = index_cls(dim=32, seed=17, **build_kwargs)
        if hasattr(idx, "fit"):
            idx.fit(vecs)
        idx.add_batch(list(range(300)), vecs)
        return idx

    idx_a = build()
    idx_b = build()

    query = vecs[7]
    hits_a = idx_a.search(query, k=5)
    hits_b = idx_b.search(query, k=5)
    assert hits_a == hits_b
