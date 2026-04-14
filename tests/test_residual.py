"""Tests for ResidualSnapIndex — two-stage Lloyd-Max TurboQuant."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snapvec import ResidualSnapIndex, SnapIndex


def _unit_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _brute_topk(q: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(q @ c.T), axis=1)[:, :k]


def test_basic_build_search() -> None:
    corpus = _unit_gaussian(500, 64, seed=1)
    idx = ResidualSnapIndex(dim=64, b1=3, b2=3, seed=0, normalized=True)
    idx.add_batch(list(range(500)), corpus)
    assert len(idx) == 500
    # Query an existing vector: top-1 must be itself.
    hits = idx.search(corpus[0], k=1)
    assert hits[0][0] == 0


def test_rerank_matches_full() -> None:
    rng = np.random.default_rng(2)
    corpus = _unit_gaussian(800, 128, seed=2)
    queries = _unit_gaussian(20, 128, seed=3)
    idx = ResidualSnapIndex(dim=128, b1=3, b2=3, seed=0, normalized=True)
    idx.add_batch(list(range(800)), corpus)
    for q in queries:
        full = [h[0] for h in idx.search(q, k=10, rerank_M=None)]
        rerank = [h[0] for h in idx.search(q, k=10, rerank_M=200)]
        # M=200 over an 800-vector corpus should nearly always match.
        common = len(set(full) & set(rerank))
        assert common >= 8, f"rerank diverged: {common}/10"


@pytest.mark.parametrize("b1, b2", [(2, 2), (3, 3), (4, 3), (4, 4)])
def test_recall_beats_or_matches_uniform_b1(
    b1: int, b2: int
) -> None:
    """At bit budget (b1+b2) >= b1, residual recall should be at least
    as good as uniform at b1 bits on easy synthetic data."""
    corpus = _unit_gaussian(1000, 64, seed=4)
    queries = _unit_gaussian(50, 64, seed=5)
    truth = _brute_topk(queries, corpus, 10)

    uni = SnapIndex(dim=64, bits=b1, seed=0, normalized=True)
    uni.add_batch(list(range(1000)), corpus)
    res = ResidualSnapIndex(dim=64, b1=b1, b2=b2, seed=0, normalized=True)
    res.add_batch(list(range(1000)), corpus)

    def recall(pred_fn) -> float:
        hits = 0
        for q, t in zip(queries, truth):
            pred = [h[0] for h in pred_fn(q)]
            hits += len(set(pred) & set(t.tolist()))
        return hits / (len(queries) * 10)

    r_uni = recall(lambda q: uni.search(q, k=10))
    r_res = recall(lambda q: res.search(q, k=10))
    assert r_res >= r_uni - 0.02, (
        f"residual {r_res:.3f} much worse than uniform-b1 {r_uni:.3f}"
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    corpus = _unit_gaussian(200, 64, seed=6)
    idx = ResidualSnapIndex(dim=64, b1=3, b2=3, seed=42, normalized=True)
    idx.add_batch([f"id_{i}" for i in range(200)], corpus)

    path = tmp_path / "idx.snpr"
    idx.save(path)
    reloaded = ResidualSnapIndex.load(path)

    assert len(reloaded) == 200
    assert reloaded.b1 == 3 and reloaded.b2 == 3
    assert reloaded.seed == 42
    # Same query → same top-k order on both.
    q = corpus[5]
    a = idx.search(q, k=5)
    b = reloaded.search(q, k=5)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-5)


def test_unnormalized_scoring_scales_with_norm() -> None:
    """When normalized=False, scores should be scaled by per-vector norm."""
    rng = np.random.default_rng(9)
    base = rng.standard_normal((50, 32)).astype(np.float32)
    scales = np.linspace(0.5, 5.0, 50).astype(np.float32)
    corpus = base * scales[:, None]
    idx = ResidualSnapIndex(dim=32, b1=3, b2=3, seed=0, normalized=False)
    idx.add_batch(list(range(50)), corpus)
    # Query = largest-scale vector — should rank it #1.
    q = corpus[-1]
    hits = idx.search(q, k=1)
    assert hits[0][0] == 49


def test_invalid_bits_raises() -> None:
    with pytest.raises(ValueError):
        ResidualSnapIndex(dim=64, b1=5, b2=3)
    with pytest.raises(ValueError):
        ResidualSnapIndex(dim=64, b1=3, b2=1)


def test_stats_shape() -> None:
    idx = ResidualSnapIndex(dim=64, b1=3, b2=3, normalized=True)
    idx.add_batch(list(range(10)), _unit_gaussian(10, 64, seed=0))
    s = idx.stats()
    assert s["n"] == 10 and s["b1"] == 3 and s["b2"] == 3
    assert s["bytes_per_vec"] == 2 * s["padded_dim"]
