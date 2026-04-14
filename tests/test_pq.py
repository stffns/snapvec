"""Tests for PQSnapIndex — product-quantization index with trained codebooks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snapvec import PQSnapIndex, SnapIndex


def _unit_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _clustered(
    n: int, d: int, n_clusters: int = 20, noise: float = 0.2, seed: int = 0,
) -> np.ndarray:
    """Mixture-of-Gaussians data — realistic for embeddings, friendly to PQ."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    x = centers[assign] + noise * rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _brute_topk(q: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(q @ c.T), axis=1)[:, :k]


def _recall(pred: list[list[int]], truth: np.ndarray, k: int) -> float:
    hits = sum(len(set(p[:k]) & set(t[:k].tolist())) for p, t in zip(pred, truth))
    return hits / (len(pred) * k)


def test_fit_add_search_basic() -> None:
    corpus = _unit_gaussian(500, 64, seed=1)
    idx = PQSnapIndex(dim=64, M=16, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(500)), corpus)
    assert len(idx) == 500
    # Query an indexed vector: top-1 should be itself.
    hits = idx.search(corpus[0], k=1)
    assert hits[0][0] == 0


def test_fit_required_before_add() -> None:
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    with pytest.raises(RuntimeError, match="fit"):
        idx.add_batch([0], np.zeros((1, 32), dtype=np.float32))


def test_fit_after_add_raises() -> None:
    corpus = _unit_gaussian(300, 32, seed=0)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch([0], corpus[:1])
    with pytest.raises(RuntimeError, match="invalidate"):
        idx.fit(corpus)


def test_invalid_M_raises() -> None:
    with pytest.raises(ValueError, match="divide"):
        PQSnapIndex(dim=384, M=50)  # 50 does not divide 384


def test_K_bounds() -> None:
    with pytest.raises(ValueError):
        PQSnapIndex(dim=64, M=8, K=1)
    with pytest.raises(ValueError):
        PQSnapIndex(dim=64, M=8, K=300)


def test_not_enough_training_vectors() -> None:
    idx = PQSnapIndex(dim=64, M=8, K=64, normalized=True)
    with pytest.raises(ValueError, match="training vectors"):
        idx.fit(_unit_gaussian(32, 64))  # 32 < K=64


def test_save_load_roundtrip(tmp_path: Path) -> None:
    corpus = _unit_gaussian(200, 64, seed=6)
    idx = PQSnapIndex(dim=64, M=16, K=16, seed=42, normalized=True)
    idx.fit(corpus)
    idx.add_batch([f"id_{i}" for i in range(200)], corpus)
    path = tmp_path / "x.snpq"
    idx.save(path)
    reloaded = PQSnapIndex.load(path)
    assert len(reloaded) == 200
    assert reloaded.M == 16 and reloaded.K == 16 and reloaded.seed == 42
    q = corpus[5]
    a = idx.search(q, k=5)
    b = reloaded.search(q, k=5)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-5)


def test_recall_beats_snapindex_at_matched_storage() -> None:
    """On clustered data, PQSnapIndex at bpv ≈ SnapIndex(bits=2) should win.

    Training-enabled codebooks adapt to cluster structure; fixed
    Lloyd-Max thresholds cannot.  We sanity-check this on synthetic
    clustered data so the test is deterministic and fast.
    """
    d, n_corpus, n_queries = 128, 1500, 100
    corpus = _clustered(n_corpus, d, n_clusters=40, seed=11)
    queries = _clustered(n_queries, d, n_clusters=40, seed=12)
    truth = _brute_topk(queries, corpus, 10)

    # SnapIndex b=2: 128 / 4 = 32 bytes per vector at d=128.
    uni = SnapIndex(dim=d, bits=2, normalized=True, seed=0)
    uni.add_batch(list(range(n_corpus)), corpus)
    r_uni = _recall(
        [[h[0] for h in uni.search(q, k=10)] for q in queries], truth, 10
    )

    # PQSnapIndex M=32, K=256: 32 bytes per vector.
    pq = PQSnapIndex(dim=d, M=32, K=256, normalized=True, seed=0)
    pq.fit(corpus[:1000])
    pq.add_batch(list(range(n_corpus)), corpus)
    r_pq = _recall(
        [[h[0] for h in pq.search(q, k=10)] for q in queries], truth, 10
    )
    assert r_pq > r_uni + 0.05, (
        f"PQ {r_pq:.3f} did not beat SnapIndex b=2 {r_uni:.3f} by 5 pp"
    )


@pytest.mark.parametrize("use_rht", [False, True])
def test_use_rht_toggle(use_rht: bool) -> None:
    """Both toggles should produce a sensible top-k — not necessarily
    the query itself at aggressive compression, but a vector highly
    similar to it (same cluster)."""
    corpus = _clustered(400, 64, n_clusters=40, seed=20)
    idx = PQSnapIndex(dim=64, M=16, K=16, normalized=True, use_rht=use_rht)
    idx.fit(corpus)
    idx.add_batch(list(range(400)), corpus)
    hits = idx.search(corpus[7], k=3)
    top_id = hits[0][0]
    # The hit must be near the query in the original cosine sense.
    assert float(corpus[7] @ corpus[top_id]) > 0.9


def test_unnormalized_scoring_scales_with_norm() -> None:
    rng = np.random.default_rng(9)
    base = _clustered(60, 32, seed=21)
    scales = np.linspace(0.5, 5.0, 60).astype(np.float32)
    corpus = base * scales[:, None]
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=False, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(60)), corpus)
    # Query = largest-scale vector, direction same as unit base; should
    # rank the largest-scale entries highest.
    q = corpus[-1]
    hits = idx.search(q, k=3)
    assert hits[0][0] == 59


def test_delete() -> None:
    corpus = _unit_gaussian(100, 32, seed=3)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(100)), corpus)
    assert idx.delete(50) is True
    assert idx.delete(50) is False
    assert len(idx) == 99
    # The remaining hits should never include the deleted id.
    for q in corpus[:5]:
        hits = idx.search(q, k=10)
        assert 50 not in [h[0] for h in hits]


def test_stats_shape() -> None:
    corpus = _unit_gaussian(50, 32, seed=0)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(50)), corpus)
    s = idx.stats()
    assert s["n"] == 50 and s["M"] == 8 and s["K"] == 16
    assert s["d_sub"] == 4 and s["fitted"] is True
    assert s["bytes_per_vec"] == 8  # M uint8 codes, normalized=True
