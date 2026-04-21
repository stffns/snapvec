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
    with pytest.raises(RuntimeError, match="already called"):
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
    assert s["bytes_per_vec"] == 8  # M uint8 codes, normalized=True (no norms)

    # And unnormalized mode reports the extra 4-byte norm per vector.
    idx2 = PQSnapIndex(dim=32, M=8, K=16, normalized=False)
    idx2.fit(corpus)
    idx2.add_batch(list(range(50)), corpus)
    assert idx2.stats()["bytes_per_vec"] == 12


def test_double_fit_raises() -> None:
    corpus = _unit_gaussian(100, 32, seed=0)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    with pytest.raises(RuntimeError, match="already called"):
        idx.fit(corpus)


def test_search_validates_query_and_k() -> None:
    corpus = _unit_gaussian(100, 32, seed=0)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(100)), corpus)
    # Zero-norm query returns [] (matching SnapIndex).
    assert idx.search(np.zeros(32, dtype=np.float32), k=5) == []
    # k < 1 raises.
    with pytest.raises(ValueError, match="k must be"):
        idx.search(corpus[0], k=0)


def test_add_batch_validates_lengths() -> None:
    corpus = _unit_gaussian(100, 32, seed=0)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    with pytest.raises(ValueError, match="same length"):
        idx.add_batch([0, 1], corpus[:3])
    with pytest.raises(ValueError, match="shape"):
        idx.add_batch([0], np.zeros((1, 16), dtype=np.float32))


def test_numeric_id_roundtrip(tmp_path: Path) -> None:
    corpus = _unit_gaussian(50, 32, seed=0)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(50)), corpus)
    path = tmp_path / "x.snpq"
    idx.save(path)
    reloaded = PQSnapIndex.load(path)
    assert reloaded._ids[0] == 0 and isinstance(reloaded._ids[0], int)
    assert reloaded._ids[-1] == 49


def test_id_too_long_raises(tmp_path: Path) -> None:
    corpus = _unit_gaussian(50, 32, seed=0)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(["x" * 70000] + list(range(49)), corpus)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        idx.save(tmp_path / "x.snpq")


def test_kmeans_pp_init_duplicated_points() -> None:
    """Degenerate corpus (many duplicates) must not crash the k-means++
    fallback to uniform sampling."""
    n, d = 100, 16
    corpus = np.tile(np.eye(d, dtype=np.float32)[:5], (n // 5, 1))
    corpus = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    idx = PQSnapIndex(dim=d, M=4, K=16, normalized=True, seed=0)
    idx.fit(corpus)  # should not raise even though most pairs are identical
    idx.add_batch(list(range(n)), corpus)
    assert len(idx) == n


# ──────────────────────────────────────────────────────────────────── #
# OPQ rotation                                                           #
# ──────────────────────────────────────────────────────────────────── #


def test_opq_fit_stores_rotation() -> None:
    corpus = _unit_gaussian(500, 32, seed=1)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True, use_opq=True, seed=0)
    assert idx._opq_rotation is None
    idx.fit(corpus)
    assert idx._opq_rotation is not None
    R = idx._opq_rotation
    assert R.shape == (32, 32)
    # R is orthogonal: R^T R = I
    RT_R = R.T @ R
    np.testing.assert_allclose(RT_R, np.eye(32), atol=1e-4)


def test_opq_and_rht_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        PQSnapIndex(dim=32, M=8, K=16, use_opq=True, use_rht=True)


def test_opq_round_trip_preserves_rotation(tmp_path: Path) -> None:
    corpus = _unit_gaussian(500, 32, seed=2)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True, use_opq=True, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(500)), corpus)
    path = tmp_path / "opq.snpq"
    idx.save(path)
    loaded = PQSnapIndex.load(path)
    assert loaded.use_opq is True
    assert loaded._opq_rotation is not None
    np.testing.assert_array_equal(loaded._opq_rotation, idx._opq_rotation)
    # Search results match (ids exactly, scores within float32 noise
    # -- exact equality would be flaky across BLAS implementations).
    q = corpus[0]
    before = idx.search(q, k=5)
    after = loaded.search(q, k=5)
    assert [h[0] for h in before] == [h[0] for h in after]
    np.testing.assert_allclose(
        [h[1] for h in before], [h[1] for h in after], atol=1e-5,
    )


def test_opq_search_determinism() -> None:
    """Two fits with same seed produce identical OPQ rotations and
    matching search results."""
    corpus = _unit_gaussian(500, 32, seed=3)
    idx1 = PQSnapIndex(dim=32, M=8, K=16, normalized=True, use_opq=True, seed=7)
    idx2 = PQSnapIndex(dim=32, M=8, K=16, normalized=True, use_opq=True, seed=7)
    idx1.fit(corpus)
    idx2.fit(corpus)
    np.testing.assert_array_equal(idx1._opq_rotation, idx2._opq_rotation)
    idx1.add_batch(list(range(500)), corpus)
    idx2.add_batch(list(range(500)), corpus)
    q = _unit_gaussian(1, 32, seed=30)[0]
    hits1 = idx1.search(q, k=5)
    hits2 = idx2.search(q, k=5)
    assert [h[0] for h in hits1] == [h[0] for h in hits2]
    np.testing.assert_allclose(
        [h[1] for h in hits1], [h[1] for h in hits2], atol=1e-5,
    )
