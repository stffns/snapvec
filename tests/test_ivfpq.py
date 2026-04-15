"""Tests for IVFPQSnapIndex — inverted-file + residual PQ."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snapvec import IVFPQSnapIndex, PQSnapIndex


def _unit_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _clustered(
    n: int, d: int, n_clusters: int = 40, noise: float = 0.15, seed: int = 0,
) -> np.ndarray:
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
    # Lower compression + unique-ish data so top-1 == the query itself.
    corpus = _unit_gaussian(500, 64, seed=1)
    idx = IVFPQSnapIndex(dim=64, nlist=8, M=16, K=64, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(500)), corpus)
    assert len(idx) == 500
    hits = idx.search(corpus[0], k=1, nprobe=8)   # full probe → must find self
    assert hits[0][0] == 0


def test_fit_required_before_add_or_search() -> None:
    idx = IVFPQSnapIndex(dim=64, nlist=8, M=8, K=16, normalized=True)
    with pytest.raises(RuntimeError, match="fit"):
        idx.add_batch([0], _unit_gaussian(1, 64))
    with pytest.raises(RuntimeError, match="fit"):
        idx.search(_unit_gaussian(1, 64)[0], k=1)


def test_double_fit_raises() -> None:
    corpus = _unit_gaussian(200, 32, seed=0)
    idx = IVFPQSnapIndex(dim=32, nlist=8, M=8, K=16, normalized=True)
    idx.fit(corpus)
    with pytest.raises(RuntimeError, match="already called"):
        idx.fit(corpus)


def test_nlist_validation() -> None:
    with pytest.raises(ValueError, match="nlist"):
        IVFPQSnapIndex(dim=64, nlist=1, M=8, K=16)


def test_invalid_M_raises() -> None:
    with pytest.raises(ValueError, match="divide"):
        IVFPQSnapIndex(dim=384, nlist=16, M=50)


def test_nprobe_default_and_bounds() -> None:
    corpus = _clustered(500, 64, n_clusters=20, seed=2)
    idx = IVFPQSnapIndex(dim=64, nlist=32, M=16, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(500)), corpus)
    # default nprobe = max(1, nlist // 16) = 2
    assert idx._default_nprobe == 2
    with pytest.raises(ValueError, match="nprobe"):
        idx.search(corpus[0], k=1, nprobe=0)
    with pytest.raises(ValueError, match="nprobe"):
        idx.search(corpus[0], k=1, nprobe=33)


def test_fit_warns_when_n_train_below_30x_nlist() -> None:
    """Standard FAISS sizing is ≥ 30 training samples per cluster.
    Below this, fit() should emit a UserWarning pointing to the
    actual ratio + recommended size."""
    import warnings

    corpus = _unit_gaussian(200, 32, seed=90)
    idx = IVFPQSnapIndex(dim=32, nlist=10, M=8, K=16, normalized=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        idx.fit(corpus)
    assert any("training vectors" in str(w.message) for w in captured), (
        f"no fit() warning raised at n_train=200, nlist=10 "
        f"(ratio 20 < 30); got {[str(w.message) for w in captured]}"
    )


def test_fit_does_not_warn_at_healthy_ratio() -> None:
    import warnings

    corpus = _unit_gaussian(400, 32, seed=91)   # 400 / 8 = 50 ≥ 30
    idx = IVFPQSnapIndex(dim=32, nlist=8, M=8, K=16, normalized=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        idx.fit(corpus)
    assert not any("training vectors" in str(w.message) for w in captured), (
        "unexpected fit() warning at n_train=400, nlist=8 (ratio 50 ≥ 30)"
    )


def test_full_probe_beats_default_nprobe() -> None:
    """With ``nprobe == nlist`` we scan every cluster; recall should
    converge to the PQSnapIndex full-scan baseline (same M/K, same
    underlying codebooks are learned from the same data)."""
    d, n_corpus, n_queries = 128, 1500, 80
    corpus = _clustered(n_corpus, d, n_clusters=30, seed=11)
    queries = _clustered(n_queries, d, n_clusters=30, seed=12)
    truth = _brute_topk(queries, corpus, 10)

    ivf = IVFPQSnapIndex(
        dim=d, nlist=16, M=16, K=64, normalized=True, seed=0,
    )
    ivf.fit(corpus[:1000])
    ivf.add_batch(list(range(n_corpus)), corpus)

    r_full = _recall(
        [[h[0] for h in ivf.search(q, k=10, nprobe=16)] for q in queries],
        truth, 10,
    )
    r_default = _recall(
        [[h[0] for h in ivf.search(q, k=10)] for q in queries],
        truth, 10,
    )
    # Full probe must recall at least as well as default (and usually more).
    assert r_full >= r_default - 0.01


def test_recall_tracks_pq_fullscan() -> None:
    """IVF at ``nprobe=nlist`` visits every cluster, so its recall must
    converge to the PQ full-scan baseline on the same data.  We avoid
    wall-clock assertions here because they are flaky across CI
    runners; the speedup claims live in experiments/bench_ivf_pq_*.py.
    """
    d, n_corpus, n_queries = 64, 2000, 50
    corpus = _clustered(n_corpus, d, n_clusters=50, seed=21)
    queries = _clustered(n_queries, d, n_clusters=50, seed=22)
    truth = _brute_topk(queries, corpus, 10)

    pq = PQSnapIndex(dim=d, M=16, K=64, normalized=True, seed=0)
    pq.fit(corpus[:1000])
    pq.add_batch(list(range(n_corpus)), corpus)

    ivf = IVFPQSnapIndex(
        dim=d, nlist=32, M=16, K=64, normalized=True, seed=0,
    )
    ivf.fit(corpus[:1000])
    ivf.add_batch(list(range(n_corpus)), corpus)

    r_pq = _recall(
        [[h[0] for h in pq.search(q, k=10)] for q in queries], truth, 10,
    )
    r_ivf_full = _recall(
        [[h[0] for h in ivf.search(q, k=10, nprobe=32)] for q in queries],
        truth, 10,
    )
    # Full probe should recall within noise of PQ full-scan; often
    # beats it because residual codebooks resolve smaller errors.
    assert r_ivf_full >= r_pq - 0.02, (
        f"IVF full-probe recall {r_ivf_full:.3f} lags PQ full-scan "
        f"{r_pq:.3f} by more than 2 pp"
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    corpus = _clustered(400, 64, seed=30)
    idx = IVFPQSnapIndex(
        dim=64, nlist=8, M=16, K=16, normalized=True, seed=42,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(400)), corpus)
    path = tmp_path / "x.snpi"
    idx.save(path)
    reloaded = IVFPQSnapIndex.load(path)
    assert len(reloaded) == 400
    assert reloaded.nlist == 8 and reloaded.M == 16 and reloaded.K == 16
    # Numeric ids survive — every id round-trips back to int.
    assert all(isinstance(i, int) for i in reloaded._ids_by_row)
    assert set(reloaded._ids_by_row) == set(range(400))
    q = corpus[5]
    a = idx.search(q, k=5, nprobe=8)
    b = reloaded.search(q, k=5, nprobe=8)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-5)


def test_unnormalized_scoring_scales_with_norm() -> None:
    base = _clustered(120, 32, n_clusters=10, seed=40)
    scales = np.linspace(0.5, 5.0, 120).astype(np.float32)
    corpus = base * scales[:, None]
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=False, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(120)), corpus)
    q = corpus[-1]
    hits = idx.search(q, k=3, nprobe=4)
    # Largest-scale entry must appear in top-3; not guaranteed #1 at K=16.
    assert 119 in [h[0] for h in hits]


def test_delete() -> None:
    corpus = _clustered(300, 32, seed=50)
    idx = IVFPQSnapIndex(
        dim=32, nlist=8, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(300)), corpus)
    assert idx.delete(100) is True
    assert idx.delete(100) is False
    assert len(idx) == 299
    # Deleted id must not reappear.
    for q in corpus[:5]:
        hits = idx.search(q, k=10, nprobe=8)
        assert 100 not in [h[0] for h in hits]


@pytest.mark.parametrize("use_rht", [False, True])
def test_use_rht_toggle(use_rht: bool) -> None:
    corpus = _clustered(400, 64, n_clusters=20, seed=60)
    idx = IVFPQSnapIndex(
        dim=64, nlist=8, M=16, K=16, normalized=True, use_rht=use_rht,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(400)), corpus)
    hits = idx.search(corpus[3], k=3, nprobe=8)
    top_id = hits[0][0]
    assert float(corpus[3] @ corpus[top_id]) > 0.9


def test_duplicate_ids_rejected() -> None:
    """Silently collapsing duplicate ids would make search return the
    wrong decoded vector for the earlier row.  Reject up front."""
    corpus = _unit_gaussian(100, 32, seed=80)
    idx = IVFPQSnapIndex(dim=32, nlist=4, M=8, K=16, normalized=True)
    idx.fit(corpus)
    with pytest.raises(ValueError, match="duplicate id in batch"):
        idx.add_batch([0, 1, 0], corpus[:3])
    idx.add_batch([0, 1, 2], corpus[:3])
    with pytest.raises(ValueError, match="already indexed"):
        idx.add_batch([1], corpus[:1])


def test_load_validates_offsets(tmp_path: Path) -> None:
    """Corrupted ``_offsets`` on disk must be caught at load time,
    not silently used to mis-index the codes buffer."""
    corpus = _unit_gaussian(100, 32, seed=81)
    idx = IVFPQSnapIndex(dim=32, nlist=4, M=8, K=16, normalized=True, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(100)), corpus)
    path = tmp_path / "x.snpi"
    idx.save(path)

    # Corrupt: scramble offsets to be non-monotone (swap first two).
    buf = bytearray(path.read_bytes())
    # Offsets are stored right after coarse + codebooks.  Easier to
    # mutate via a fresh reload, edit, re-save.
    reloaded = IVFPQSnapIndex.load(path)
    reloaded._offsets = reloaded._offsets.copy()
    reloaded._offsets[0] = 99  # break the offsets[0]==0 invariant
    reloaded.save(path)
    with pytest.raises(ValueError, match="offsets"):
        IVFPQSnapIndex.load(path)


def test_stats_shape() -> None:
    corpus = _unit_gaussian(300, 32, seed=70)
    idx = IVFPQSnapIndex(
        dim=32, nlist=8, M=8, K=16, normalized=True,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(300)), corpus)
    s = idx.stats()
    assert s["n"] == 300 and s["nlist"] == 8 and s["M"] == 8
    assert s["bytes_per_vec"] == 8  # M uint8 codes, no norm in normalized mode
    assert s["cluster_size_min"] >= 0
    assert s["cluster_size_max"] > 0
    assert s["default_nprobe"] == 1
