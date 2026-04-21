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


def test_search_batch_matches_per_query_loop() -> None:
    """search_batch must produce exactly the same top-k as a per-query
    loop over search() — in id, ordering, and score (within float
    noise from the einsum-built batched LUT)."""
    d, n_corpus, n_queries = 128, 1500, 25
    corpus = _clustered(n_corpus, d, n_clusters=30, seed=81)
    queries = _clustered(n_queries, d, n_clusters=30, seed=82)
    ivf = IVFPQSnapIndex(
        dim=d, nlist=32, M=16, K=64, normalized=True, seed=0,
    )
    ivf.fit(corpus[:1000])
    ivf.add_batch(list(range(n_corpus)), corpus)

    serial = [ivf.search(q, k=10, nprobe=16) for q in queries]
    batched = ivf.search_batch(queries, k=10, nprobe=16, num_threads=1)
    assert len(serial) == len(batched)
    for i, (s, b) in enumerate(zip(serial, batched)):
        # ordering, ids, and scores must all match
        assert [h[0] for h in s] == [h[0] for h in b], (
            f"batched ordering diverged from serial at query {i}"
        )
        assert np.allclose(
            [h[1] for h in s], [h[1] for h in b], atol=1e-5,
        ), f"batched scores diverged from serial at query {i}"


def test_search_batch_threaded_matches_serial() -> None:
    """Threaded batch must match single-thread batch results — same
    ordering and scores, not just the same set of ids."""
    d, n_corpus, n_queries = 128, 1500, 32
    corpus = _clustered(n_corpus, d, n_clusters=30, seed=83)
    queries = _clustered(n_queries, d, n_clusters=30, seed=84)
    ivf = IVFPQSnapIndex(
        dim=d, nlist=32, M=16, K=64, normalized=True, seed=0,
    )
    ivf.fit(corpus[:1000])
    ivf.add_batch(list(range(n_corpus)), corpus)
    serial = ivf.search_batch(queries, k=10, nprobe=16, num_threads=1)
    threaded = ivf.search_batch(queries, k=10, nprobe=16, num_threads=4)
    for i, (s, t) in enumerate(zip(serial, threaded)):
        assert [h[0] for h in s] == [h[0] for h in t], (
            f"threaded batch ordering diverged at query {i}"
        )
        assert np.allclose(
            [h[1] for h in s], [h[1] for h in t], atol=1e-6,
        ), f"threaded batch scores diverged at query {i}"


def test_search_batch_unnormalized_matches_loop() -> None:
    """search_batch with normalized=False (per-vector norms scaled
    in) must still match the per-query loop, since the multi-step
    pipeline (norm, RHT-or-not, gather, norm-multiply) is more
    sensitive than the normalized=True path."""
    d, n_corpus, n_queries = 64, 500, 20
    base = _clustered(n_corpus, d, n_clusters=20, seed=95)
    scales = np.linspace(0.5, 5.0, n_corpus).astype(np.float32)
    corpus = base * scales[:, None]
    queries = _clustered(n_queries, d, n_clusters=20, seed=96)
    ivf = IVFPQSnapIndex(
        dim=d, nlist=16, M=8, K=64, normalized=False, seed=0,
    )
    ivf.fit(corpus[:300])
    ivf.add_batch(list(range(n_corpus)), corpus)

    serial = [ivf.search(q, k=5, nprobe=8) for q in queries]
    batched = ivf.search_batch(queries, k=5, nprobe=8)
    for i, (s, b) in enumerate(zip(serial, batched)):
        assert [h[0] for h in s] == [h[0] for h in b], (
            f"unnormalized batch ordering diverged at query {i}"
        )


def test_search_batch_handles_zero_norm_queries() -> None:
    """Zero-norm queries in a batch must return [] for those slots,
    not crash and not poison the rest of the batch."""
    d, n_corpus = 64, 600
    corpus = _clustered(n_corpus, d, seed=85)
    queries = _clustered(5, d, seed=86)
    queries[2] = 0  # poison one
    ivf = IVFPQSnapIndex(
        dim=d, nlist=16, M=16, K=16, normalized=True, seed=0,
    )
    ivf.fit(corpus[:500])
    ivf.add_batch(list(range(n_corpus)), corpus)
    out = ivf.search_batch(queries, k=5, nprobe=8)
    assert out[2] == []
    for i in (0, 1, 3, 4):
        assert len(out[i]) == 5


def test_search_batch_validates_shape() -> None:
    corpus = _unit_gaussian(200, 32, seed=87)
    idx = IVFPQSnapIndex(dim=32, nlist=4, M=8, K=16, normalized=True)
    idx.fit(corpus)
    idx.add_batch(list(range(200)), corpus)
    with pytest.raises(ValueError, match="queries"):
        idx.search_batch(np.zeros((3, 16), dtype=np.float32), k=1)
    # Empty batch returns empty list.
    assert idx.search_batch(np.zeros((0, 32), dtype=np.float32), k=1) == []


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


def test_rerank_beats_pq_only_recall() -> None:
    """rerank_candidates must lift recall above the PQ-only ceiling.
    On clustered synthetic data with M=8 K=16 the PQ approximation is
    aggressive enough to drop a few neighbours; the float32 rerank
    pass should recover them."""
    d, n_corpus, n_queries = 64, 1500, 50
    corpus = _clustered(n_corpus, d, n_clusters=30, seed=100)
    queries = _clustered(n_queries, d, n_clusters=30, seed=101)
    truth = _brute_topk(queries, corpus, 10)

    idx = IVFPQSnapIndex(
        dim=d, nlist=16, M=8, K=16, normalized=True, seed=0,
        keep_full_precision=True,
    )
    idx.fit(corpus[:1000])
    idx.add_batch(list(range(n_corpus)), corpus)

    r_pq = _recall(
        [[h[0] for h in idx.search(q, k=10, nprobe=8)] for q in queries],
        truth, 10,
    )
    r_rerank = _recall(
        [[h[0] for h in idx.search(q, k=10, nprobe=8, rerank_candidates=50)]
         for q in queries], truth, 10,
    )
    assert r_rerank > r_pq, (
        f"rerank ({r_rerank:.3f}) did not improve over PQ-only ({r_pq:.3f})"
    )
    # And rerank with a wide candidate pool should approach the
    # float32 ceiling.
    r_full_rerank = _recall(
        [[h[0] for h in idx.search(q, k=10, nprobe=16, rerank_candidates=200)]
         for q in queries], truth, 10,
    )
    assert r_full_rerank >= 0.95, (
        f"full-rerank recall {r_full_rerank:.3f} should be ≥ 0.95"
    )


def test_rerank_requires_keep_full_precision() -> None:
    corpus = _unit_gaussian(200, 32, seed=110)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True,
        keep_full_precision=False,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(200)), corpus)
    with pytest.raises(ValueError, match="keep_full_precision"):
        idx.search(corpus[0], k=5, rerank_candidates=20)


def test_rerank_candidates_must_be_at_least_k() -> None:
    corpus = _unit_gaussian(200, 32, seed=111)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True,
        keep_full_precision=True,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(200)), corpus)
    with pytest.raises(ValueError, match=">= k"):
        idx.search(corpus[0], k=10, rerank_candidates=5)


def test_rerank_candidate_pool_respects_norms_when_unnormalized() -> None:
    """When ``normalized=False``, the PQ-based candidate selection
    must scale scores by per-vector norms — otherwise a
    high-norm vector that genuinely wins on ⟨q, v⟩ can miss the
    candidate pool because its unit-sphere PQ score is modest.
    This was the high-priority bug caught on the PR #27 review."""
    # Build corpus where recall truth depends heavily on per-vector
    # norm magnitude: same directions scaled across a wide range.
    d, n_corpus = 64, 400
    base = _clustered(n_corpus, d, n_clusters=20, seed=140)
    scales = np.linspace(0.2, 10.0, n_corpus).astype(np.float32)
    corpus = base * scales[:, None]
    queries = _clustered(20, d, n_clusters=20, seed=141)
    truth = _brute_topk(queries, corpus, 10)

    idx = IVFPQSnapIndex(
        dim=d, nlist=8, M=8, K=32, normalized=False, seed=0,
        keep_full_precision=True,
    )
    idx.fit(corpus[:250])
    idx.add_batch(list(range(n_corpus)), corpus)
    r = _recall(
        [[h[0] for h in idx.search(q, k=10, nprobe=8, rerank_candidates=40)]
         for q in queries], truth, 10,
    )
    # Without the norm scaling in the candidate-selection step, the
    # pool would miss many large-norm winners and recall would tank.
    # With the fix, rerank reaches > 0.8 on this stress test.
    assert r > 0.8, (
        f"rerank recall {r:.3f} below 0.8 — candidate selection may "
        f"be ignoring per-vector norms in the normalized=False path."
    )


def test_save_load_with_full_precision_roundtrips(tmp_path: Path) -> None:
    """File-format v3 must round-trip the float32 cache so that a
    reloaded index reproduces rerank scores exactly."""
    corpus = _clustered(200, 32, seed=120)
    queries = _clustered(5, 32, seed=121)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=42,
        keep_full_precision=True,
    )
    idx.fit(corpus[:120])
    idx.add_batch(list(range(200)), corpus)
    path = tmp_path / "x.snpi"
    idx.save(path)
    reloaded = IVFPQSnapIndex.load(path)
    assert reloaded.keep_full_precision
    assert reloaded._full_precision.shape == idx._full_precision.shape
    np.testing.assert_array_equal(
        reloaded._full_precision, idx._full_precision,
    )
    for q in queries:
        a = idx.search(q, k=3, nprobe=4, rerank_candidates=20)
        b = reloaded.search(q, k=3, nprobe=4, rerank_candidates=20)
        assert [x[0] for x in a] == [x[0] for x in b]
        assert np.allclose([x[1] for x in a], [x[1] for x in b], atol=1e-6)


def test_stats_reports_full_precision_overhead() -> None:
    corpus = _unit_gaussian(100, 64, seed=130)
    idx = IVFPQSnapIndex(
        dim=64, nlist=4, M=8, K=16, normalized=True,
        keep_full_precision=True,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(100)), corpus)
    s = idx.stats()
    assert s["keep_full_precision"] is True
    # M codes (8 bytes) + 64 dim × 2 bytes float16 cache = 8 + 128 = 136
    assert s["bytes_per_vec_codes_only"] == 8
    assert s["bytes_per_vec"] == 8 + 64 * 2


def test_full_precision_cache_is_fp16_in_memory() -> None:
    """Contract: _full_precision must be fp16 from v0.7 onward.
    Guards against accidental dtype regressions that would double
    the RAM footprint of keep_full_precision=True indices."""
    corpus = _unit_gaussian(50, 32, seed=145)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True,
        keep_full_precision=True,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(50)), corpus)
    assert idx._full_precision.dtype == np.float16


def test_fp16_cache_recall_matches_fp32_within_noise() -> None:
    """fp16 is the v0.7 default for the rerank cache.  The precision
    loss from the float32 → float16 cast (at ``add_batch`` time) must
    leave rerank recall within a small delta of what the un-truncated
    float32 cache would have produced.

    Correctness hazard this test guards against (caught in PR #31
    review): an earlier version of the test upcast the *stored* fp16
    back to fp32 and ran both paths against that.  Since upcasting
    preserves every representable fp16 value exactly, the two runs
    produced bit-identical scores — the assertion was vacuously
    satisfied and a future regression that made the cast lossier
    would have slipped through.

    Fix: reconstruct the *pre-truncation* fp32 cache by re-running
    ``_preprocess`` on the original float32 corpus.  The fp32 cache
    holds the true values before quantisation; the fp16 cache holds
    the quantised ones.  The A/B now actually exercises the cast's
    rounding error.
    """
    d, n_corpus, n_queries = 64, 1200, 40
    corpus = _clustered(n_corpus, d, n_clusters=25, seed=150)
    queries = _clustered(n_queries, d, n_clusters=25, seed=151)
    truth = _brute_topk(queries, corpus, 10)

    idx = IVFPQSnapIndex(
        dim=d, nlist=16, M=8, K=16, normalized=True, seed=0,
        keep_full_precision=True,
    )
    idx.fit(corpus[:800])
    idx.add_batch(list(range(n_corpus)), corpus)

    # fp16 path (the shipped default).
    r_fp16 = _recall(
        [[h[0] for h in idx.search(q, k=10, nprobe=8, rerank_candidates=40)]
         for q in queries], truth, 10,
    )

    # True fp32 reference: re-run the same preprocessing on the
    # original float32 corpus, *without* the fp16 cast the index
    # applies.  Then reorder to match the cluster-sorted row layout
    # the index holds internally — row i in our fp32 reference must
    # correspond to the same vector as row i in the fp16 cache.  The
    # ids in ``_ids_by_row`` are integers 0..n, so they double as
    # indices back into the original corpus.
    pre_fp32, _ = idx._preprocess(corpus.astype(np.float32))
    fp32_cache = pre_fp32[np.array(idx._ids_by_row, dtype=np.int64)]
    assert fp32_cache.dtype == np.float32
    assert fp32_cache.shape == idx._full_precision.shape

    original_fp16 = idx._full_precision
    idx._full_precision = fp32_cache
    try:
        r_fp32 = _recall(
            [[h[0] for h in idx.search(q, k=10, nprobe=8, rerank_candidates=40)]
             for q in queries], truth, 10,
        )
    finally:
        idx._full_precision = original_fp16

    assert abs(r_fp16 - r_fp32) < 0.02, (
        f"fp16 recall {r_fp16:.3f} diverged from the un-truncated fp32 "
        f"reference {r_fp32:.3f} by {abs(r_fp16 - r_fp32):.3f}; the "
        f"fp16 cache cast is now materially lossy"
    )
    # And sanity: the two runs MUST differ at least a little in
    # score space (otherwise we regressed to the vacuous version).
    # Pick one query and confirm at least one score changed.
    hits_fp16 = idx.search(queries[0], k=10, nprobe=8, rerank_candidates=40)
    idx._full_precision = fp32_cache
    try:
        hits_fp32 = idx.search(queries[0], k=10, nprobe=8, rerank_candidates=40)
    finally:
        idx._full_precision = original_fp16
    # Compare both id ordering and scores: if precision loss reshuffles
    # the ranking, the per-rank ids will diverge even when the raw
    # scores at those ranks happen to be near-identical — so we treat
    # either a score delta or an id delta as evidence that the test is
    # exercising fp16 vs fp32 rather than fp16 vs fp16.
    results_differ = any(
        a[0] != b[0] or abs(a[1] - b[1]) > 1e-8
        for a, b in zip(hits_fp16, hits_fp32)
    )
    assert results_differ, (
        "fp16 and fp32 produced byte-identical results — test regressed "
        "back to the vacuous version it was meant to replace"
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

    # Corrupt: break the offsets[0] == 0 invariant so the loader rejects
    # the file.  Easier to reload, mutate in memory, and re-save than to
    # patch raw bytes at the right offset inside the binary format.
    reloaded = IVFPQSnapIndex.load(path)
    reloaded._offsets = reloaded._offsets.copy()
    reloaded._offsets[0] = 99
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


# ──────────────────────────────────────────────────────────────────── #
# OPQ rotation                                                           #
# ──────────────────────────────────────────────────────────────────── #


def test_ivfpq_opq_fit_stores_rotation() -> None:
    corpus = _clustered(500, 32, n_clusters=8, seed=50)
    idx = IVFPQSnapIndex(
        dim=32, nlist=8, M=8, K=16, normalized=True, use_opq=True, seed=0,
    )
    assert idx._opq_rotation is None
    idx.fit(corpus)
    R = idx._opq_rotation
    assert R is not None
    assert R.shape == (32, 32)
    # R is orthogonal.
    np.testing.assert_allclose(R.T @ R, np.eye(32), atol=1e-4)


def test_ivfpq_opq_and_rht_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        IVFPQSnapIndex(dim=32, nlist=4, M=8, K=16, use_opq=True, use_rht=True)


def test_ivfpq_opq_round_trip(tmp_path: Path) -> None:
    corpus = _clustered(500, 32, n_clusters=8, seed=51)
    idx = IVFPQSnapIndex(
        dim=32, nlist=8, M=8, K=16, normalized=True, use_opq=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(500)), corpus)
    path = tmp_path / "opq.snpi"
    idx.save(path)
    loaded = IVFPQSnapIndex.load(path)
    assert loaded.use_opq is True
    assert loaded._opq_rotation is not None
    np.testing.assert_array_equal(loaded._opq_rotation, idx._opq_rotation)
    q = corpus[0]
    before = idx.search(q, k=5, nprobe=4)
    after = loaded.search(q, k=5, nprobe=4)
    assert [h[0] for h in before] == [h[0] for h in after]
    np.testing.assert_allclose(
        [h[1] for h in before], [h[1] for h in after], atol=1e-5,
    )


def test_ivfpq_opq_search_batch_matches_search() -> None:
    """search_batch() must rotate queries the same way search() does
    when use_opq=True -- both paths go through the learned rotation.
    Regression guard for the bug Gemini caught in PR #64 review where
    search_batch was OPQ-blind and silently diverged from search."""
    corpus = _clustered(500, 32, n_clusters=8, seed=70)
    queries = _clustered(10, 32, n_clusters=8, seed=71)
    idx = IVFPQSnapIndex(
        dim=32, nlist=8, M=8, K=16, normalized=True, use_opq=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(500)), corpus)

    serial = [idx.search(q, k=5, nprobe=4) for q in queries]
    batched = idx.search_batch(queries, k=5, nprobe=4, num_threads=1)
    for i, (s, b) in enumerate(zip(serial, batched)):
        assert [h[0] for h in s] == [h[0] for h in b], (
            f"OPQ search_batch ids diverge from search at query {i}: "
            f"{[h[0] for h in s]} vs {[h[0] for h in b]}"
        )
        np.testing.assert_allclose(
            [h[1] for h in s], [h[1] for h in b], atol=1e-5,
            err_msg=f"OPQ search_batch scores diverge at query {i}",
        )


def test_ivfpq_opq_recall_not_worse_than_baseline() -> None:
    """On clustered synthetic data with a challenging PQ budget
    (M=4, K=16, d=32), OPQ should match or beat the unrotated
    baseline.  Not a strict recall-delta test because OPQ gains are
    data-dependent; only asserts no regression."""
    corpus = _clustered(1500, 64, n_clusters=20, seed=60)
    queries = _clustered(50, 64, n_clusters=20, seed=61)

    def build_search(use_opq: bool) -> list[list[int]]:
        idx = IVFPQSnapIndex(
            dim=64, nlist=16, M=4, K=16,
            normalized=True, use_opq=use_opq, seed=0,
        )
        idx.fit(corpus[:1000])
        idx.add_batch(list(range(1500)), corpus)
        return [
            [h[0] for h in idx.search(q, k=10, nprobe=8)]
            for q in queries
        ]

    truth = _brute_topk(queries, corpus, 10)
    base = build_search(use_opq=False)
    opq = build_search(use_opq=True)
    base_recall = _recall(base, truth, 10)
    opq_recall = _recall(opq, truth, 10)
    # Allow a small tolerance for data-specific variance.  OPQ should
    # not materially regress (within 3 percentage points).
    assert opq_recall >= base_recall - 0.03, (
        f"OPQ recall {opq_recall:.3f} regressed below baseline "
        f"{base_recall:.3f} on clustered synthetic data"
    )
