"""Property-based tests.

These exercise invariants that should hold for *any* input within a
modest range.  They complement the hand-written unit tests by catching
regressions on inputs nobody thought to write down.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from snapvec import IVFPQSnapIndex, PQSnapIndex, SnapIndex


PROFILE = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _corpus(n: int, dim: int, seed: int) -> NDArray[np.float32]:
    """Clustered corpus so PQ / IVF-PQ tests exercise real structure."""
    rng = np.random.default_rng(seed)
    n_clusters = max(2, n // 10)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3
    assign = rng.integers(0, n_clusters, size=n)
    jitter = rng.standard_normal((n, dim)).astype(np.float32) * 0.3
    return centers[assign] + jitter


# --------------------------------------------------------------------------- #
# SnapIndex invariants                                                         #
# --------------------------------------------------------------------------- #


@PROFILE
@given(
    dim=st.sampled_from([8, 16, 32, 64]),
    n=st.integers(min_value=1, max_value=100),
    seed=st.integers(min_value=0, max_value=2**16),
    bits=st.sampled_from([2, 3, 4]),
)
def test_snap_add_then_len_matches(dim: int, n: int, seed: int, bits: int) -> None:
    """len(idx) equals the number of distinct ids added."""
    vecs = _corpus(n, dim, seed)
    idx = SnapIndex(dim=dim, bits=bits, seed=0)
    idx.add_batch(list(range(n)), vecs)
    assert len(idx) == n


@PROFILE
@given(
    dim=st.sampled_from([8, 16, 32]),
    n=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_snap_search_returns_at_most_k(dim: int, n: int, seed: int) -> None:
    """search(q, k) returns at most k hits, sorted by descending score."""
    vecs = _corpus(n, dim, seed)
    idx = SnapIndex(dim=dim, bits=4, seed=0)
    idx.add_batch(list(range(n)), vecs)

    for k in (1, min(10, n), n + 5):
        hits = idx.search(vecs[0], k=k)
        assert len(hits) <= k
        assert len(hits) <= n
        scores = [h[1] for h in hits]
        assert scores == sorted(scores, reverse=True)


@PROFILE
@given(
    dim=st.sampled_from([8, 16, 32]),
    n=st.integers(min_value=3, max_value=50),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_snap_delete_reduces_len(dim: int, n: int, seed: int) -> None:
    """Deleting an existing id reduces len by exactly 1."""
    vecs = _corpus(n, dim, seed)
    idx = SnapIndex(dim=dim, bits=4, seed=0)
    idx.add_batch(list(range(n)), vecs)

    removed = idx.delete(0)
    assert removed is True
    assert len(idx) == n - 1

    # Deleting a non-existent id is a no-op.
    assert idx.delete(10**9) is False
    assert len(idx) == n - 1


@PROFILE
@given(
    dim=st.sampled_from([8, 16, 32]),
    n=st.integers(min_value=2, max_value=50),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_snap_save_load_preserves_search(
    dim: int, n: int, seed: int, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Round-trip through save/load returns bit-identical search results."""
    vecs = _corpus(n, dim, seed)
    idx = SnapIndex(dim=dim, bits=4, seed=0)
    idx.add_batch(list(range(n)), vecs)

    path: Path = tmp_path_factory.mktemp("prop") / "idx.snpv"
    idx.save(path)
    loaded = SnapIndex.load(path)

    before = idx.search(vecs[0], k=min(5, n))
    after = loaded.search(vecs[0], k=min(5, n))
    assert before == after


@PROFILE
@given(
    dim=st.sampled_from([8, 16]),
    n=st.integers(min_value=5, max_value=40),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_snap_filter_hits_are_in_filter_set(dim: int, n: int, seed: int) -> None:
    """With filter_ids=S, every returned hit is in S."""
    vecs = _corpus(n, dim, seed)
    idx = SnapIndex(dim=dim, bits=4, seed=0)
    idx.add_batch(list(range(n)), vecs)

    # Pick a sparse filter: first third of the ids.
    filter_set = set(range(max(1, n // 3)))
    hits = idx.search(vecs[0], k=min(5, n), filter_ids=filter_set)
    for doc_id, _score in hits:
        assert doc_id in filter_set


# --------------------------------------------------------------------------- #
# PQSnapIndex invariants                                                       #
# --------------------------------------------------------------------------- #


@PROFILE
@given(
    dim_pair=st.sampled_from([(16, 4), (32, 8), (64, 8)]),
    n=st.integers(min_value=20, max_value=100),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_pq_fit_then_add_then_len(
    dim_pair: tuple[int, int], n: int, seed: int
) -> None:
    """PQSnapIndex after fit() + add_batch reports the right len()."""
    dim, M = dim_pair
    vecs = _corpus(n, dim, seed)
    idx = PQSnapIndex(dim=dim, M=M, K=16, seed=0)
    idx.fit(vecs)
    idx.add_batch(list(range(n)), vecs)
    assert len(idx) == n


# --------------------------------------------------------------------------- #
# IVFPQSnapIndex invariants                                                    #
# --------------------------------------------------------------------------- #


@PROFILE
@given(
    n=st.integers(min_value=100, max_value=300),
    nprobe=st.sampled_from([1, 2, 4, 8]),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_ivfpq_search_respects_k(n: int, nprobe: int, seed: int) -> None:
    """IVFPQSnapIndex returns <= k results, sorted by score descending."""
    dim, M, K, nlist = 16, 4, 16, 8
    vecs = _corpus(n, dim, seed)
    idx = IVFPQSnapIndex(dim=dim, nlist=nlist, M=M, K=K, seed=0)
    idx.fit(vecs)
    idx.add_batch(list(range(n)), vecs)

    for k in (1, 5, 20):
        hits = idx.search(vecs[0], k=k, nprobe=nprobe)
        assert len(hits) <= k
        scores = [h[1] for h in hits]
        assert scores == sorted(scores, reverse=True)


@PROFILE
@given(
    n=st.integers(min_value=100, max_value=200),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_ivfpq_unknown_filter_returns_empty(n: int, seed: int) -> None:
    """filter_ids with no matching id returns []."""
    dim, M, K, nlist = 16, 4, 16, 8
    vecs = _corpus(n, dim, seed)
    idx = IVFPQSnapIndex(dim=dim, nlist=nlist, M=M, K=K, seed=0)
    idx.fit(vecs)
    idx.add_batch(list(range(n)), vecs)

    hits = idx.search(vecs[0], k=5, filter_ids={"nope-nope-nope"})
    assert hits == []
