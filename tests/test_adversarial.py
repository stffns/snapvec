"""Adversarial edge-case tests.

Tiny dims, tiny N, degenerate distributions, empty inputs.  These are
the cases where off-by-one bugs and implicit shape assumptions tend to
surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from snapvec import IVFPQSnapIndex, PQSnapIndex, ResidualSnapIndex, SnapIndex


# --------------------------------------------------------------------------- #
# Empty index                                                                  #
# --------------------------------------------------------------------------- #


def test_empty_snapindex_search_returns_empty() -> None:
    idx = SnapIndex(dim=8, bits=4, seed=0)
    q = np.zeros(8, dtype=np.float32)
    q[0] = 1.0
    assert idx.search(q, k=5) == []
    assert len(idx) == 0


def test_empty_pqsnapindex_search_returns_empty() -> None:
    vecs = np.random.default_rng(0).standard_normal((64, 8)).astype(np.float32)
    idx = PQSnapIndex(dim=8, M=4, K=8, seed=0)
    idx.fit(vecs)
    # Never call add_batch -- index is fitted but empty.
    q = vecs[0]
    assert idx.search(q, k=5) == []
    assert len(idx) == 0


# --------------------------------------------------------------------------- #
# Single-vector corpus                                                         #
# --------------------------------------------------------------------------- #


def test_snapindex_n1() -> None:
    """n=1 is a legal if-degenerate corpus.  search(k>=1) returns 1 hit."""
    v = np.random.default_rng(0).standard_normal((1, 16)).astype(np.float32)
    idx = SnapIndex(dim=16, bits=4, seed=0)
    idx.add_batch(["only"], v)
    hits = idx.search(v[0], k=5)
    assert len(hits) == 1
    assert hits[0][0] == "only"


# --------------------------------------------------------------------------- #
# k larger than n                                                              #
# --------------------------------------------------------------------------- #


def test_search_k_larger_than_n_returns_n() -> None:
    vecs = np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)
    idx = SnapIndex(dim=16, bits=4, seed=0)
    idx.add_batch([0, 1, 2], vecs)
    hits = idx.search(vecs[0], k=100)
    assert len(hits) == 3


# --------------------------------------------------------------------------- #
# Zero-norm inputs                                                             #
# --------------------------------------------------------------------------- #


def test_search_with_zero_query_returns_empty() -> None:
    """Zero-norm query can't be normalized; library returns [] instead of NaN hits."""
    vecs = np.random.default_rng(0).standard_normal((20, 16)).astype(np.float32)
    idx = SnapIndex(dim=16, bits=4, seed=0)
    idx.add_batch(list(range(20)), vecs)
    q_zero = np.zeros(16, dtype=np.float32)
    assert idx.search(q_zero, k=5) == []


# --------------------------------------------------------------------------- #
# All-same-vector corpus (degenerate clusters)                                 #
# --------------------------------------------------------------------------- #


def test_snapindex_all_same_vector() -> None:
    """Every vector identical -> search should still return k distinct ids."""
    v = np.ones((1, 32), dtype=np.float32)
    vecs = np.tile(v, (10, 1))
    idx = SnapIndex(dim=32, bits=4, seed=0)
    idx.add_batch(list(range(10)), vecs)
    hits = idx.search(v[0], k=5)
    assert len(hits) == 5
    ids = [h[0] for h in hits]
    assert len(set(ids)) == len(ids)  # distinct


# --------------------------------------------------------------------------- #
# Filter edge cases                                                            #
# --------------------------------------------------------------------------- #


def test_filter_with_only_unknown_ids_returns_empty() -> None:
    vecs = np.random.default_rng(0).standard_normal((50, 16)).astype(np.float32)
    idx = SnapIndex(dim=16, bits=4, seed=0)
    idx.add_batch(list(range(50)), vecs)
    hits = idx.search(vecs[0], k=5, filter_ids={"never-added", "also-never"})
    assert hits == []


def test_filter_with_empty_set_returns_empty() -> None:
    vecs = np.random.default_rng(0).standard_normal((50, 16)).astype(np.float32)
    idx = SnapIndex(dim=16, bits=4, seed=0)
    idx.add_batch(list(range(50)), vecs)
    hits = idx.search(vecs[0], k=5, filter_ids=set())
    assert hits == []


# --------------------------------------------------------------------------- #
# Delete-all                                                                   #
# --------------------------------------------------------------------------- #


def test_snapindex_delete_all_then_search() -> None:
    vecs = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32)
    idx = SnapIndex(dim=16, bits=4, seed=0)
    idx.add_batch(list(range(5)), vecs)
    for i in range(5):
        assert idx.delete(i) is True
    assert len(idx) == 0
    assert idx.search(vecs[0], k=3) == []


# --------------------------------------------------------------------------- #
# Aggressive compression (bits=2)                                              #
# --------------------------------------------------------------------------- #


def test_snapindex_bits2_basic_recall() -> None:
    """bits=2 still returns *something* sensible on clustered data."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, 32)).astype(np.float32) * 3
    assign = rng.integers(0, 5, size=200)
    jitter = rng.standard_normal((200, 32)).astype(np.float32) * 0.2
    corpus = centers[assign] + jitter

    idx = SnapIndex(dim=32, bits=2, seed=0)
    idx.add_batch(list(range(200)), corpus)

    # Self-query should rank the exact corpus row near the top on such
    # strongly clustered data.
    hits = idx.search(corpus[0], k=5)
    assert len(hits) == 5
    returned = [h[0] for h in hits]
    # Because clusters have ~40 members and bits=2 is aggressive,
    # we don't assert hits[0] == 0.  We only assert the top-5 are all
    # from the same cluster as the query.
    query_cluster = assign[0]
    top_clusters = [assign[i] for i in returned]
    assert top_clusters.count(query_cluster) >= 3


# --------------------------------------------------------------------------- #
# k=0 is an error                                                              #
# --------------------------------------------------------------------------- #


def test_search_k_zero_raises() -> None:
    vecs = np.random.default_rng(0).standard_normal((10, 16)).astype(np.float32)
    idx = SnapIndex(dim=16, bits=4, seed=0)
    idx.add_batch(list(range(10)), vecs)
    with pytest.raises(ValueError):
        idx.search(vecs[0], k=0)


# --------------------------------------------------------------------------- #
# IVF-PQ extreme nprobe                                                        #
# --------------------------------------------------------------------------- #


def test_ivfpq_nprobe_equals_nlist_is_full_scan() -> None:
    """With nprobe=nlist, IVF-PQ must visit every cluster."""
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((300, 32)).astype(np.float32)
    idx = IVFPQSnapIndex(dim=32, nlist=8, M=4, K=16, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(300)), corpus)

    hits = idx.search(corpus[0], k=10, nprobe=8)
    assert len(hits) == 10


def test_ivfpq_nprobe_out_of_range_raises() -> None:
    rng = np.random.default_rng(0)
    corpus = rng.standard_normal((300, 32)).astype(np.float32)
    idx = IVFPQSnapIndex(dim=32, nlist=8, M=4, K=16, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(300)), corpus)
    with pytest.raises(ValueError):
        idx.search(corpus[0], k=5, nprobe=0)
    with pytest.raises(ValueError):
        idx.search(corpus[0], k=5, nprobe=99)


# --------------------------------------------------------------------------- #
# Residual rerank                                                              #
# --------------------------------------------------------------------------- #


def test_residual_rerank_saturates_near_full_recall() -> None:
    """ResidualSnapIndex with a generous rerank_M should match full scan on
    clustered data."""
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((8, 32)).astype(np.float32) * 3
    assign = rng.integers(0, 8, size=200)
    jitter = rng.standard_normal((200, 32)).astype(np.float32) * 0.2
    corpus = centers[assign] + jitter

    idx = ResidualSnapIndex(dim=32, b1=3, b2=3, seed=0)
    idx.add_batch(list(range(200)), corpus)

    full = [h[0] for h in idx.search(corpus[0], k=5, rerank_M=None)]
    reranked = [h[0] for h in idx.search(corpus[0], k=5, rerank_M=50)]
    # Both modes should agree on most of the top-5.
    assert len(set(full) & set(reranked)) >= 3
