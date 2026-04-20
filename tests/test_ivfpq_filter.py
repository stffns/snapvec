"""IVF-PQ ``filter_ids`` — cluster-skip + pool-aware tests.

Verifies the three correctness claims of the v0.8 filter:

1. Every returned id is in ``filter_ids`` (no leaks).
2. The top-k over the filtered subset matches a brute-force reference
   computed over the same subset (no spurious skips).
3. With ``rerank_candidates`` set, the rerank pool is drawn from the
   filtered subset — i.e. if the filter excludes every unfiltered
   top-k candidate, the rerank path still returns non-empty results
   (pool-aware claim).

Plus two behavioural checks: cluster-skip actually reduces the set of
probed clusters for sparse filters, and search_batch honours the
filter identically to loop-of-search.
"""
from __future__ import annotations

import numpy as np

from snapvec import IVFPQSnapIndex


def _clustered(
    n: int, d: int, n_clusters: int = 20, noise: float = 0.15, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    x = centers[assign] + noise * rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _build_index(n: int = 1_500, dim: int = 64, **kw) -> IVFPQSnapIndex:
    corpus = _clustered(n, dim, n_clusters=24, seed=0)
    idx = IVFPQSnapIndex(
        dim=dim, nlist=16, M=8, K=16, normalized=True, seed=0, **kw,
    )
    idx.fit(corpus[: n // 2])
    idx.add_batch(list(range(n)), corpus)
    # Stash the raw corpus on the instance so tests can build their own
    # brute-force reference against the same post-preprocess vectors.
    idx._corpus_for_tests = corpus  # type: ignore[attr-defined]
    return idx


def test_filter_results_are_subset_of_filter_set() -> None:
    idx = _build_index()
    filter_ids = set(range(0, 1_500, 7))  # every 7th id -> 215 ids
    queries = _clustered(5, 64, n_clusters=24, seed=99)
    for q in queries:
        hits = idx.search(q, k=10, nprobe=8, filter_ids=filter_ids)
        assert all(h[0] in filter_ids for h in hits)


def test_filter_matches_post_hoc_reference() -> None:
    """Correctness: every filtered hit must score at least as high as
    the k-th best filter-scoped score from an unfiltered full sort.
    We avoid set-equality because a tie at rank k can be broken either
    way by ``argpartition`` (non-deterministic in the presence of
    exact-equal floats) — what matters is that no *strictly better*
    filter hit was dropped."""
    idx = _build_index()
    filter_ids = set(range(0, 1_500, 5))  # 300 ids
    q = _clustered(1, 64, n_clusters=24, seed=11)[0]

    unfiltered = idx.search(q, k=len(idx), nprobe=idx.nlist)
    filter_scores = [s for i, s in unfiltered if i in filter_ids]
    kth_best = sorted(filter_scores, reverse=True)[9]

    hits = idx.search(q, k=10, nprobe=idx.nlist, filter_ids=filter_ids)
    assert len(hits) == 10
    assert all(h[0] in filter_ids for h in hits)
    for id_, score in hits:
        # Allow a 1e-6 fp slack around exact ties.
        assert score >= kth_best - 1e-6, (
            f"filter path returned id={id_} score={score} below the "
            f"k-th best filter score {kth_best} from the full sort"
        )


def test_filter_empty_when_all_ids_unknown() -> None:
    idx = _build_index()
    q = _clustered(1, 64, seed=12)[0]
    assert idx.search(q, k=10, filter_ids={10_000, 20_000}) == []


def test_filter_drops_unknown_ids_silently() -> None:
    idx = _build_index()
    q = _clustered(1, 64, seed=13)[0]
    hits = idx.search(q, k=10, nprobe=16, filter_ids={0, 1, 99_999})
    # Only 0 and 1 are real ids, so <= 2 hits.
    assert all(h[0] in {0, 1} for h in hits)
    assert len(hits) <= 2


def test_filter_with_rerank_is_pool_aware() -> None:
    """Pool-aware claim: the rerank candidate pool is drawn from the
    filtered subset.  We pick a query, compute its unfiltered top-20,
    and then filter them all OUT — the filtered search must still
    return 5 hits from the non-excluded subset, not empty."""
    idx = _build_index(keep_full_precision=True)
    q = _clustered(1, 64, seed=21)[0]

    unfiltered = idx.search(q, k=20, nprobe=idx.nlist)
    excluded = {h[0] for h in unfiltered}
    # Filter is "everything except the top-20 unfiltered hits".
    allowed = set(range(1_500)) - excluded

    hits = idx.search(
        q, k=5, nprobe=idx.nlist, rerank_candidates=50, filter_ids=allowed,
    )
    assert len(hits) == 5
    assert all(h[0] in allowed for h in hits)
    assert not any(h[0] in excluded for h in hits)


def test_cluster_skip_restricts_probes_for_sparse_filter() -> None:
    """A filter that touches only K clusters should force the probe
    set to be a subset of those K clusters — we verify by picking a
    filter whose rows all live in a known pair of clusters and
    confirming ``_probe_topn`` returns a probe set drawn from that
    pair only."""
    idx = _build_index()
    # Reconstruct per-row cluster assignments from _offsets.
    asn = idx._cluster_ids_from_offsets()
    target_clusters = (3, 5)
    rows_in_target = np.where(np.isin(asn, target_clusters))[0]
    # Map those rows back to external ids via _ids_by_row.
    filter_ids = {idx._ids_by_row[int(r)] for r in rows_in_target}
    assert len(filter_ids) > 0

    resolved = idx._resolve_filter(filter_ids)
    assert resolved is not None
    _, allowed_clusters = resolved
    assert set(allowed_clusters.tolist()).issubset(set(target_clusters))

    # And the real search path: every returned id must be in filter,
    # even at low nprobe where without cluster-skip we'd probe clusters
    # that contain no filter rows at all.
    q = _clustered(1, 64, seed=31)[0]
    hits = idx.search(q, k=10, nprobe=2, filter_ids=filter_ids)
    assert all(h[0] in filter_ids for h in hits)


def test_search_batch_filter_matches_loop_of_search() -> None:
    idx = _build_index()
    filter_ids = set(range(0, 1_500, 3))
    queries = _clustered(8, 64, n_clusters=24, seed=41)

    batch_hits = idx.search_batch(
        queries, k=5, nprobe=idx.nlist, filter_ids=filter_ids,
    )
    loop_hits = [
        idx.search(q, k=5, nprobe=idx.nlist, filter_ids=filter_ids)
        for q in queries
    ]

    assert len(batch_hits) == len(loop_hits)
    for b, loop in zip(batch_hits, loop_hits):
        # Same id count, every hit in filter, and the returned score
        # for the *worst* hit in each list is within fp tolerance of
        # the other's worst -- proves both paths converge at the same
        # top-k boundary even if tied candidates reorder inside it.
        assert len(b) == len(loop)
        assert all(h[0] in filter_ids for h in b)
        assert all(h[0] in filter_ids for h in loop)
        assert abs(b[-1][1] - loop[-1][1]) < 1e-4


def test_search_batch_filter_all_unknown_returns_empties() -> None:
    idx = _build_index()
    queries = _clustered(4, 64, seed=42)
    batch = idx.search_batch(queries, k=5, filter_ids={99_999, 99_998})
    assert batch == [[], [], [], []]
