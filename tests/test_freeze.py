"""freeze() contract tests + targeted frozen-search concurrency.

Covers the four index classes uniformly for the freeze / unfreeze
mutation contract: after ``idx.freeze()`` the mutating methods raise
``RuntimeError`` naming the operation, and ``unfreeze()`` restores
the original mutation surface.

Frozen-search thread-safety is covered for the two classes that hold
shared state reachable from ``search`` / ``search_batch``:

* ``SnapIndex`` — ``freeze()`` must pre-warm the lazy ``_cache``, so
  concurrent first-queries on a frozen index don't race to
  materialise it.  The ``_cache``-hits-after-freeze assertion plus
  the 32-thread stress test below exercise that.
* ``IVFPQSnapIndex.search_batch(num_threads > 1)`` — lazy executor
  creation is serialised by a lock, and the init path is now
  idempotent on shutdown (we do not tear down and recreate the pool
  under concurrent callers).  Verified by monkeypatching the
  ``ThreadPoolExecutor`` constructor and counting instantiations.

``PQSnapIndex`` / ``ResidualSnapIndex`` have no lazy shared state
reachable from ``search`` today, so there is nothing extra to
assert beyond the mutation contract.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from snapvec import (
    IVFPQSnapIndex,
    PQSnapIndex,
    ResidualSnapIndex,
    SnapIndex,
)


def _unit_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _clustered(
    n: int, d: int, n_clusters: int = 20, noise: float = 0.15, seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32)
    assign = rng.integers(0, n_clusters, size=n)
    x = centers[assign] + noise * rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────────── #
# freeze / unfreeze contract — one parametrised test per mutator       #
# ──────────────────────────────────────────────────────────────────── #

def _build_snap() -> SnapIndex:
    idx = SnapIndex(dim=32, bits=4, normalized=True, seed=0)
    idx.add_batch(list(range(30)), _unit_gaussian(30, 32, seed=1))
    return idx


def _build_residual() -> ResidualSnapIndex:
    idx = ResidualSnapIndex(dim=32, b1=3, b2=3, normalized=True, seed=0)
    idx.add_batch(list(range(30)), _unit_gaussian(30, 32, seed=2))
    return idx


def _build_pq() -> PQSnapIndex:
    corpus = _unit_gaussian(60, 32, seed=3)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(60)), corpus)
    return idx


def _build_ivfpq() -> IVFPQSnapIndex:
    corpus = _unit_gaussian(200, 32, seed=4)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(200)), corpus)
    return idx


@pytest.mark.parametrize("builder, sample", [
    (_build_snap, _unit_gaussian(30, 32, seed=10)),
    (_build_residual, _unit_gaussian(30, 32, seed=11)),
    (_build_pq, _unit_gaussian(60, 32, seed=12)),
    (_build_ivfpq, _unit_gaussian(200, 32, seed=13)),
])
def test_frozen_rejects_mutations(builder, sample) -> None:
    """After freeze() every mutator raises RuntimeError with a
    message that names the operation, so a user who sees the
    exception in prod can tell which call they need to gate."""
    idx = builder()
    idx.freeze()
    assert idx.frozen is True
    with pytest.raises(RuntimeError, match="add_batch"):
        idx.add_batch([999], sample[:1])
    # Only delete() is exposed on all four index types with the same
    # semantics.  fit() and close() are tested separately below.
    if hasattr(idx, "delete"):
        # delete is allowed to return False for unknown ids without
        # mutating — but it must still raise once frozen, since the
        # code path touches internal state unconditionally.
        with pytest.raises(RuntimeError, match="delete"):
            idx.delete(0)


def test_frozen_pq_rejects_fit() -> None:
    """fit() on PQSnapIndex is only called once before add_batch —
    but a frozen index must reject even the first call attempt
    (invariant is: no writes through the class, period)."""
    corpus = _unit_gaussian(60, 32, seed=20)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True, seed=0)
    idx.freeze()
    with pytest.raises(RuntimeError, match="fit"):
        idx.fit(corpus)


def test_frozen_ivfpq_rejects_close_and_fit() -> None:
    corpus = _unit_gaussian(200, 32, seed=21)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(200)), corpus)
    idx.freeze()
    with pytest.raises(RuntimeError, match="close"):
        idx.close()

    # fit() on a freshly-constructed-but-frozen index must also raise,
    # not just on already-fitted instances.
    fresh = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=0,
    )
    fresh.freeze()
    with pytest.raises(RuntimeError, match="fit"):
        fresh.fit(corpus)


@pytest.mark.parametrize("builder", [
    _build_snap, _build_residual, _build_pq, _build_ivfpq,
])
def test_unfreeze_restores_mutation(builder) -> None:
    idx = builder()
    idx.freeze()
    idx.unfreeze()
    assert idx.frozen is False
    # Mutation should now succeed.  Pick delete because all 4 indexes
    # have it and it's a no-op on an unknown id.
    if hasattr(idx, "delete"):
        removed = idx.delete(99999)  # likely unknown
        assert removed in (True, False)  # just must not raise


def test_snapindex_freeze_prewarms_cache() -> None:
    """``SnapIndex._cache`` is lazily built on first search.  If we
    freeze before warming it, two concurrent searches would both see
    it as None and race to assign it.  ``freeze()`` must pre-warm."""
    idx = _build_snap()
    assert idx._cache is None  # cache is lazy, never touched yet
    idx.freeze()
    assert idx._cache is not None
    # The pre-warm must reach the same state as the lazy path.
    assert idx._cache.dtype == np.float16
    assert idx._cache.shape[0] == len(idx)


def test_snapindex_frozen_concurrent_search_is_race_free() -> None:
    """Stress the pre-warmed cache path from many threads.  Asserts
    every result is identical to a serial search — if the cache were
    mutated concurrently, some threads would see a half-initialised
    ndarray and either crash or return wrong scores."""
    idx = _build_snap()
    q = _unit_gaussian(1, 32, seed=70)[0]
    idx.freeze()
    serial = idx.search(q, k=5)

    def worker(_: int) -> list:
        return idx.search(q, k=5)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(64)))

    for r in results:
        assert [h[0] for h in r] == [h[0] for h in serial]


def test_frozen_search_still_works() -> None:
    """The whole point of freeze() — reads must keep functioning
    after mutations are gated."""
    idx = _build_ivfpq()
    q = _unit_gaussian(1, 32, seed=30)[0]
    before = idx.search(q, k=3, nprobe=4)
    idx.freeze()
    after = idx.search(q, k=3, nprobe=4)
    assert [h[0] for h in before] == [h[0] for h in after]


# ──────────────────────────────────────────────────────────────────── #
# Concurrent-search stress tests                                       #
# ──────────────────────────────────────────────────────────────────── #

def test_frozen_ivfpq_search_from_multiple_threads() -> None:
    """Fire 64 concurrent searches from 4 worker threads against a
    frozen IVFPQSnapIndex.  With the freeze contract in place no
    lock is taken on the hot path, so we're testing that the hot
    path genuinely has no mutation to race on.  The assertion is
    just that every call produces a non-empty result and no
    exception propagates — which would have failed pre-freeze if an
    internal lazy cache was being mutated."""
    corpus = _clustered(1_000, 64, n_clusters=20, seed=40)
    idx = IVFPQSnapIndex(
        dim=64, nlist=16, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus[:500])
    idx.add_batch(list(range(1_000)), corpus)
    idx.freeze()

    queries = _clustered(16, 64, n_clusters=20, seed=41)

    def worker(i: int) -> list:
        q = queries[i % len(queries)]
        return idx.search(q, k=5, nprobe=8)

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(worker, range(64)))

    assert len(results) == 64
    for hits in results:
        assert len(hits) == 5


def test_search_batch_concurrent_lazy_executor_init_is_race_free(
    monkeypatch,
) -> None:
    """Before the _executor_lock was added, two concurrent
    ``search_batch(num_threads > 1)`` calls could both see
    ``self._executor is None`` and each create their own pool —
    leaking the first one's worker threads.

    Just asserting that calls succeed is not enough: the final cached
    pool would still have num_threads workers even if an earlier
    duplicate leaked.  So we monkeypatch the module-level
    ``ThreadPoolExecutor`` that ``search_batch`` consults and count
    how many pools get constructed under concurrency.  A correct
    implementation constructs exactly one.
    """
    from snapvec import _ivfpq as ivfpq_mod

    corpus = _clustered(500, 32, n_clusters=10, seed=50)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus[:250])
    idx.add_batch(list(range(500)), corpus)
    # Intentionally don't freeze — we want the lazy-init path hot.
    assert idx._executor is None

    created: list[ThreadPoolExecutor] = []
    real_tpe = ivfpq_mod.ThreadPoolExecutor

    def counting_tpe(*args, **kwargs):
        pool = real_tpe(*args, **kwargs)
        created.append(pool)
        return pool

    monkeypatch.setattr(ivfpq_mod, "ThreadPoolExecutor", counting_tpe)

    queries = _clustered(8, 32, n_clusters=10, seed=51)

    def worker(_: int) -> list:
        return idx.search_batch(queries, k=3, nprobe=4, num_threads=4)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(worker, range(8)))

    assert len(results) == 8
    for batch_result in results:
        assert len(batch_result) == 8
    assert idx._executor_workers == 4
    # The load-bearing assertion: exactly one pool was constructed
    # across 8 concurrent first-time-ish callers.
    assert len(created) == 1, (
        f"lazy executor init leaked {len(created)} pools; expected 1"
    )
    idx.close()


def test_search_batch_rejects_changing_num_threads_after_init() -> None:
    """Once the executor is initialised, a later call asking for a
    different num_threads must raise instead of silently shutting
    down the shared pool — another caller may be about to submit
    work to it."""
    corpus = _clustered(200, 32, n_clusters=10, seed=60)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus[:150])
    idx.add_batch(list(range(200)), corpus)

    queries = _clustered(4, 32, seed=61)
    idx.search_batch(queries, k=3, num_threads=2)
    assert idx._executor_workers == 2
    with pytest.raises(ValueError, match="cannot change num_threads"):
        idx.search_batch(queries, k=3, num_threads=4)
    idx.close()
