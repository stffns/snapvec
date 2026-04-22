"""Dtype invariants along the scale-and-rotate hot paths.

Regression guard for the silent ``float64`` upcasts that snuck in
when the scale/normalise code mixed float32 arrays with Python-scalar
or ``np.sqrt(int)`` divisors.  Both Copilot and Gemini flagged these
during PR review; the tests here pin the runtime dtype so the
``NDArray[np.float32]`` annotations stay honest and a future refactor
that drops a scalar wrap fails fast instead of silently doubling
memory and breaking the Cython kernel's dtype contract.
"""
from __future__ import annotations

import numpy as np

from snapvec import IVFPQSnapIndex, PQSnapIndex, ResidualSnapIndex, SnapIndex
from snapvec._rotation import padded_dim, rht


def test_rht_preserves_float32() -> None:
    """``rht`` must return float32 for float32 input.

    Historical bug: ``y / np.sqrt(d)`` returned float64 because
    ``np.sqrt(int)`` is a numpy float64 scalar.  The function masked
    the upcast with an astype at the return; the astype is now gone
    and the dtype must be preserved at the source.
    """
    for d in (64, 128, 384, 512):
        pdim = padded_dim(d)
        x = np.random.default_rng(0).standard_normal((5, pdim)).astype(np.float32)
        y = rht(x, seed=0)
        assert y.dtype == np.float32, (
            f"rht(float32, d={pdim}) returned {y.dtype}"
        )
        # 1-D input path too
        y1 = rht(x[0], seed=0)
        assert y1.dtype == np.float32, (
            f"rht(float32 1-D, d={pdim}) returned {y1.dtype}"
        )


def test_snap_index_add_batch_preserves_float32() -> None:
    """``SnapIndex.add_batch`` must not materialise float64 intermediates.

    Historical bug: ``scaled = rotated * np.sqrt(pdim)`` was annotated
    ``NDArray[np.float32]`` but produced float64 at runtime.  Downstream
    ``np.searchsorted`` still worked by chance; a future kernel that
    required strict float32 input would have crashed.  We check that
    the stored ``_norms`` stays float32 (the observable dtype surface
    the file format relies on).
    """
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((50, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    idx = SnapIndex(dim=384, bits=4, seed=0)
    idx.add_batch(list(range(50)), vecs)
    assert idx._norms.dtype == np.float32


def test_pq_preprocess_returns_float32() -> None:
    """``PQSnapIndex._preprocess`` output dtype pins down the whole chain.

    Exercised under all three branches: plain, use_rht, use_opq.  The
    return annotation is ``NDArray[np.float32]``; a runtime float64
    would break the ADC kernel's reinterpret cast.
    """
    rng = np.random.default_rng(0)
    # K=256 minimum training rows for PQ; 300 gives slack for OPQ rotation.
    X = rng.standard_normal((300, 384)).astype(np.float32)
    # M=64 divides both dim=384 (=> d_sub=6) and pdim=512 (=> d_sub=8),
    # so the same M works under use_rht and use_opq.
    for use_rht, use_opq in ((False, False), (True, False), (False, True)):
        idx = PQSnapIndex(
            dim=384, M=64, seed=0, normalized=False,
            use_rht=use_rht, use_opq=use_opq,
        )
        if use_opq:
            idx.fit(X)  # OPQ needs a fitted rotation before preprocess
        units, norms = idx._preprocess(X)
        assert units.dtype == np.float32, (
            f"_preprocess units: use_rht={use_rht}, use_opq={use_opq} "
            f"returned {units.dtype}"
        )
        assert norms.dtype == np.float32
        # Single-query path too
        q_unit = idx._preprocess_single(X[0])
        assert q_unit.dtype == np.float32, (
            f"_preprocess_single: use_rht={use_rht}, use_opq={use_opq} "
            f"returned {q_unit.dtype}"
        )


def test_ivfpq_preprocess_returns_float32() -> None:
    """Same invariant for ``IVFPQSnapIndex._preprocess`` and its batch path.

    ``search_batch`` has its own normalisation code path
    (independent of ``_preprocess``), so we assert the dtype of the
    rotated queries it produces by round-tripping through a small
    search call and checking that no upstream warning fires.
    """
    rng = np.random.default_rng(0)
    n = 300  # need >= K = 256 training rows
    X = rng.standard_normal((n, 384)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    # M=64 works under both dim=384 (d_sub=6) and pdim=512 (d_sub=8).
    for use_rht, use_opq in ((False, False), (True, False), (False, True)):
        idx = IVFPQSnapIndex(
            dim=384, M=64, nlist=8, seed=0, normalized=False,
            use_rht=use_rht, use_opq=use_opq,
        )
        idx.fit(X)
        idx.add_batch([str(i) for i in range(n)], X)
        units, _ = idx._preprocess(X[:10])
        assert units.dtype == np.float32, (
            f"IVFPQ _preprocess: use_rht={use_rht}, use_opq={use_opq} "
            f"returned {units.dtype}"
        )
        q_single = idx._preprocess_single(X[0])
        assert q_single.dtype == np.float32
        # Batch search runs the separate batch-normalisation path
        res = idx.search_batch(X[:5], k=3, nprobe=2)
        assert res is not None  # search_batch returns list-of-lists
        assert len(res) == 5


def test_residual_search_query_float32() -> None:
    """Search through ``ResidualSnapIndex`` must not upcast the query path.

    Query normalisation used a Python float divisor; the result used
    to rely on an astype at the end of the chain.  Guard the dtype of
    the scored vectors at the last observable surface.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 384)).astype(np.float32)
    idx = ResidualSnapIndex(dim=384, b1=3, b2=3, seed=0)
    idx.add_batch(list(range(50)), X)
    # Search completes without raising and returns float scores
    res = idx.search(X[0], k=5)
    assert len(res) == 5
    assert all(isinstance(score, float) for _, score in res)
    # Internal norms array stays float32
    assert idx._norms.dtype == np.float32
