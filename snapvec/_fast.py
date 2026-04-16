"""Optional compiled kernels for snapvec hot paths.

Requires ``numba`` (install via ``pip install snapvec[fast]``).
All functions have a pure-NumPy fallback so the package works
without numba installed.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_HAS_NUMBA = False

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    pass


# ── ADC scoring kernel ─────────────────────────────────────────────

def _adc_colmajor_numpy(
    lut: NDArray[np.float32],
    codes: NDArray[np.uint8],
    scores: NDArray[np.float32],
) -> None:
    """Pure-NumPy fallback: per-subspace loop."""
    M = codes.shape[0]
    for j in range(M):
        scores += lut[j][codes[j]]


if _HAS_NUMBA:
    @nb.njit(cache=True, boundscheck=False)
    def _adc_colmajor_numba(
        lut: NDArray[np.float32],
        codes: NDArray[np.uint8],
        scores: NDArray[np.float32],
    ) -> None:
        """Fused ADC: one compiled loop over (candidates x subspaces)."""
        M = codes.shape[0]
        n = codes.shape[1]
        for i in range(n):
            s = np.float32(0.0)
            for j in range(M):
                s += lut[j, codes[j, i]]
            scores[i] += s

    @nb.njit(cache=True, boundscheck=False, parallel=True)
    def _adc_colmajor_numba_par(
        lut: NDArray[np.float32],
        codes: NDArray[np.uint8],
        scores: NDArray[np.float32],
    ) -> None:
        """Parallel fused ADC: candidates split across cores."""
        M = codes.shape[0]
        n = codes.shape[1]
        for i in nb.prange(n):
            s = np.float32(0.0)
            for j in range(M):
                s += lut[j, codes[j, i]]
            scores[i] += s


def adc_colmajor(
    lut: NDArray[np.float32],
    codes: NDArray[np.uint8],
    scores: NDArray[np.float32],
    parallel: bool = False,
) -> None:
    """Score candidates via ADC, dispatching to the fastest backend.

    Parameters
    ----------
    lut : (M, K) float32
        Per-subspace lookup table.
    codes : (M, n) uint8
        Quantisation codes, column-major.
    scores : (n,) float32
        Accumulator -- coarse offsets should already be written here.
    parallel : bool
        Use multi-threaded kernel (only effective with numba).
    """
    if _HAS_NUMBA:
        if parallel and codes.shape[1] >= 2000:
            _adc_colmajor_numba_par(lut, codes, scores)
        else:
            _adc_colmajor_numba(lut, codes, scores)
    else:
        _adc_colmajor_numpy(lut, codes, scores)


# ── Fused gather + ADC kernel ──────────────────────────────────────

def _fused_gather_adc_numpy(
    all_codes: NDArray[np.uint8],
    row_idx: NDArray[np.int64],
    coarse_offsets: NDArray[np.float32],
    lut: NDArray[np.float32],
    scores: NDArray[np.float32],
) -> None:
    """Pure-NumPy fallback: gather into temp buffer then ADC loop."""
    M = all_codes.shape[0]
    n = len(row_idx)
    cat = all_codes[:, row_idx]
    for i in range(n):
        scores[i] = coarse_offsets[i]
    for j in range(M):
        scores += lut[j][cat[j]]


if _HAS_NUMBA:
    @nb.njit(cache=True, boundscheck=False)
    def _fused_gather_adc_numba(
        all_codes: NDArray[np.uint8],
        row_idx: NDArray[np.int64],
        coarse_offsets: NDArray[np.float32],
        lut: NDArray[np.float32],
        scores: NDArray[np.float32],
    ) -> None:
        """Fused gather+ADC: serial, safe inside ThreadPoolExecutor."""
        M = all_codes.shape[0]
        n = len(row_idx)
        for i in range(n):
            r = row_idx[i]
            acc = coarse_offsets[i]
            for j in range(M):
                acc += lut[j, all_codes[j, r]]
            scores[i] = acc

    @nb.njit(cache=True, boundscheck=False, parallel=True)
    def _fused_gather_adc_numba_par(
        all_codes: NDArray[np.uint8],
        row_idx: NDArray[np.int64],
        coarse_offsets: NDArray[np.float32],
        lut: NDArray[np.float32],
        scores: NDArray[np.float32],
    ) -> None:
        """Fused gather+ADC: parallel via prange."""
        M = all_codes.shape[0]
        n = len(row_idx)
        for i in nb.prange(n):
            r = row_idx[i]
            acc = coarse_offsets[i]
            for j in range(M):
                acc += lut[j, all_codes[j, r]]
            scores[i] = acc


def fused_gather_adc(
    all_codes: NDArray[np.uint8],
    row_idx: NDArray[np.int64],
    coarse_offsets: NDArray[np.float32],
    lut: NDArray[np.float32],
    scores: NDArray[np.float32],
    parallel: bool = True,
) -> None:
    """Score candidates by reading codes directly (no intermediate buffer).

    Parameters
    ----------
    all_codes : (M, N) uint8
        Full code storage (column-major).
    row_idx : (n,) int64
        Indices into all_codes axis=1 for each candidate.
    coarse_offsets : (n,) float32
        Per-candidate coarse dot-product offset.
    lut : (M, K) float32
        Per-subspace lookup table.
    scores : (n,) float32
        Output scores.
    parallel : bool
        Use multi-threaded kernel. Set False when called from
        within a ThreadPoolExecutor to avoid deadlocks.
    """
    if _HAS_NUMBA:
        if parallel and len(row_idx) >= 2000:
            _fused_gather_adc_numba_par(all_codes, row_idx,
                                        coarse_offsets, lut, scores)
        else:
            _fused_gather_adc_numba(all_codes, row_idx,
                                    coarse_offsets, lut, scores)
    else:
        _fused_gather_adc_numpy(all_codes, row_idx, coarse_offsets,
                                lut, scores)
