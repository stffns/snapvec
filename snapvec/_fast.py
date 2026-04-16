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
