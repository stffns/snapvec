"""Compiled kernels for snapvec hot paths (numba)."""
from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

_N_PARALLEL_THRESHOLD = 2000


# ── ADC scoring kernels ────────────────────────────────────────────

@nb.njit(cache=True, boundscheck=False)
def _adc_serial(
    lut: NDArray[np.float32],
    codes: NDArray[np.uint8],
    scores: NDArray[np.float32],
) -> None:
    M = codes.shape[0]
    n = codes.shape[1]
    for i in range(n):
        s = np.float32(0.0)
        for j in range(M):
            s += lut[j, codes[j, i]]
        scores[i] += s


@nb.njit(cache=True, boundscheck=False, parallel=True)
def _adc_parallel(
    lut: NDArray[np.float32],
    codes: NDArray[np.uint8],
    scores: NDArray[np.float32],
) -> None:
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
    if parallel and codes.shape[1] >= _N_PARALLEL_THRESHOLD:
        _adc_parallel(lut, codes, scores)
    else:
        _adc_serial(lut, codes, scores)


# ── Fused gather + ADC kernels ─────────────────────────────────────

@nb.njit(cache=True, boundscheck=False)
def _fused_serial(
    all_codes: NDArray[np.uint8],
    row_idx: NDArray[np.int64],
    coarse_offsets: NDArray[np.float32],
    lut: NDArray[np.float32],
    scores: NDArray[np.float32],
) -> None:
    M = all_codes.shape[0]
    n = len(row_idx)
    for i in range(n):
        r = row_idx[i]
        acc = coarse_offsets[i]
        for j in range(M):
            acc += lut[j, all_codes[j, r]]
        scores[i] = acc


@nb.njit(cache=True, boundscheck=False, parallel=True)
def _fused_parallel(
    all_codes: NDArray[np.uint8],
    row_idx: NDArray[np.int64],
    coarse_offsets: NDArray[np.float32],
    lut: NDArray[np.float32],
    scores: NDArray[np.float32],
) -> None:
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
    if parallel and len(row_idx) >= _N_PARALLEL_THRESHOLD:
        _fused_parallel(all_codes, row_idx, coarse_offsets, lut, scores)
    else:
        _fused_serial(all_codes, row_idx, coarse_offsets, lut, scores)
