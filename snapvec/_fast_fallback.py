"""Pure-NumPy fallback kernels when Cython extension is not available."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def adc_colmajor(
    lut: NDArray[np.float32],
    codes: NDArray[np.uint8],
    scores: NDArray[np.float32],
    parallel: bool = False,
) -> None:
    M = codes.shape[0]
    for j in range(M):
        scores += lut[j][codes[j]]


def fused_gather_adc(
    all_codes: NDArray[np.uint8],
    row_idx: NDArray[np.int64],
    coarse_offsets: NDArray[np.float32],
    lut: NDArray[np.float32],
    scores: NDArray[np.float32],
    parallel: bool = True,
) -> None:
    M = all_codes.shape[0]
    cat = all_codes[:, row_idx]
    scores[:] = coarse_offsets
    for j in range(M):
        scores += lut[j][cat[j]]
