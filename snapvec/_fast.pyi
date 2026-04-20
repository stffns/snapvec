"""Type stubs for the compiled Cython kernels (``_fast.pyx``).

The real module is built from Cython and does not ship a ``.pyi``
from the compiler; this stub lets ``mypy --strict`` see the same
Python-level shapes the Cython kernels expose to callers.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def adc_colmajor(
    lut: NDArray[np.float32],
    codes: NDArray[np.uint8],
    scores: NDArray[np.float32],
    parallel: bool = ...,
) -> None: ...


def fused_gather_adc(
    all_codes: NDArray[np.uint8],
    row_idx: NDArray[np.int64],
    coarse_offsets: NDArray[np.float32],
    lut: NDArray[np.float32],
    scores: NDArray[np.float32],
    parallel: bool = ...,
) -> None: ...
