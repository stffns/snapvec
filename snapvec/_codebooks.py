"""Hardcoded Lloyd-Max optimal codebooks for the standard normal N(0,1).

After the randomized Hadamard transform, each coordinate of a unit vector
is approximately i.i.d. N(0, 1/d).  We quantize in the scaled space
x_scaled = x_rotated * sqrt(d) ≈ N(0, 1), so these codebooks apply directly.

The codebooks were precomputed offline using 100 iterations of the Lloyd-Max
algorithm with Gaussian integration (scipy.integrate.quad).  They are hardcoded
here to eliminate the scipy runtime dependency — numpy only.

References
----------
Max, J. (1960). Quantizing for minimum distortion.
    IRE Transactions on Information Theory, 6(1), 7-12.
Zandieh et al. (2025). TurboQuant: Online Vector Quantization with
    Near-optimal Distortion Rate.  arXiv:2504.19874.
"""
from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Precomputed Lloyd-Max codebooks for N(0,1), b ∈ {2, 3, 4}
#
# Layout: (centroids, boundaries)
#   centroids  — 2^b reconstruction values, ascending
#   boundaries — 2^b - 1 decision thresholds (boundaries[i] = midpoint between
#                centroids[i] and centroids[i+1])
#
# Symmetry: centroids[i] = -centroids[2^b - 1 - i]  (antisymmetric around 0)
#           boundaries[i] = -boundaries[2^b - 2 - i]  (antisymmetric around 0)
# ──────────────────────────────────────────────────────────────────────────────

_CODEBOOKS: dict[int, tuple[list[float], list[float]]] = {
    # 2-bit: 4 levels
    2: (
        [-1.5104, -0.4528,  0.4528,  1.5104],
        [-0.9816,  0.0000,  0.9816],
    ),
    # 3-bit: 8 levels
    3: (
        [-2.1520, -1.3439, -0.7560, -0.2451,  0.2451,  0.7560,  1.3439,  2.1520],
        [-1.7480, -1.0500, -0.5006,  0.0000,  0.5006,  1.0500,  1.7480],
    ),
    # 4-bit: 16 levels
    4: (
        [
            -2.7326, -2.0690, -1.6180, -1.2562,
            -0.9424, -0.6424, -0.3685, -0.1227,
             0.1227,  0.3685,  0.6424,  0.9424,
             1.2562,  1.6180,  2.0690,  2.7326,
        ],
        [
            -2.4008, -1.8435, -1.4372, -1.0993,
            -0.7996, -0.5224, -0.2582,  0.0000,
             0.2582,  0.5224,  0.7996,  1.0993,
             1.4372,  1.8435,  2.4008,
        ],
    ),
}


def get_codebook(bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids, boundaries) for the given bit-width.

    Parameters
    ----------
    bits : int
        Quantization bits.  Must be 2, 3, or 4.

    Returns
    -------
    centroids : np.ndarray, shape (2**bits,), float32
        Reconstruction values.
    boundaries : np.ndarray, shape (2**bits - 1,), float32
        Decision thresholds for np.searchsorted.
    """
    if bits not in _CODEBOOKS:
        raise ValueError(f"bits must be 2, 3, or 4; got {bits}")
    c, b = _CODEBOOKS[bits]
    return (
        np.array(c, dtype=np.float32),
        np.array(b, dtype=np.float32),
    )
