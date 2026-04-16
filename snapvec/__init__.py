"""snapvec — compressed vector index for fast ANN search.

Fast approximate nearest-neighbor search via randomized Hadamard transform
and optimal Gaussian scalar quantization.  NumPy + Numba compiled kernels.

    >>> from snapvec import SnapIndex
    >>> idx = SnapIndex(dim=384, bits=4)
    >>> idx.add_batch(ids, vectors)
    >>> results = idx.search(query, k=10)

See https://github.com/stffns/snapvec for documentation.
"""
from __future__ import annotations

from ._codebooks import get_codebook
from ._index import SnapIndex
from ._ivfpq import IVFPQSnapIndex
from ._pq import PQSnapIndex
from ._residual import ResidualSnapIndex
from ._rotation import padded_dim, rht

__version__ = "0.8.1"
__all__ = [
    "SnapIndex",
    "PQSnapIndex",
    "IVFPQSnapIndex",
    "ResidualSnapIndex",
    "get_codebook",
    "rht",
    "padded_dim",
]
