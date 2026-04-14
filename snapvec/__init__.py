"""snapvec — compressed vector index for fast ANN search.

Fast approximate nearest-neighbor search via randomized Hadamard transform
and optimal Gaussian scalar quantization.  Pure NumPy, no heavy dependencies.

    >>> from snapvec import SnapIndex
    >>> idx = SnapIndex(dim=384, bits=4)
    >>> idx.add_batch(ids, vectors)
    >>> results = idx.search(query, k=10)

See https://github.com/stffns/snapvec for documentation.
"""
from __future__ import annotations

from ._index import SnapIndex
from ._codebooks import get_codebook
from ._residual import ResidualSnapIndex
from ._rotation import rht, padded_dim

__version__ = "0.3.0"
__all__ = [
    "SnapIndex",
    "ResidualSnapIndex",
    "get_codebook",
    "rht",
    "padded_dim",
]
