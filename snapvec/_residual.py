"""Residual coarse-to-fine TurboQuant index.

Stores two Lloyd-Max stages per vector:

1. **Coarse** (``b1`` bits): the usual TurboQuant quantization of the
   rotated, per-coord-standardized vector.
2. **Residual** (``b2`` bits): the stage-1 error rescaled by
   ``σ_r = √ε(b1)`` (the theoretical Lloyd-Max distortion at ``b1``
   bits for N(0, 1)) and quantized again with the ``b2``-bit codebook.

Reconstruction is ``x̃ ≈ decode(c1) + σ_r · decode(c2)``.  The two
stages reuse the same ``{2, 3, 4}``-bit codebooks snapvec already ships,
so no new training is needed.

Empirically on BGE-small/SciFact, this opens operating points that
uniform single-stage snapvec cannot reach (5, 6, 7 bits/coord with
recall@10 up to 0.956), at the cost of roughly 2× storage.  See
``experiments/bench_residual_coarse_fine.py`` for the measurements.

Tight bit-packing of both code streams is a follow-up — v1 stores
each stage as one byte per coordinate.  At ``pdim = 512`` that is
already 12× compression vs float32 for a 6-bit combined rate.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ._codebooks import get_codebook
from ._file_format import ChecksumWriter, save_with_checksum_atomic, verify_checksum
from ._freezable import FreezableIndex
from ._rotation import padded_dim, rht


_MAX_ID_BYTES = 0xFFFF  # file format stores id length as uint16


def _decode_id(raw: str) -> Any:
    """Round-trip numeric-looking ids back to int/float, matching SnapIndex."""
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


# Empirical Lloyd-Max MSE on N(0, 1) at 2M samples — used to compute
# the theoretical residual std σ_r.  Verified to match measured σ_r
# to within <2% in the full-scale benchmark.
_LLOYD_MSE: dict[int, float] = {2: 0.11767, 3: 0.03454, 4: 0.00961}

_MAGIC = b"SNPR"
_VERSION = 1


def _sigma_r(b1: int) -> float:
    """Theoretical residual std: σ_r = √ε(b1)."""
    return float(np.sqrt(_LLOYD_MSE[b1]))


class ResidualSnapIndex(FreezableIndex):
    """Two-stage Lloyd-Max TurboQuant index.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    b1 : int, default 3
        Coarse-stage bits per coordinate.  Must be in ``{2, 3, 4}``.
    b2 : int, default 3
        Residual-stage bits per coordinate.  Must be in ``{2, 3, 4}``.
    seed : int, default 0
        RHT rotation seed.  Must match across build and query.
    normalized : bool, default False
        When True, inputs are assumed unit-length and the per-vector
        norm is not computed or stored.

    Notes
    -----
    Bit-packing is not applied in this version — each stage stores
    one uint8 per coordinate.  A 6-bit combined rate thus takes
    ~16 B/coord pre-pack; tight packing would bring it to 6/8 B/coord.
    """

    def __init__(
        self,
        dim: int,
        b1: int = 3,
        b2: int = 3,
        seed: int = 0,
        normalized: bool = False,
    ) -> None:
        if b1 not in (2, 3, 4) or b2 not in (2, 3, 4):
            raise ValueError(
                f"b1, b2 must each be 2, 3, or 4; got b1={b1}, b2={b2}"
            )
        self.dim = dim
        self.b1 = b1
        self.b2 = b2
        self.seed = seed
        self.normalized = normalized

        self._pdim = padded_dim(dim)
        self._sigma_r = _sigma_r(b1)
        self._cent1, self._thr1 = get_codebook(b1)
        self._cent2, self._thr2 = get_codebook(b2)

        self._ids: list[Any] = []
        self._id_to_pos: dict[Any, int] = {}
        self._codes1: NDArray[np.uint8] = np.zeros((0, self._pdim), dtype=np.uint8)
        self._codes2: NDArray[np.uint8] = np.zeros((0, self._pdim), dtype=np.uint8)
        self._norms: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────── #
    # build                                                             #
    # ──────────────────────────────────────────────────────────────── #

    def add(self, id: Any, vector: NDArray[np.float32]) -> None:
        self._check_not_frozen("add")
        self.add_batch([id], np.asarray(vector, dtype=np.float32)[None, :])

    def add_batch(
        self, ids: list[Any], vectors: NDArray[np.float32]
    ) -> None:
        self._check_not_frozen("add_batch")
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"vectors must be shape (n, {self.dim}); got {arr.shape}"
            )
        n = len(arr)
        if len(ids) != n:
            raise ValueError(
                f"ids and vectors must have the same length; "
                f"got {len(ids)} ids and {n} vectors"
            )
        if n == 0:
            return

        if self.normalized:
            units = arr
            batch_norms = None  # not stored in normalized mode
        else:
            raw_norms = np.linalg.norm(arr, axis=1)
            safe = cast(
                "NDArray[np.float32]",
                np.where(raw_norms > 1e-10, raw_norms, np.float32(1.0)),
            )
            units = cast("NDArray[np.float32]", arr / safe[:, None])
            batch_norms = np.where(raw_norms > 1e-10, raw_norms, 0.0).astype(np.float32)

        pdim = self._pdim
        padded = np.zeros((n, pdim), dtype=np.float32)
        padded[:, : self.dim] = units
        rotated = rht(padded, self.seed)
        scaled = (rotated * np.sqrt(pdim)).astype(np.float32)

        c1 = np.clip(
            np.searchsorted(self._thr1, scaled),
            0, (2 ** self.b1) - 1,
        ).astype(np.uint8)
        dec1 = self._cent1[c1]
        residual = (scaled - dec1) / self._sigma_r
        c2 = np.clip(
            np.searchsorted(self._thr2, residual),
            0, (2 ** self.b2) - 1,
        ).astype(np.uint8)

        start = len(self._ids)
        self._ids.extend(ids)
        for i, id_val in enumerate(ids):
            self._id_to_pos[id_val] = start + i

        self._codes1 = c1 if len(self._codes1) == 0 else np.vstack([self._codes1, c1])
        self._codes2 = c2 if len(self._codes2) == 0 else np.vstack([self._codes2, c2])
        if batch_norms is not None:
            self._norms = (
                batch_norms
                if len(self._norms) == 0
                else np.concatenate([self._norms, batch_norms])
            )

    # ──────────────────────────────────────────────────────────────── #
    # search                                                            #
    # ──────────────────────────────────────────────────────────────── #

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        rerank_M: int | None = None,
    ) -> list[tuple[Any, float]]:
        """Approximate cosine top-k.

        Parameters
        ----------
        query : NDArray[np.float32]
            Query vector (need not be normalized).
        k : int, default 10
            Number of results to return.
        rerank_M : int | None, default None
            When set to an int, do a coarse pass (b1 bits) over the
            full corpus to select top-``M`` candidates, then rerank
            those using the full ``(b1 + b2)`` reconstruction.  When
            None, use full reconstruction over the whole corpus.
            ``M = 100`` is sufficient for convergence at ``k ≤ 10``
            on the tested datasets.
        """
        if not self._ids:
            return []
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if rerank_M is not None and rerank_M < k:
            raise ValueError(
                f"rerank_M must be >= k; got rerank_M={rerank_M}, k={k}"
            )
        q = np.asarray(query, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-10:
            return []
        pdim = self._pdim
        q_unit = q / q_norm
        q_padded = np.zeros(pdim, dtype=np.float32)
        q_padded[: self.dim] = q_unit
        q_rot = rht(q_padded[None, :], self.seed)[0]
        q_scaled = (q_rot * np.sqrt(pdim)).astype(np.float32)

        if rerank_M is None:
            decoded = self._cent1[self._codes1] + self._sigma_r * self._cent2[self._codes2]
            scores = decoded @ q_scaled
            rows = np.arange(len(scores))
        else:
            coarse = self._cent1[self._codes1]
            cscores = coarse @ q_scaled
            M = min(rerank_M, len(cscores))
            cand = np.argpartition(-cscores, M - 1)[:M]
            fine = (
                self._cent1[self._codes1[cand]]
                + self._sigma_r * self._cent2[self._codes2[cand]]
            )
            scores = fine @ q_scaled
            rows = cand

        # Score is the dot product in the rotated scaled space, which
        # equals (pdim · cosine) in expectation — divide to recover a
        # cosine-like value, then multiply by ‖v‖ if normalized=False.
        scores = scores / pdim
        if not self.normalized:
            scores = scores * self._norms[rows]

        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[int(rows[i])], float(scores[i])) for i in top]

    # ──────────────────────────────────────────────────────────────── #
    # diagnostics + persistence                                         #
    # ──────────────────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"ResidualSnapIndex(dim={self.dim}, b1={self.b1}, b2={self.b2}, "
            f"n={len(self._ids)})"
        )

    def stats(self) -> dict[str, Any]:
        pdim = self._pdim
        bytes_per_vec = 2 * pdim + (4 if not self.normalized else 0)
        return {
            "n": len(self._ids),
            "dim": self.dim,
            "padded_dim": pdim,
            "b1": self.b1,
            "b2": self.b2,
            "sigma_r": self._sigma_r,
            "bytes_per_vec": bytes_per_vec,
            "compression_ratio": (self.dim * 4) / bytes_per_vec
            if bytes_per_vec > 0 else 0.0,
        }

    def save(self, path: str | Path) -> None:
        flags = 0
        if self.normalized:
            flags |= 1
        n = len(self._ids)

        def _write(f: "ChecksumWriter") -> None:
            f.write(_MAGIC)
            f.write(struct.pack("<IIIIIIII", _VERSION, self.dim, self.b1,
                                self.b2, self.seed, n, flags, self._pdim))
            if n > 0:
                f.write(self._codes1.tobytes())
                f.write(self._codes2.tobytes())
                if not self.normalized:
                    f.write(self._norms.tobytes())
                for id_val in self._ids:
                    s = str(id_val).encode("utf-8")
                    if len(s) > _MAX_ID_BYTES:
                        raise ValueError(
                            f"id {id_val!r} encodes to {len(s)} UTF-8 bytes; "
                            f"the file format stores id length as uint16 "
                            f"(max {_MAX_ID_BYTES})"
                        )
                    f.write(struct.pack("<H", len(s)))
                    f.write(s)

        save_with_checksum_atomic(path, _write)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualSnapIndex":
        path = Path(path)
        verify_checksum(path)  # no-op for legacy files without a trailer
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"bad magic {magic!r}, expected {_MAGIC!r}")
            version, dim, b1, b2, seed, n, flags, pdim = struct.unpack(
                "<IIIIIIII", f.read(32)
            )
            if version != _VERSION:
                raise ValueError(
                    f"unsupported .snpr version {version}; this build of "
                    f"snapvec writes and reads version {_VERSION}.  If "
                    f"{version} > {_VERSION} this file was written by a "
                    f"newer snapvec -- upgrade via `pip install -U snapvec`."
                )
            normalized = bool(flags & 1)
            idx = cls(dim=dim, b1=b1, b2=b2, seed=seed, normalized=normalized)
            if pdim != idx._pdim:
                raise ValueError(
                    f"on-disk pdim={pdim} differs from computed pdim={idx._pdim} "
                    f"for dim={dim}; file may be corrupted or from a future version"
                )
            if n > 0:
                idx._codes1 = np.frombuffer(f.read(n * pdim), dtype=np.uint8).reshape(n, pdim).copy()
                idx._codes2 = np.frombuffer(f.read(n * pdim), dtype=np.uint8).reshape(n, pdim).copy()
                if not normalized:
                    idx._norms = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
                idx._ids = []
                for _ in range(n):
                    (ln,) = struct.unpack("<H", f.read(2))
                    idx._ids.append(_decode_id(f.read(ln).decode("utf-8")))
                idx._id_to_pos = {id_val: i for i, id_val in enumerate(idx._ids)}
        return idx
