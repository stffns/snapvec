"""SnapIndex — compressed approximate nearest-neighbor search.

Two search modes
----------------
HadaMax_mse  (use_prod=False, default)
    Pipeline : normalize → pad to 2^k → RHT → Lloyd-Max(b bits) → uint8
    Score    : dot(centroids[idx_i], q_rot) / d    ← biased, low variance
    Recall   : ~0.93 recall@10 at d=384, 4-bit  (vs float32 brute-force)
    Cost     : 1 matmul (float16)

HadaMax_prod (use_prod=True)
    Pipeline : same RHT, but Lloyd-Max at (b-1) bits + 1-bit QJL residual
    Score    : mse_score + sqrt(π/2)/d · ‖r‖ · dot(S·q, sign(S·r))
    Recall   : unbiased inner-product estimator — better absolute accuracy
    Cost     : 2 matmuls  (mse + QJL correction)
    Requires : bits >= 3

Reference: Zandieh, Daliri, Hadian, Mirrokni (2025).
    "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."
    arXiv:2504.19874.  ICLR 2026.

File format
-----------
v1 (legacy)  — mse only, 20-byte header
v2           — adds flags field (bit-0 = use_prod) + QJL arrays
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from ._codebooks import get_codebook
from ._rotation import padded_dim, rht

_MAGIC = b"SNPV"
_VERSION = 2
_FLAG_PROD = 0x1


class SnapIndex:
    """Compressed in-memory ANN index using randomized Hadamard + Lloyd-Max.

    Parameters
    ----------
    dim : int
        Embedding dimension (e.g. 384 for BGE-small, 1536 for OpenAI ada-002).
    bits : int
        Bits per coordinate: 2, 3, or 4.
        In ``use_prod`` mode, (bits-1) go to MSE stage and 1 bit to QJL.
        ``bits >= 3`` is required when ``use_prod=True``.
    seed : int
        Rotation seed — must be the same at index build and query time.
    use_prod : bool
        Enable TurboQuant_prod unbiased inner-product estimator.
        Slower (~2x) but zero systematic bias.

    Examples
    --------
    >>> import numpy as np
    >>> from snapvec import SnapIndex
    >>> idx = SnapIndex(dim=384, bits=4)
    >>> vecs = np.random.randn(1000, 384).astype(np.float32)
    >>> idx.add_batch(list(range(1000)), vecs)
    >>> results = idx.search(vecs[0], k=5)
    >>> results[0]  # (id, score)  — top match is the vector itself
    (0, ...)
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        seed: int = 0,
        use_prod: bool = False,
    ) -> None:
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4; got {bits}")
        if use_prod and bits < 3:
            raise ValueError(
                f"use_prod=True requires bits >= 3 (got {bits}). "
                "TurboQuant_prod uses (bits-1) bits for MSE stage, minimum 2."
            )

        self.dim = dim
        self.bits = bits
        self.seed = seed
        self.use_prod = use_prod

        self._pdim: int = padded_dim(dim)
        self._mse_bits: int = bits - 1 if use_prod else bits
        self._centroids, _ = get_codebook(self._mse_bits)

        # Storage arrays (grown on add_batch, compacted on delete)
        self._ids: list[Any] = []
        self._id_to_pos: dict[Any, int] = {}       # O(1) id → row lookup
        self._indices = np.zeros((0, self._pdim), dtype=np.uint8)
        self._norms = np.zeros(0, dtype=np.float32)

        # TurboQuant_prod arrays (None in mse mode)
        self._qjl: np.ndarray | None = None        # (N, pdim) int8, ±1
        self._rnorms: np.ndarray | None = None     # (N,) float32, ‖r_rot‖
        self._S: np.ndarray | None = None          # (pdim, pdim) float32

        if use_prod:
            rng = np.random.default_rng(seed + 1337)
            self._S = rng.standard_normal((self._pdim, self._pdim)).astype(np.float32)

        # float16 centroid cache — invalidated on writes
        self._cache: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────── #
    # Core API                                                              #
    # ──────────────────────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        mode = "prod" if self.use_prod else "mse"
        return (
            f"SnapIndex(dim={self.dim}, bits={self.bits}, "
            f"mode={mode}, n={len(self)})"
        )

    def add(self, id: Any, vector: np.ndarray) -> None:
        """Add a single vector. Prefer :meth:`add_batch` for bulk inserts."""
        self.add_batch([id], np.asarray(vector, dtype=np.float32).reshape(1, -1))

    def add_batch(self, ids: list[Any], vectors: np.ndarray) -> None:
        """Add vectors in bulk — ~50x faster than repeated :meth:`add`.

        Parameters
        ----------
        ids : list[Any]
            Identifier for each vector (int, str, …).  Must be unique.
        vectors : np.ndarray, shape (n, dim), float32
            Raw embedding vectors (need not be normalized).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)
        if n == 0:
            return

        # 1. Normalize to unit sphere
        raw_norms = np.linalg.norm(vectors, axis=1)             # (n,)
        safe = np.where(raw_norms > 1e-10, raw_norms, 1.0)
        units = vectors / safe[:, None]                          # (n, dim)

        # 2. Zero-pad to power of 2, batch RHT — O(n·d·log d)
        pdim = self._pdim
        padded = np.zeros((n, pdim), dtype=np.float32)
        padded[:, :self.dim] = units
        rotated = rht(padded, self.seed)                         # (n, pdim)
        scaled = rotated * np.sqrt(pdim)                         # ≈ N(0,1)

        # 3. MSE quantization at _mse_bits
        _, boundaries = get_codebook(self._mse_bits)
        flat_idx = np.searchsorted(boundaries, scaled.ravel()).astype(np.uint8)
        batch_idx = flat_idx.reshape(n, pdim)
        batch_norms = np.where(raw_norms > 1e-10, raw_norms, 0.0).astype(np.float32)

        # 4. QJL residual (prod mode)
        qjl_signs: np.ndarray | None = None
        residual_norms: np.ndarray | None = None
        if self.use_prod:
            assert self._S is not None
            reconstructed = self._centroids[batch_idx]           # (n, pdim)
            r_scaled = (scaled - reconstructed).astype(np.float32)
            # r_rot = r_scaled / sqrt(pdim)  — work in unscaled space
            # sign(S·r_rot) = sign(S·r_scaled)  — scale-invariant
            S_r = (self._S @ r_scaled.T).T                      # (n, pdim)
            qjl_signs = np.sign(S_r).astype(np.int8)
            qjl_signs[qjl_signs == 0] = 1
            # Store ‖r_rot‖ = ‖r_scaled‖/sqrt(pdim)
            residual_norms = (
                np.linalg.norm(r_scaled, axis=1) / np.sqrt(pdim)
            ).astype(np.float32)

        # 5. Append to storage
        start = len(self._ids)
        self._ids.extend(ids)
        for i, id_val in enumerate(ids):
            self._id_to_pos[id_val] = start + i

        self._indices = (
            batch_idx if len(self._indices) == 0
            else np.vstack([self._indices, batch_idx])
        )
        self._norms = (
            batch_norms if len(self._norms) == 0
            else np.concatenate([self._norms, batch_norms])
        )

        if self.use_prod:
            self._qjl = (
                qjl_signs if self._qjl is None
                else np.vstack([self._qjl, qjl_signs])
            )
            self._rnorms = (
                residual_norms if self._rnorms is None
                else np.concatenate([self._rnorms, residual_norms])
            )

        self._cache = None  # invalidate float16 cache

    def delete(self, id: Any) -> bool:
        """Remove a vector by id.  O(1) lookup, O(n) position compaction.

        Returns True if the id was found and removed, False otherwise.
        """
        if id not in self._id_to_pos:
            return False

        pos = self._id_to_pos.pop(id)
        self._ids.pop(pos)
        self._indices = np.delete(self._indices, pos, axis=0)
        self._norms = np.delete(self._norms, pos)
        if self._qjl is not None:
            self._qjl = np.delete(self._qjl, pos, axis=0)
            self._rnorms = np.delete(self._rnorms, pos)

        # Compact position map
        for id_val, p in self._id_to_pos.items():
            if p > pos:
                self._id_to_pos[id_val] = p - 1

        self._cache = None
        return True

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[Any, float]]:
        """Find k nearest neighbors by approximate cosine similarity.

        Parameters
        ----------
        query : np.ndarray, shape (dim,)
            Query vector (raw, need not be normalized).
        k : int
            Number of results to return.

        Returns
        -------
        list of (id, score) sorted by descending similarity.
        Score is an approximation of cosine similarity ∈ [-1, 1].
        """
        if not self._ids:
            return []

        q = np.asarray(query, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []

        pdim = self._pdim
        q_unit = q / q_norm
        q_padded = np.zeros(pdim, dtype=np.float32)
        q_padded[:self.dim] = q_unit
        q_rot = rht(q_padded, self.seed)
        q_scaled = (q_rot * np.sqrt(pdim)).astype(np.float32)

        # Stage 1 — MSE score
        if self._cache is None:
            self._cache = self._centroids[self._indices].astype(np.float16)
        scores = (self._cache @ q_scaled).astype(np.float32) / pdim    # (N,)

        # Stage 2 — QJL correction (prod mode)
        # score_i += sqrt(π/2) / d · ‖r_i‖ · dot(S·q_rot, sign(S·r_i))
        if self.use_prod and self._qjl is not None:
            assert self._S is not None and self._rnorms is not None
            q_unit_rot = q_scaled / np.sqrt(pdim)
            S_q = self._S @ q_unit_rot                       # (pdim,)
            qjl_dots = self._qjl.astype(np.float32) @ S_q   # (N,)
            correction = (
                np.sqrt(np.pi / 2.0) / pdim
                * self._rnorms
                * qjl_dots
            )
            scores = scores + correction

        actual_k = min(k, len(self._ids))
        top_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        return [(self._ids[i], float(scores[i])) for i in top_idx]

    # ──────────────────────────────────────────────────────────────────── #
    # Persistence                                                           #
    # ──────────────────────────────────────────────────────────────────── #

    def save(self, path: str | Path) -> None:
        """Persist index to a binary ``.snpv`` file (atomic write).

        Writes to ``<path>.tmp`` first, then renames — safe against crashes.
        """
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")

        n = len(self._ids)
        flags = _FLAG_PROD if self.use_prod else 0
        packed = _pack(self._indices, self._mse_bits)

        with open(tmp, "wb") as f:
            # Header: magic(4) + version(4) + dim(4) + bits(4) + seed(4) + n(4) + flags(4)
            f.write(_MAGIC)
            f.write(struct.pack("<IIIIII", _VERSION, self.dim, self.bits, self.seed, n, flags))
            # MSE indices (bit-packed)
            f.write(struct.pack("<I", len(packed)))
            f.write(packed)
            # Per-vector norms
            f.write(self._norms.tobytes())
            # QJL data (prod mode)
            if self.use_prod and self._qjl is not None:
                f.write(self._qjl.tobytes())
                f.write(self._rnorms.tobytes())
            # IDs
            for id_val in self._ids:
                enc = str(id_val).encode("utf-8")
                f.write(struct.pack("<H", len(enc)))
                f.write(enc)

        tmp.replace(path)  # atomic on POSIX

    @classmethod
    def load(cls, path: str | Path) -> "SnapIndex":
        """Load index from a ``.snpv`` file.

        Supports v1 (mse-only legacy) and v2 (with prod/flags) formats.
        """
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"Invalid file: expected {_MAGIC!r} magic, got {magic!r}")

            (version,) = struct.unpack("<I", f.read(4))

            if version == 1:
                dim, bits, seed, n = struct.unpack("<IIII", f.read(16))
                use_prod = False
            elif version == 2:
                dim, bits, seed, n, flags = struct.unpack("<IIIII", f.read(20))
                use_prod = bool(flags & _FLAG_PROD)
            else:
                raise ValueError(f"Unsupported file version: {version}")

            idx = cls(dim=dim, bits=bits, seed=seed, use_prod=use_prod)
            pdim = idx._pdim
            mse_bits = idx._mse_bits

            (packed_len,) = struct.unpack("<I", f.read(4))
            packed_bytes = f.read(packed_len)
            norms_bytes = f.read(n * 4)

            idx._indices = _unpack(packed_bytes, n, pdim, mse_bits)
            idx._norms = np.frombuffer(norms_bytes, dtype=np.float32).copy()

            if use_prod and n > 0:
                qjl_bytes = f.read(n * pdim)
                rnorm_bytes = f.read(n * 4)
                idx._qjl = np.frombuffer(qjl_bytes, dtype=np.int8).reshape(n, pdim).copy()
                idx._rnorms = np.frombuffer(rnorm_bytes, dtype=np.float32).copy()

            for pos in range(n):
                (id_len,) = struct.unpack("<H", f.read(2))
                raw = f.read(id_len).decode("utf-8")
                try:
                    id_val: Any = int(raw)
                except ValueError:
                    try:
                        id_val = float(raw)
                    except ValueError:
                        id_val = raw
                idx._ids.append(id_val)
                idx._id_to_pos[id_val] = pos

        return idx

    # ──────────────────────────────────────────────────────────────────── #
    # Diagnostics                                                           #
    # ──────────────────────────────────────────────────────────────────── #

    def stats(self) -> dict[str, Any]:
        """Return memory / compression statistics."""
        n = len(self._ids)
        orig = n * self.dim * 4
        mse_b = self._indices.nbytes + self._norms.nbytes
        qjl_b = (
            (self._qjl.nbytes + self._rnorms.nbytes)
            if self._qjl is not None else 0
        )
        total = mse_b + qjl_b
        return {
            "n": n,
            "dim": self.dim,
            "padded_dim": self._pdim,
            "bits": self.bits,
            "mse_bits": self._mse_bits,
            "use_prod": self.use_prod,
            "float32_bytes": orig,
            "compressed_bytes": mse_b,
            "qjl_bytes": qjl_b,
            "total_bytes": total,
            "compression_ratio": round(orig / max(total, 1), 2),
        }


# ──────────────────────────────────────────────────────────────────────────── #
# Bit-packing helpers                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def _pack(mat: np.ndarray, bits: int) -> bytes:
    """Pack uint8 index matrix into a bit-packed byte string."""
    if bits == 8:
        return mat.tobytes()
    ipb = 8 // bits                    # indices per byte
    mask = (1 << bits) - 1
    flat = mat.ravel()
    pad = (-len(flat)) % ipb
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    out = np.zeros(len(flat) // ipb, dtype=np.uint8)
    for i in range(ipb):
        out |= (flat[i::ipb] & mask).astype(np.uint8) << (i * bits)
    return out.tobytes()


def _unpack(data: bytes, n_rows: int, n_cols: int, bits: int) -> np.ndarray:
    """Unpack bit-packed bytes back to uint8 matrix."""
    if bits == 8:
        return np.frombuffer(data, dtype=np.uint8).reshape(n_rows, n_cols).copy()
    ipb = 8 // bits
    mask = (1 << bits) - 1
    packed = np.frombuffer(data, dtype=np.uint8)
    total = n_rows * n_cols
    out = np.zeros(len(packed) * ipb, dtype=np.uint8)
    for i in range(ipb):
        out[i::ipb] = (packed >> (i * bits)) & mask
    return out[:total].reshape(n_rows, n_cols).copy()
