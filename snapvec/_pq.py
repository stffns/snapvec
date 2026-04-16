"""Product Quantization index for snapvec.

Trades ``SnapIndex``'s training-free property for a one-off ``fit``
step on a sample of the corpus, and in return delivers substantially
higher recall at the same storage — on BGE-small/SciFact, ``K = 256``
matches ``SnapIndex(bits=4)`` recall at half the bytes per vector.

Pipeline:

1. (optional) Randomized Hadamard Transform — off by default.
   The `use_rht=True` path Gaussianizes coordinates in the classic
   TurboQuant sense; on real embeddings we found this actively hurts
   PQ because the rotation destroys the subspace structure ``k-means``
   was going to exploit.  See ``experiments/bench_pq_scaleup_validation``.
2. Unit-normalize each vector (skipped when ``normalized=True``).
3. Split the (optionally-rotated) vector into ``M`` subspaces of size
   ``d_sub = (pdim or dim) / M``.  Division must be exact.
4. Per subspace, train a codebook of ``K`` centroids with k-means++
   init followed by Lloyd iterations (MSE).
5. Store per-vector codes ``(M,) uint8`` plus the original norm.

Search uses asymmetric distance computation (ADC): build a per-query
lookup table ``(M, K)`` of ``⟨q_j, c_{j,k}⟩``, then the score for each
stored vector is the sum of ``LUT[j, codes[i, j]]`` over ``j``.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._file_format import save_with_checksum_atomic, verify_checksum
from ._freezable import FreezableIndex
from ._kmeans import kmeans_mse
from ._rotation import padded_dim, rht

_MAGIC = b"SNPQ"
_VERSION = 1
_MAX_ID_BYTES = 0xFFFF  # file format stores id length as uint16

# Flags bitfield
_FLAG_NORMALIZED = 1 << 0
_FLAG_USE_RHT = 1 << 1


def _divisors(n: int, lo: int = 2, hi: int = 1024) -> list[int]:
    """Factors of ``n`` in ``[lo, hi]`` — used for ValueError hints."""
    return [d for d in range(lo, min(n, hi) + 1) if n % d == 0]


def _decode_id(raw: str) -> Any:
    """Round-trip numeric-looking ids back to int/float, matching SnapIndex."""
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


class PQSnapIndex(FreezableIndex):
    """Product-quantization index trained once on a corpus sample.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    M : int
        Number of subspaces.  ``(pdim or dim)`` must be divisible by ``M``.
    K : int, default 256
        Centroids per subspace.  Must satisfy ``2 ≤ K ≤ 256``.
    seed : int, default 0
        Seed for the RHT (if used) and for k-means++ init.
    normalized : bool, default False
        When True, inputs are assumed unit-length and no per-vector
        norm is stored.
    use_rht : bool, default False
        When True, prepend the randomized Hadamard transform before
        splitting into subspaces.  Off by default — on modern
        embeddings it hurts PQ by destroying subspace structure.
    """

    def __init__(
        self,
        dim: int,
        M: int,
        K: int = 256,
        seed: int = 0,
        normalized: bool = False,
        use_rht: bool = False,
    ) -> None:
        if not (2 <= K <= 256):
            raise ValueError(f"K must be in [2, 256]; got {K}")
        if M < 1:
            raise ValueError(f"M must be >= 1; got {M}")

        pdim = padded_dim(dim) if use_rht else dim
        if pdim % M != 0:
            label = "pdim" if use_rht else "dim"
            divisors = _divisors(pdim)
            raise ValueError(
                f"M={M} must divide {label}={pdim}; "
                f"valid options include {divisors}"
            )

        self.dim = dim
        self.M = M
        self.K = K
        self.seed = seed
        self.normalized = normalized
        self.use_rht = use_rht

        self._pdim = pdim
        self._d_sub = pdim // M
        self._fitted: bool = False

        self._codebooks: NDArray[np.float32] = np.zeros(
            (M, K, self._d_sub), dtype=np.float32
        )
        self._ids: list[Any] = []
        self._id_to_pos: dict[Any, int] = {}
        self._codes: NDArray[np.uint8] = np.zeros((M, 0), dtype=np.uint8)
        self._norms: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────── #
    # internals                                                         #
    # ──────────────────────────────────────────────────────────────── #

    def _preprocess(
        self, X: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return (unit-normalized rotated vectors, per-vector raw norms).

        Norms here are norms of the *input* (not of the rotated
        vector), so that `normalized=False` scoring can multiply by
        the original length at search time — matching ``SnapIndex``.
        """
        arr = np.asarray(X, dtype=np.float32)
        if self.normalized:
            # Skip norm storage entirely — scoring does not use them.
            norms = np.empty(0, dtype=np.float32)
            units = arr
        else:
            raw = np.linalg.norm(arr, axis=1)
            safe = np.where(raw > 1e-10, raw, 1.0)
            units = arr / safe[:, None]
            norms = np.where(raw > 1e-10, raw, 0.0).astype(np.float32)

        if self.use_rht:
            padded = np.zeros((len(arr), self._pdim), dtype=np.float32)
            padded[:, : self.dim] = units
            rot = rht(padded, self.seed)
            # Re-normalize post-RHT so subspaces see unit-length input
            # (RHT preserves norm up to numerical error, but explicit
            # normalization keeps the ADC score interpretable).
            rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-12
            return rot.astype(np.float32), norms
        return units.astype(np.float32), norms

    def _preprocess_single(
        self, q: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Single-vector version of ``_preprocess``; returns unit-length."""
        q = np.asarray(q, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-10:
            return np.zeros(self._pdim, dtype=np.float32)
        q_unit = q / q_norm
        if not self.use_rht:
            return q_unit.astype(np.float32)
        padded = np.zeros(self._pdim, dtype=np.float32)
        padded[: self.dim] = q_unit
        rot = rht(padded[None, :], self.seed)[0]
        rot /= np.linalg.norm(rot) + 1e-12
        return rot.astype(np.float32)

    # ──────────────────────────────────────────────────────────────── #
    # training                                                          #
    # ──────────────────────────────────────────────────────────────── #

    def fit(
        self,
        training_vectors: NDArray[np.float32],
        kmeans_iters: int = 15,
    ) -> None:
        """Train per-subspace codebooks on ``training_vectors``.

        Must be called exactly once, before the first ``add`` /
        ``add_batch``.  Calling ``fit`` a second time (whether or not
        any vectors have been indexed) raises — double-fit would
        silently overwrite the codebooks and, if any vectors had been
        indexed, invalidate their codes.
        """
        self._check_not_frozen("fit")
        if self._fitted:
            # Covers both re-fit with or without prior add_batch — either
            # would overwrite the codebooks and (silently) invalidate any
            # codes already written.
            raise RuntimeError(
                "fit() already called; create a fresh PQSnapIndex to "
                "train on different data."
            )
        arr = np.asarray(training_vectors, dtype=np.float32)
        if len(arr) < self.K:
            raise ValueError(
                f"need at least K={self.K} training vectors; got {len(arr)}"
            )
        pre, _ = self._preprocess(arr)
        for j in range(self.M):
            Xj = pre[:, j * self._d_sub : (j + 1) * self._d_sub]
            self._codebooks[j] = kmeans_mse(
                Xj, self.K, n_iters=kmeans_iters, seed=self.seed + j,
            )
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before add/search")

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
        self._require_fitted()
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"vectors must be shape (n, {self.dim}); got {arr.shape}"
            )
        if len(arr) == 0:
            return
        if len(ids) != len(arr):
            raise ValueError(
                f"ids and vectors must have the same length; "
                f"got {len(ids)} ids and {len(arr)} vectors"
            )

        pre, norms = self._preprocess(arr)
        codes = np.empty((self.M, len(arr)), dtype=np.uint8)
        for j in range(self.M):
            Xj = pre[:, j * self._d_sub : (j + 1) * self._d_sub]
            d2 = (
                (Xj ** 2).sum(1, keepdims=True)
                - 2 * Xj @ self._codebooks[j].T
                + (self._codebooks[j] ** 2).sum(1)[None, :]
            )
            codes[j] = d2.argmin(1).astype(np.uint8)

        start = len(self._ids)
        self._ids.extend(ids)
        for i, id_val in enumerate(ids):
            self._id_to_pos[id_val] = start + i
        self._codes = (
            codes if self._codes.shape[1] == 0
            else np.hstack([self._codes, codes])
        )
        if not self.normalized:
            self._norms = (
                norms if len(self._norms) == 0
                else np.concatenate([self._norms, norms])
            )

    def delete(self, id: Any) -> bool:
        self._check_not_frozen("delete")
        if id not in self._id_to_pos:
            return False
        pos = self._id_to_pos.pop(id)
        self._ids.pop(pos)
        self._codes = np.delete(self._codes, pos, axis=1)
        if not self.normalized:
            self._norms = np.delete(self._norms, pos)
        for k, p in self._id_to_pos.items():
            if p > pos:
                self._id_to_pos[k] = p - 1
        return True

    # ──────────────────────────────────────────────────────────────── #
    # search                                                            #
    # ──────────────────────────────────────────────────────────────── #

    def search(
        self, query: NDArray[np.float32], k: int = 10,
    ) -> list[tuple[Any, float]]:
        self._require_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if not self._ids:
            return []
        q = np.asarray(query, dtype=np.float32)
        if float(np.linalg.norm(q)) < 1e-10:
            return []
        q_pre = self._preprocess_single(q)

        # LUT[j, k] = ⟨q_j, c_{j,k}⟩ -- single batched matmul.
        q_split = q_pre.reshape(self.M, self._d_sub, 1)   # (M, d_sub, 1)
        lut = np.matmul(self._codebooks, q_split)          # (M, K, 1)
        lut = lut.squeeze(-1)                              # (M, K)

        # Score[i] = Σ_j LUT[j, codes[j, i]]
        scores = np.zeros(self._codes.shape[1], dtype=np.float32)
        for j in range(self.M):
            scores += lut[j][self._codes[j]]

        if not self.normalized:
            scores = scores * self._norms

        k_eff = min(k, len(scores))
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[int(i)], float(scores[i])) for i in top]

    # ──────────────────────────────────────────────────────────────── #
    # diagnostics                                                       #
    # ──────────────────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"PQSnapIndex(dim={self.dim}, M={self.M}, K={self.K}, "
            f"use_rht={self.use_rht}, n={len(self._ids)})"
        )

    def stats(self) -> dict[str, Any]:
        bytes_per_vec = self.M + (4 if not self.normalized else 0)
        ram_bytes = (
            self._codebooks.nbytes
            + self._codes.nbytes
            + self._norms.nbytes
        )
        return {
            "n": len(self._ids),
            "dim": self.dim,
            "padded_dim": self._pdim,
            "M": self.M,
            "K": self.K,
            "d_sub": self._d_sub,
            "use_rht": self.use_rht,
            "fitted": self._fitted,
            "bytes_per_vec": bytes_per_vec,
            "compression_ratio": (self.dim * 4) / bytes_per_vec
            if bytes_per_vec > 0 else 0.0,
            "ram_bytes": int(ram_bytes),
        }

    # ──────────────────────────────────────────────────────────────── #
    # persistence                                                       #
    # ──────────────────────────────────────────────────────────────── #

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        flags = 0
        if self.normalized:
            flags |= _FLAG_NORMALIZED
        if self.use_rht:
            flags |= _FLAG_USE_RHT
        n = len(self._ids)

        def _write(f):
            f.write(_MAGIC)
            f.write(
                struct.pack(
                    "<IIIIIIIII",
                    _VERSION, self.dim, self.M, self.K, self._d_sub,
                    self.seed, n, flags, self._pdim,
                )
            )
            f.write(self._codebooks.tobytes())
            if n > 0:
                f.write(np.ascontiguousarray(self._codes.T).tobytes())
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
    def load(cls, path: str | Path) -> "PQSnapIndex":
        path = Path(path)
        verify_checksum(path)  # no-op for legacy files without a trailer
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"bad magic {magic!r}, expected {_MAGIC!r}")
            version, dim, M, K, d_sub, seed, n, flags, pdim = struct.unpack(
                "<IIIIIIIII", f.read(36)
            )
            if version != _VERSION:
                raise ValueError(f"unsupported .snpq version {version}")
            normalized = bool(flags & _FLAG_NORMALIZED)
            use_rht = bool(flags & _FLAG_USE_RHT)
            idx = cls(
                dim=dim, M=M, K=K, seed=seed,
                normalized=normalized, use_rht=use_rht,
            )
            if pdim != idx._pdim or d_sub != idx._d_sub:
                raise ValueError(
                    f"on-disk pdim={pdim}, d_sub={d_sub} differ from computed "
                    f"pdim={idx._pdim}, d_sub={idx._d_sub} for "
                    f"(dim={dim}, M={M}, use_rht={use_rht}); file may be "
                    f"corrupted or from a future version"
                )
            idx._codebooks = (
                np.frombuffer(f.read(M * K * d_sub * 4), dtype=np.float32)
                .reshape(M, K, d_sub)
                .copy()
            )
            idx._fitted = True
            if n > 0:
                idx._codes = (
                    np.frombuffer(f.read(n * M), dtype=np.uint8)
                    .reshape(n, M)
                    .T.copy()
                )
                if not normalized:
                    idx._norms = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
                idx._ids = []
                for _ in range(n):
                    (ln,) = struct.unpack("<H", f.read(2))
                    idx._ids.append(_decode_id(f.read(ln).decode("utf-8")))
                idx._id_to_pos = {v: i for i, v in enumerate(idx._ids)}
        return idx
