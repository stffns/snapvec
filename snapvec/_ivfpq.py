"""IVF + residual Product Quantization index for snapvec.

Two-level retrieval structure on top of the ``PQSnapIndex`` machinery:

1. **Coarse k-means partition.**  ``nlist`` centroids over the corpus.
   Each vector is assigned to its nearest centroid on ``add``.
2. **Residual PQ.**  For each vector ``x`` assigned to cluster ``c``,
   the quantizer encodes the residual ``r = x − centroid_c`` with an
   ``M × K`` product codebook shared across clusters.

At query time:

- Find the top-``nprobe`` clusters by ``⟨q, centroid_c⟩``.
- Build the usual ``(M, K)`` ADC LUT against the residual codebooks.
- For each probed cluster, score its vectors as
  ``score = ⟨q, centroid_c⟩ + Σ_j LUT[j, codes[i, j]]``.
- Top-k merge across clusters.

Storage is cluster-contiguous: ``add_batch`` sorts the incoming codes
by cluster id and keeps an ``offsets`` array so probed-cluster access
is a single slice, not a boolean-mask rebuild per query.  This is what
makes the IVF Pareto actually pay off in wall-clock terms on top of
the ``PQSnapIndex`` baseline — see
``experiments/bench_ivf_pq_contiguous.py`` for the measurements.

Typical operating regime: ``N ≳ 10⁴`` (below that, ``PQSnapIndex``
full-scan already fits in microseconds and the IVF indirection is not
worth it).  Classic FAISS rule-of-thumb ``nlist ≈ 4·√N``.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._rotation import padded_dim, rht

_MAGIC = b"SNPI"
_VERSION = 1
_MAX_ID_BYTES = 0xFFFF

# Flags bitfield
_FLAG_NORMALIZED = 1 << 0
_FLAG_USE_RHT = 1 << 1


def _divisors(n: int, lo: int = 2, hi: int = 1024) -> list[int]:
    return [d for d in range(lo, min(n, hi) + 1) if n % d == 0]


def _decode_id(raw: str) -> Any:
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


# ──────────────────────────────────────────────────────────────────── #
# K-means primitives (private, mirror _pq.py to keep modules decoupled) #
# ──────────────────────────────────────────────────────────────────── #

def _kmeans_pp_init(
    X: NDArray[np.float32], K: int, rng: np.random.Generator,
) -> NDArray[np.float32]:
    n = X.shape[0]
    centers = [X[int(rng.integers(n))]]
    d2 = ((X - centers[0]) ** 2).sum(1)
    for _ in range(1, K):
        total = d2.sum()
        p = d2 / total if total > 1e-12 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=p))
        centers.append(X[nxt])
        d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
    return np.stack(centers).astype(np.float32)


def _kmeans_mse(
    X: NDArray[np.float32], K: int, n_iters: int, seed: int,
) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    C = _kmeans_pp_init(X, K, rng)
    x_sq = (X ** 2).sum(1, keepdims=True)
    for _ in range(n_iters):
        d2 = x_sq - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
        asn = d2.argmin(1)
        newC = np.empty_like(C)
        for k in range(K):
            m = asn == k
            if m.any():
                newC[k] = X[m].mean(0)
            else:
                newC[k] = X[d2.min(1).argmax()]
        if np.allclose(newC, C, atol=1e-5):
            return newC
        C = newC
    return C


def _assign(X: NDArray[np.float32], C: NDArray[np.float32]) -> NDArray[np.int64]:
    d2 = (X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    return d2.argmin(1)


# ──────────────────────────────────────────────────────────────────── #
# IVFPQSnapIndex                                                        #
# ──────────────────────────────────────────────────────────────────── #

class IVFPQSnapIndex:
    """Inverted-file + residual Product Quantization.

    Parameters
    ----------
    dim : int
        Embedding dimension.
    nlist : int
        Number of coarse clusters.  Typical values: ``4 · √N``.
        Must be ≥ 2 and ≤ (training sample size).
    M : int
        Number of PQ subspaces.  Must divide ``dim`` (or ``pdim`` when
        ``use_rht=True``).
    K : int, default 256
        Centroids per subspace.  ``2 ≤ K ≤ 256``.
    seed : int, default 0
    normalized : bool, default False
        When True, inputs are assumed unit-length and no per-vector
        norm is stored.
    use_rht : bool, default False
        Off by default — same rationale as ``PQSnapIndex``.
    """

    def __init__(
        self,
        dim: int,
        nlist: int,
        M: int,
        K: int = 256,
        seed: int = 0,
        normalized: bool = False,
        use_rht: bool = False,
    ) -> None:
        if not (2 <= K <= 256):
            raise ValueError(f"K must be in [2, 256]; got {K}")
        if nlist < 2:
            raise ValueError(f"nlist must be >= 2; got {nlist}")
        if M < 1:
            raise ValueError(f"M must be >= 1; got {M}")

        pdim = padded_dim(dim) if use_rht else dim
        if pdim % M != 0:
            label = "pdim" if use_rht else "dim"
            raise ValueError(
                f"M={M} must divide {label}={pdim}; "
                f"valid options include {_divisors(pdim)}"
            )

        self.dim = dim
        self.nlist = nlist
        self.M = M
        self.K = K
        self.seed = seed
        self.normalized = normalized
        self.use_rht = use_rht

        self._pdim = pdim
        self._d_sub = pdim // M
        self._fitted = False
        self._default_nprobe = max(1, nlist // 16)

        self._coarse: NDArray[np.float32] = np.zeros((nlist, pdim), dtype=np.float32)
        self._codebooks: NDArray[np.float32] = np.zeros(
            (M, K, self._d_sub), dtype=np.float32
        )
        # cluster-contiguous storage
        self._codes: NDArray[np.uint8] = np.zeros((0, M), dtype=np.uint8)
        self._offsets: NDArray[np.int64] = np.zeros(nlist + 1, dtype=np.int64)
        self._norms: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        # Internal row (cluster-sorted) → external id mapping.
        self._ids_by_row: list[Any] = []
        self._id_to_row: dict[Any, int] = {}

    # ──────────────────────────────────────────────────────────────── #
    # preprocessing                                                     #
    # ──────────────────────────────────────────────────────────────── #

    def _preprocess(
        self, X: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        arr = np.asarray(X, dtype=np.float32)
        if self.normalized:
            units = arr
            norms = np.empty(0, dtype=np.float32)
        else:
            raw = np.linalg.norm(arr, axis=1)
            safe = np.where(raw > 1e-10, raw, 1.0)
            units = arr / safe[:, None]
            norms = np.where(raw > 1e-10, raw, 0.0).astype(np.float32)
        if self.use_rht:
            padded = np.zeros((len(arr), self._pdim), dtype=np.float32)
            padded[:, : self.dim] = units
            rot = rht(padded, self.seed)
            rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-12
            return rot.astype(np.float32), norms
        return units.astype(np.float32), norms

    def _preprocess_single(self, q: NDArray[np.float32]) -> NDArray[np.float32]:
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

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before add/search")

    # ──────────────────────────────────────────────────────────────── #
    # training                                                          #
    # ──────────────────────────────────────────────────────────────── #

    def fit(
        self,
        training_vectors: NDArray[np.float32],
        kmeans_iters: int = 15,
    ) -> None:
        """Train coarse centroids and residual codebooks."""
        if self._fitted:
            raise RuntimeError(
                "fit() already called; create a fresh IVFPQSnapIndex to "
                "train on different data."
            )
        arr = np.asarray(training_vectors, dtype=np.float32)
        if len(arr) < max(self.K, self.nlist):
            raise ValueError(
                f"need at least max(K, nlist) = "
                f"{max(self.K, self.nlist)} training vectors; got {len(arr)}"
            )
        pre, _ = self._preprocess(arr)
        # 1. Coarse k-means.
        self._coarse = _kmeans_mse(
            pre, self.nlist, n_iters=kmeans_iters, seed=self.seed,
        )
        # 2. Shared residual codebooks (trained on pooled residuals).
        asn = _assign(pre, self._coarse)
        residuals = pre - self._coarse[asn]
        for j in range(self.M):
            Rj = residuals[:, j * self._d_sub : (j + 1) * self._d_sub].astype(np.float32)
            self._codebooks[j] = _kmeans_mse(
                Rj, self.K, n_iters=kmeans_iters, seed=self.seed + 1000 + j,
            )
        self._fitted = True

    # ──────────────────────────────────────────────────────────────── #
    # build                                                             #
    # ──────────────────────────────────────────────────────────────── #

    def add(self, id: Any, vector: NDArray[np.float32]) -> None:
        self.add_batch([id], np.asarray(vector, dtype=np.float32)[None, :])

    def add_batch(
        self, ids: list[Any], vectors: NDArray[np.float32]
    ) -> None:
        """Append a batch.  Re-sorts the whole corpus by cluster id to
        preserve the contiguous layout — O(N) per call, so bulk-ingest
        before search is the intended pattern.
        """
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
        asn_new = _assign(pre, self._coarse)
        residuals = pre - self._coarse[asn_new]
        new_codes = np.empty((len(arr), self.M), dtype=np.uint8)
        for j in range(self.M):
            Rj = residuals[:, j * self._d_sub : (j + 1) * self._d_sub].astype(np.float32)
            d2 = (
                (Rj ** 2).sum(1, keepdims=True)
                - 2 * Rj @ self._codebooks[j].T
                + (self._codebooks[j] ** 2).sum(1)[None, :]
            )
            new_codes[:, j] = d2.argmin(1).astype(np.uint8)

        # Merge with existing storage, then re-sort by cluster id.
        combined_codes = (
            new_codes if len(self._codes) == 0
            else np.vstack([self._codes, new_codes])
        )
        combined_ids = self._ids_by_row + list(ids)
        existing_asn = self._cluster_ids_from_offsets() if len(self._codes) > 0 else np.zeros(0, dtype=np.int64)
        combined_asn = np.concatenate([existing_asn, asn_new])
        if not self.normalized:
            combined_norms = (
                norms if len(self._norms) == 0
                else np.concatenate([self._norms, norms])
            )
        else:
            combined_norms = np.empty(0, dtype=np.float32)

        order = np.argsort(combined_asn, kind="stable")
        self._codes = combined_codes[order]
        self._ids_by_row = [combined_ids[i] for i in order]
        if not self.normalized:
            self._norms = combined_norms[order]
        counts = np.bincount(combined_asn, minlength=self.nlist)
        self._offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._id_to_row = {v: i for i, v in enumerate(self._ids_by_row)}

    def _cluster_ids_from_offsets(self) -> NDArray[np.int64]:
        """Reconstruct the per-row cluster id from ``_offsets``."""
        asn = np.empty(len(self._codes), dtype=np.int64)
        for c in range(self.nlist):
            s, e = int(self._offsets[c]), int(self._offsets[c + 1])
            if e > s:
                asn[s:e] = c
        return asn

    def delete(self, id: Any) -> bool:
        """Remove a vector by id.  O(n) — rebuilds the contiguous layout."""
        if id not in self._id_to_row:
            return False
        row = self._id_to_row[id]
        asn = self._cluster_ids_from_offsets()
        mask = np.ones(len(self._codes), dtype=bool)
        mask[row] = False
        self._codes = self._codes[mask]
        asn = asn[mask]
        self._ids_by_row = [self._ids_by_row[i] for i in range(len(mask)) if mask[i]]
        if not self.normalized:
            self._norms = self._norms[mask]
        counts = np.bincount(asn, minlength=self.nlist)
        self._offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._id_to_row = {v: i for i, v in enumerate(self._ids_by_row)}
        return True

    # ──────────────────────────────────────────────────────────────── #
    # search                                                            #
    # ──────────────────────────────────────────────────────────────── #

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        nprobe: int | None = None,
    ) -> list[tuple[Any, float]]:
        """Approximate top-k via IVF probing + residual PQ ADC.

        ``nprobe=None`` defaults to ``max(1, nlist // 16)``, which
        in the benchmarks lands near the knee of the recall/speedup
        curve (recall drop ≤ 0.02 vs. full scan, ~9× speedup).
        """
        self._require_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if nprobe is None:
            nprobe = self._default_nprobe
        if not (1 <= nprobe <= self.nlist):
            raise ValueError(
                f"nprobe must be in [1, nlist={self.nlist}]; got {nprobe}"
            )
        if len(self._ids_by_row) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        if float(np.linalg.norm(q)) < 1e-10:
            return []
        q_pre = self._preprocess_single(q)

        coarse_scores = self._coarse @ q_pre          # (nlist,)
        probe = np.argpartition(-coarse_scores, nprobe - 1)[:nprobe]

        # Per-subspace residual LUT.
        lut = np.empty((self.M, self.K), dtype=np.float32)
        for j in range(self.M):
            qj = q_pre[j * self._d_sub : (j + 1) * self._d_sub]
            lut[j] = self._codebooks[j] @ qj

        starts = self._offsets[probe]
        ends = self._offsets[probe + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            return []

        # Contiguous gather: one concatenation, one sum.
        cat = np.empty((total, self.M), dtype=np.uint8)
        row_idx = np.empty(total, dtype=np.int64)
        cursor = 0
        coarse_rep = np.empty(total, dtype=np.float32)
        for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
            n_c = e - s
            if n_c == 0:
                continue
            cat[cursor : cursor + n_c] = self._codes[s:e]
            row_idx[cursor : cursor + n_c] = np.arange(s, e, dtype=np.int64)
            coarse_rep[cursor : cursor + n_c] = coarse_scores[c]
            cursor += n_c

        scores = coarse_rep.copy()
        for j in range(self.M):
            scores += lut[j][cat[:, j]]

        if not self.normalized:
            scores = scores * self._norms[row_idx]

        k_eff = min(k, total)
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top])]
        return [
            (self._ids_by_row[int(row_idx[i])], float(scores[i]))
            for i in top
        ]

    # ──────────────────────────────────────────────────────────────── #
    # diagnostics                                                       #
    # ──────────────────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self._ids_by_row)

    def __repr__(self) -> str:
        return (
            f"IVFPQSnapIndex(dim={self.dim}, nlist={self.nlist}, "
            f"M={self.M}, K={self.K}, n={len(self._ids_by_row)})"
        )

    def stats(self) -> dict[str, Any]:
        # codes (M) + norm (4 if unnormalized).  Coarse id is implicit
        # in the cluster-contiguous layout — it costs nlist+1 int64s
        # globally (offsets), amortized to ~0 per vector.
        bytes_per_vec = self.M + (4 if not self.normalized else 0)
        sizes = np.diff(self._offsets) if len(self._offsets) > 1 else np.zeros(1)
        return {
            "n": len(self._ids_by_row),
            "dim": self.dim,
            "padded_dim": self._pdim,
            "nlist": self.nlist,
            "M": self.M,
            "K": self.K,
            "d_sub": self._d_sub,
            "use_rht": self.use_rht,
            "fitted": self._fitted,
            "bytes_per_vec": bytes_per_vec,
            "cluster_size_min": int(sizes.min()) if len(sizes) else 0,
            "cluster_size_median": int(np.median(sizes)) if len(sizes) else 0,
            "cluster_size_max": int(sizes.max()) if len(sizes) else 0,
            "default_nprobe": self._default_nprobe,
        }

    # ──────────────────────────────────────────────────────────────── #
    # persistence                                                       #
    # ──────────────────────────────────────────────────────────────── #

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        flags = 0
        if self.normalized:
            flags |= _FLAG_NORMALIZED
        if self.use_rht:
            flags |= _FLAG_USE_RHT
        n = len(self._ids_by_row)
        with open(tmp, "wb") as f:
            f.write(_MAGIC)
            f.write(
                struct.pack(
                    "<IIIIIIIIII",
                    _VERSION, self.dim, self._pdim, self.nlist,
                    self.M, self.K, self._d_sub, self.seed, n, flags,
                )
            )
            f.write(self._coarse.tobytes())
            f.write(self._codebooks.tobytes())
            f.write(self._offsets.tobytes())
            if n > 0:
                f.write(self._codes.tobytes())
                if not self.normalized:
                    f.write(self._norms.tobytes())
                for id_val in self._ids_by_row:
                    s = str(id_val).encode("utf-8")
                    if len(s) > _MAX_ID_BYTES:
                        raise ValueError(
                            f"id {id_val!r} encodes to {len(s)} UTF-8 bytes; "
                            f"the file format stores id length as uint16 "
                            f"(max {_MAX_ID_BYTES})"
                        )
                    f.write(struct.pack("<H", len(s)))
                    f.write(s)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> "IVFPQSnapIndex":
        path = Path(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"bad magic {magic!r}, expected {_MAGIC!r}")
            (version, dim, pdim, nlist, M, K, d_sub, seed, n, flags) = struct.unpack(
                "<IIIIIIIIII", f.read(40)
            )
            if version != _VERSION:
                raise ValueError(f"unsupported .snpi version {version}")
            normalized = bool(flags & _FLAG_NORMALIZED)
            use_rht = bool(flags & _FLAG_USE_RHT)
            idx = cls(
                dim=dim, nlist=nlist, M=M, K=K, seed=seed,
                normalized=normalized, use_rht=use_rht,
            )
            if pdim != idx._pdim or d_sub != idx._d_sub:
                raise ValueError(
                    f"on-disk (pdim={pdim}, d_sub={d_sub}) differs from "
                    f"computed (pdim={idx._pdim}, d_sub={idx._d_sub}); "
                    f"file may be corrupted or from a future version"
                )
            idx._coarse = (
                np.frombuffer(f.read(nlist * pdim * 4), dtype=np.float32)
                .reshape(nlist, pdim).copy()
            )
            idx._codebooks = (
                np.frombuffer(f.read(M * K * d_sub * 4), dtype=np.float32)
                .reshape(M, K, d_sub).copy()
            )
            idx._offsets = (
                np.frombuffer(f.read((nlist + 1) * 8), dtype=np.int64).copy()
            )
            idx._fitted = True
            if n > 0:
                idx._codes = (
                    np.frombuffer(f.read(n * M), dtype=np.uint8)
                    .reshape(n, M).copy()
                )
                if not normalized:
                    idx._norms = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
                idx._ids_by_row = []
                for _ in range(n):
                    (ln,) = struct.unpack("<H", f.read(2))
                    idx._ids_by_row.append(
                        _decode_id(f.read(ln).decode("utf-8"))
                    )
                idx._id_to_row = {
                    v: i for i, v in enumerate(idx._ids_by_row)
                }
        return idx
