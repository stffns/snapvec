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
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._kmeans import assign_l2, kmeans_mse, probe_scores_l2_monotone
from ._rotation import padded_dim, rht

_MAGIC = b"SNPI"
_VERSION = 1
_MAX_ID_BYTES = 0xFFFF

# Standard FAISS rule of thumb for stable coarse k-means: at least
# 30 training vectors per cluster.  Below this, many clusters end up
# empty or with too few samples to learn a meaningful centroid, and
# recall@nprobe stops responding to nprobe (we measured this on the
# v0.5 N=1M baseline: 50k train / 4096 nlist → recall pinned at 0.731).
_FAISS_MIN_RATIO = 30

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
        recommended = self.nlist * _FAISS_MIN_RATIO
        if len(arr) < recommended:
            warnings.warn(
                f"only {len(arr)} training vectors for nlist={self.nlist} "
                f"(ratio {len(arr) / self.nlist:.0f}); FAISS rule of thumb "
                f"is ≥ {_FAISS_MIN_RATIO} samples per cluster (~{recommended} "
                f"total).  Below this many coarse clusters end up empty or "
                f"under-trained and recall stops responding to nprobe.  "
                f"Either pass more training data or lower nlist.",
                UserWarning,
                stacklevel=2,
            )
        pre, _ = self._preprocess(arr)
        # 1. Coarse k-means.
        self._coarse = kmeans_mse(
            pre, self.nlist, n_iters=kmeans_iters, seed=self.seed,
        )
        # 2. Shared residual codebooks (trained on pooled residuals).
        asn = assign_l2(pre, self._coarse)
        residuals = pre - self._coarse[asn]
        for j in range(self.M):
            Rj = residuals[:, j * self._d_sub : (j + 1) * self._d_sub].astype(np.float32)
            self._codebooks[j] = kmeans_mse(
                Rj, self.K, n_iters=kmeans_iters, seed=self.seed + 1000 + j,
            )
        self._fitted = True

    # ──────────────────────────────────────────────────────────────── #
    # build                                                             #
    # ──────────────────────────────────────────────────────────────── #

    def add(self, id: Any, vector: NDArray[np.float32]) -> None:
        self.add_batch([id], np.asarray(vector, dtype=np.float32)[None, :])

    # Encode vectors in chunks so peak memory stays bounded — at
    # N=1M, d=384, M=192, the residuals matrix alone is 1.5 GB and
    # would push a typical laptop into swap.  Tuned to keep each
    # chunk's intermediate buffers under ~150 MB.
    _ENCODE_CHUNK = 65_536

    def add_batch(
        self, ids: list[Any], vectors: NDArray[np.float32]
    ) -> None:
        """Append a batch.  Re-sorts the whole corpus by cluster id to
        preserve the contiguous layout — O(N) per call, so bulk-ingest
        before search is the intended pattern.

        Encoding is chunked so peak transient memory stays bounded
        regardless of batch size.  ``_id_to_row`` is rebuilt once at
        the end via numpy bulk operations rather than a Python dict
        comprehension, which avoids a O(N) interpreter pass at the
        end of large batches.
        """
        self._require_fitted()
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(
                f"vectors must be shape (n, {self.dim}); got {arr.shape}"
            )
        n = len(arr)
        if n == 0:
            return
        if len(ids) != n:
            raise ValueError(
                f"ids and vectors must have the same length; "
                f"got {len(ids)} ids and {n} vectors"
            )
        # Reject duplicates up front (within the batch + against the
        # already-indexed set).  Done with a single set construction
        # and one membership pass over _id_to_row, instead of an
        # interleaved per-id loop.
        new_set = set(ids)
        if len(new_set) != n:
            # Find the first duplicate for a useful error message.
            seen: set[Any] = set()
            for id_val in ids:
                if id_val in seen:
                    raise ValueError(f"duplicate id in batch: {id_val!r}")
                seen.add(id_val)
        clash = new_set & self._id_to_row.keys()
        if clash:
            raise ValueError(
                f"id already indexed: {next(iter(clash))!r}"
            )

        # Encode in chunks → bounded peak memory.
        new_codes = np.empty((n, self.M), dtype=np.uint8)
        new_asn = np.empty(n, dtype=np.int64)
        new_norms = np.empty(n if not self.normalized else 0, dtype=np.float32)
        cb_norms = (self._codebooks ** 2).sum(2)  # (M, K) precomputed
        cb_T = np.transpose(self._codebooks, (0, 2, 1))  # (M, d_sub, K)
        for start in range(0, n, self._ENCODE_CHUNK):
            end = min(start + self._ENCODE_CHUNK, n)
            sub = arr[start:end]
            pre, norms = self._preprocess(sub)
            asn_chunk = assign_l2(pre, self._coarse)
            new_asn[start:end] = asn_chunk
            residuals = pre - self._coarse[asn_chunk]
            for j in range(self.M):
                Rj = residuals[:, j * self._d_sub : (j + 1) * self._d_sub]
                # ‖R - c_j,k‖² = ‖R‖² − 2 R · c + ‖c‖²
                d2 = (
                    (Rj * Rj).sum(1, keepdims=True)
                    - 2 * Rj @ cb_T[j]
                    + cb_norms[j][None, :]
                )
                new_codes[start:end, j] = d2.argmin(1).astype(np.uint8)
            if not self.normalized:
                new_norms[start:end] = norms

        # Combine with existing state, sort once by cluster id.
        if len(self._codes) == 0:
            combined_codes = new_codes
            combined_asn = new_asn
            combined_ids_seq: list[Any] = list(ids)
            combined_norms = new_norms
        else:
            combined_codes = np.vstack([self._codes, new_codes])
            combined_asn = np.concatenate(
                [self._cluster_ids_from_offsets(), new_asn]
            )
            combined_ids_seq = self._ids_by_row + list(ids)
            combined_norms = (
                new_norms if self.normalized
                else np.concatenate([self._norms, new_norms])
            )

        order = np.argsort(combined_asn, kind="stable")
        self._codes = combined_codes[order]
        # Reorder ids via numpy object array → bulk gather in C, no
        # Python list comprehension over N elements.
        ids_arr = np.array(combined_ids_seq, dtype=object)
        self._ids_by_row = ids_arr[order].tolist()
        if not self.normalized:
            self._norms = combined_norms[order]
        counts = np.bincount(combined_asn, minlength=self.nlist)
        self._offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._id_to_row = dict(zip(self._ids_by_row, range(len(self._ids_by_row))))

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

        # Rank clusters by L2-monotone score (matches the metric used
        # during assignment — plain ⟨q, c⟩ would be wrong because
        # coarse centroids are means of unit vectors so ‖c‖ varies).
        # For scoring probed vectors we still use ⟨q, centroid_c⟩ as
        # the additive offset, since the decoded vector is
        # centroid_c + decoded_residual and we score ⟨q, decoded⟩.
        probe_ranking = probe_scores_l2_monotone(self._coarse, q_pre)
        coarse_dot = self._coarse @ q_pre          # (nlist,)
        probe = np.argpartition(-probe_ranking, nprobe - 1)[:nprobe]

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
        scores = np.empty(total, dtype=np.float32)
        cursor = 0
        for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
            n_c = e - s
            if n_c == 0:
                continue
            cat[cursor : cursor + n_c] = self._codes[s:e]
            row_idx[cursor : cursor + n_c] = np.arange(s, e, dtype=np.int64)
            scores[cursor : cursor + n_c] = coarse_dot[c]
            cursor += n_c

        for j in range(self.M):
            scores += lut[j][cat[:, j]]

        if not self.normalized:
            scores *= self._norms[row_idx]

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
            # Validate offsets: length, monotone non-decreasing,
            # boundary values.  Corrupted or truncated files would
            # otherwise index into wrong ranges of _codes at search
            # time and silently return garbage.
            if idx._offsets.shape != (nlist + 1,):
                raise ValueError(
                    f"offsets length {idx._offsets.shape} does not match "
                    f"nlist+1 = {nlist + 1}"
                )
            if int(idx._offsets[0]) != 0:
                raise ValueError(
                    f"offsets[0] must be 0; got {int(idx._offsets[0])}"
                )
            if int(idx._offsets[-1]) != n:
                raise ValueError(
                    f"offsets[-1] must equal n={n}; got "
                    f"{int(idx._offsets[-1])}"
                )
            if np.any(np.diff(idx._offsets) < 0):
                raise ValueError("offsets must be non-decreasing")
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
