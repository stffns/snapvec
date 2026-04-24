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

import struct
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

try:
    from ._fast import fused_gather_adc
except ImportError:
    from ._fast_fallback import fused_gather_adc
from ._file_format import ChecksumWriter, save_with_checksum_atomic, verify_checksum
from ._freezable import FreezableIndex
from ._kmeans import (
    assign_l2,
    fit_opq_rotation,
    kmeans_mse,
    probe_scores_l2_monotone,
)
from ._rotation import padded_dim, rht

_MAGIC = b"SNPI"
_VERSION = 5  # v5: adds optional OPQ rotation block (present iff _FLAG_USE_OPQ)
_LEGACY_VERSIONS = {1, 2, 3, 4}
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
_FLAG_KEEP_FULL_PRECISION = 1 << 2
_FLAG_USE_OPQ = 1 << 3


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

class IVFPQSnapIndex(FreezableIndex):
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
    use_opq : bool, default False
        When True, learn an orthogonal OPQ-P rotation (Ge et al.,
        2013) during ``fit()`` and apply it to both corpus and
        queries before the coarse k-means and the residual PQ.
        Balances per-subspace variance, typically +0.5-2 pp recall
        at the same bytes/vec.  Mutually exclusive with ``use_rht``.
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
        keep_full_precision: bool = False,
        use_opq: bool = False,
    ) -> None:
        if not (2 <= K <= 256):
            raise ValueError(f"K must be in [2, 256]; got {K}")
        if nlist < 2:
            raise ValueError(f"nlist must be >= 2; got {nlist}")
        if M < 1:
            raise ValueError(f"M must be >= 1; got {M}")
        if use_opq and use_rht:
            raise ValueError(
                "use_opq and use_rht are mutually exclusive: OPQ learns "
                "a data-specific rotation during fit(), RHT applies a "
                "fixed random one.  Pick one."
            )

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
        self.keep_full_precision = keep_full_precision
        self.use_opq = use_opq
        # (pdim, pdim) OPQ rotation, set in fit() when use_opq=True.
        self._opq_rotation: NDArray[np.float32] | None = None

        self._pdim = pdim
        self._d_sub = pdim // M
        self._fitted = False
        self._default_nprobe = max(1, nlist // 16)

        self._coarse: NDArray[np.float32] = np.zeros((nlist, pdim), dtype=np.float32)
        self._codebooks: NDArray[np.float32] = np.zeros(
            (M, K, self._d_sub), dtype=np.float32
        )
        # cluster-contiguous storage
        # codes laid out column-major (M, n) — each subspace's codes
        # are contiguous so gather inside search is one cache-line walk
        # instead of a stride-M = 192 hop per lookup.
        self._codes: NDArray[np.uint8] = np.zeros((M, 0), dtype=np.uint8)
        self._offsets: NDArray[np.int64] = np.zeros(nlist + 1, dtype=np.int64)
        self._norms: NDArray[np.float32] = np.zeros(0, dtype=np.float32)
        # Optional float16 cache of the original (post-preprocess) vectors
        # for the `rerank_candidates` search path.  Stored cluster-
        # contiguously and kept in sync with _codes by add_batch / delete.
        # Layout (n, dim_eff) where dim_eff is pdim if use_rht else dim.
        # float16 halves both disk and RAM footprint of
        # the rerank cache at a negligible recall cost (~0.001 on
        # FIQA at rerank_candidates=100).  The rerank matmul
        # ``cand_full @ q_pre`` runs in float32 because ``q_pre`` is
        # float32 and NumPy's type-promotion rules for mixed-dtype
        # matmul yield the wider of the two.  If ``q_pre`` were ever
        # downgraded to fp16 the matmul would follow suit and
        # arithmetic precision would drop — keep it float32.
        self._full_precision: NDArray[np.float16] = np.zeros(
            (0, pdim), dtype=np.float16,
        )

        # Internal row (cluster-sorted) → external id mapping.
        self._ids_by_row: list[Any] = []
        self._id_to_row: dict[Any, int] = {}

        # Lazy thread pool, created on first ``search_batch`` call with
        # ``num_threads > 1``.  Reused across queries.  ``_executor_workers``
        # tracks the configured worker count without reaching into
        # ThreadPoolExecutor's private ``_max_workers`` attribute.
        self._executor: ThreadPoolExecutor | None = None
        self._executor_workers: int = 0
        # Serialises lazy executor creation in ``search_batch`` so two
        # threads calling search_batch(num_threads > 1) concurrently
        # don't race on the ``if self._executor is None`` check.
        self._executor_lock = threading.Lock()

    def close(self) -> None:
        """Release the lazy thread pool, if one was created.

        Safe to call multiple times.  Useful when an index is being
        torn down explicitly (e.g., long-lived workers cycling
        indices) — Python's GC will also reclaim the executor when
        the index goes out of scope, but explicit cleanup avoids
        worker threads lingering past their last useful query.
        """
        self._check_not_frozen("close")
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            self._executor_workers = 0

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
            # Optimized: ~4x faster than np.linalg.norm(..., axis=1) via einsum
            raw = np.sqrt(np.einsum('ij,ij->i', arr, arr))
            safe = cast(
                "NDArray[np.float32]",
                np.where(raw > 1e-10, raw, np.float32(1.0)),
            )
            units = cast("NDArray[np.float32]", arr / safe[:, None])
            norms = cast(
                "NDArray[np.float32]",
                np.where(raw > 1e-10, raw, np.float32(0.0)),
            )
        if self.use_rht:
            padded = np.zeros((len(arr), self._pdim), dtype=np.float32)
            padded[:, : self.dim] = units
            rot = rht(padded, self.seed)
            # Optimized: ~4x faster than np.linalg.norm(..., axis=1) via einsum.
            # ``1e-12`` is a Python float (float64 under pre-NEP-50 numpy);
            # wrap as np.float32 to keep ``rot`` in float32 without needing
            # a downcasting astype at the return.
            rot /= (
                np.sqrt(np.einsum('ij,ij->i', rot, rot))[:, np.newaxis]
                + np.float32(1e-12)
            )
            return rot, norms
        if self.use_opq and self._opq_rotation is not None:
            # Rotation is orthogonal, so unit-norm inputs stay unit-norm.
            return cast(
                "NDArray[np.float32]", units @ self._opq_rotation
            ), norms
        return units, norms

    def _preprocess_single(self, q: NDArray[np.float32]) -> NDArray[np.float32]:
        q = np.asarray(q, dtype=np.float32)
        # Optimized: ~1.5x speedup bypassing np.linalg.norm overhead
        q_norm = float(np.sqrt(np.inner(q, q)))
        if q_norm < 1e-10:
            return np.zeros(self._pdim, dtype=np.float32)
        # ``q_norm`` is a Python float (float64 under pre-NEP-50 numpy);
        # wrap in np.float32 so ``q / q_norm`` stays in float32.  Same
        # treatment for the ``1e-12`` epsilon inside the RHT branch.
        q_unit = q / np.float32(q_norm)
        if self.use_rht:
            padded = np.zeros(self._pdim, dtype=np.float32)
            padded[: self.dim] = q_unit
            rot = rht(padded[None, :], self.seed)[0]
            # Optimized: ~1.5x speedup bypassing np.linalg.norm overhead
            rot /= np.float32(np.sqrt(np.inner(rot, rot))) + np.float32(1e-12)
            return cast("NDArray[np.float32]", rot)
        if self.use_opq and self._opq_rotation is not None:
            return cast(
                "NDArray[np.float32]", q_unit @ self._opq_rotation
            )
        return cast("NDArray[np.float32]", q_unit)

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
        self._check_not_frozen("fit")
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
                f"(ratio {len(arr) / self.nlist:.1f}); FAISS rule of thumb "
                f"is ≥ {_FAISS_MIN_RATIO} samples per cluster (~{recommended} "
                f"total).  Below this many coarse clusters end up empty or "
                f"under-trained and recall stops responding to nprobe.  "
                f"Either pass more training data or lower nlist.",
                UserWarning,
                stacklevel=2,
            )
        if self.use_opq:
            # Fit rotation on unit-normalised inputs.  _preprocess
            # returns unit vectors and skips the rotation step because
            # _opq_rotation is still None at this point, so we can
            # reuse it instead of duplicating the norm code here.
            X_unit, _ = self._preprocess(arr)
            self._opq_rotation = fit_opq_rotation(X_unit, self.M)
        pre, _ = self._preprocess(arr)
        # 1. Coarse k-means.
        self._coarse = kmeans_mse(
            pre, self.nlist, n_iters=kmeans_iters, seed=self.seed,
        )
        # 2. Shared residual codebooks (trained on pooled residuals).
        asn = assign_l2(pre, self._coarse)
        residuals = pre - self._coarse[asn]
        for j in range(self.M):
            # ``residuals[:, slice]`` is a non-contiguous view with stride
            # = d * 4 bytes.  k-means runs matmul on it, and BLAS is much
            # faster on contiguous input -- so make the per-subspace slice
            # contiguous here.  (The earlier ``.astype(np.float32)`` did
            # this implicitly; calling it out explicitly makes the intent
            # legible without the "is this dtype guard or a copy?" confusion.)
            Rj = np.ascontiguousarray(
                residuals[:, j * self._d_sub : (j + 1) * self._d_sub]
            )
            self._codebooks[j] = kmeans_mse(
                Rj, self.K, n_iters=kmeans_iters, seed=self.seed + 1000 + j,
            )
        self._fitted = True

    # ──────────────────────────────────────────────────────────────── #
    # build                                                             #
    # ──────────────────────────────────────────────────────────────── #

    def add(self, id: Any, vector: NDArray[np.float32]) -> None:
        self._check_not_frozen("add")
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
        self._check_not_frozen("add_batch")
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

        # Encode in chunks → bounded peak memory.  Codes built
        # column-major (M, n) so the per-subspace gather inside
        # search() is contiguous instead of strided by M.
        new_codes = np.empty((self.M, n), dtype=np.uint8)
        new_asn = np.empty(n, dtype=np.int64)
        new_norms = np.empty(n if not self.normalized else 0, dtype=np.float32)
        new_full = (
            np.empty((n, self._pdim), dtype=np.float16)
            if self.keep_full_precision else
            np.empty((0, self._pdim), dtype=np.float16)
        )
        cb_norms = (self._codebooks ** 2).sum(2)             # (M, K)
        cb_T = np.transpose(self._codebooks, (0, 2, 1))      # (M, d_sub, K)
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
                new_codes[j, start:end] = d2.argmin(1).astype(np.uint8)
            if not self.normalized:
                new_norms[start:end] = norms
            if self.keep_full_precision:
                # Store the post-preprocess (RHT-rotated if use_rht,
                # unit-normed otherwise) vectors — these are the ones
                # the search-time dot product is consistent with.
                new_full[start:end] = pre

        # Merge new into the existing cluster-contiguous layout with a
        # single allocation.  For each cluster c, copy the existing
        # (contiguous) slice of old rows, followed by the new rows
        # landing in that cluster.  Both copies are contiguous memcpy,
        # which is materially faster on modern CPUs than the scatter
        # via fancy indexing we used in the first iteration of this
        # optimisation (prefetcher, SIMD, single-stream bandwidth).
        N_old = self._codes.shape[1]

        if N_old == 0:
            # Empty index: sort new once and write it in.  No merge
            # bookkeeping needed.
            new_order = np.argsort(new_asn, kind="stable")
            self._codes = new_codes[:, new_order]
            ids_arr = np.array(list(ids), dtype=object)
            self._ids_by_row = ids_arr[new_order].tolist()
            if not self.normalized:
                self._norms = new_norms[new_order]
            if self.keep_full_precision:
                self._full_precision = new_full[new_order]
            counts = np.bincount(new_asn, minlength=self.nlist)
        else:
            # Sort the new batch by cluster id so each cluster's new
            # items are a contiguous run; that lets the per-cluster
            # copy below use a single slice per side.
            new_order = np.argsort(new_asn, kind="stable")
            new_codes_sorted = new_codes[:, new_order]
            new_norms_sorted = (
                new_norms[new_order] if not self.normalized else new_norms
            )
            new_full_sorted = (
                new_full[new_order] if self.keep_full_precision else new_full
            )
            ids_sorted = np.array(list(ids), dtype=object)[new_order]

            new_counts = np.bincount(new_asn, minlength=self.nlist)
            new_batch_offsets = np.concatenate(
                [[0], np.cumsum(new_counts)]
            ).astype(np.int64)
            old_counts = np.diff(self._offsets).astype(np.int64)
            counts = old_counts + new_counts

            N_total = N_old + n
            combined_codes = np.empty((self.M, N_total), dtype=np.uint8)
            combined_ids = np.empty(N_total, dtype=object)
            combined_norms = (
                np.empty(N_total, dtype=np.float32)
                if not self.normalized else None
            )
            combined_full = (
                np.empty((N_total, self._pdim), dtype=np.float16)
                if self.keep_full_precision else None
            )

            old_ids_arr = np.array(self._ids_by_row, dtype=object)
            write_pos = 0
            for c in range(self.nlist):
                old_s = int(self._offsets[c])
                old_n = int(old_counts[c])
                new_s = int(new_batch_offsets[c])
                new_n = int(new_counts[c])
                if old_n:
                    combined_codes[:, write_pos : write_pos + old_n] = (
                        self._codes[:, old_s : old_s + old_n]
                    )
                    combined_ids[write_pos : write_pos + old_n] = (
                        old_ids_arr[old_s : old_s + old_n]
                    )
                    if combined_norms is not None:
                        combined_norms[write_pos : write_pos + old_n] = (
                            self._norms[old_s : old_s + old_n]
                        )
                    if combined_full is not None:
                        combined_full[write_pos : write_pos + old_n] = (
                            self._full_precision[old_s : old_s + old_n]
                        )
                    write_pos += old_n
                if new_n:
                    combined_codes[:, write_pos : write_pos + new_n] = (
                        new_codes_sorted[:, new_s : new_s + new_n]
                    )
                    combined_ids[write_pos : write_pos + new_n] = (
                        ids_sorted[new_s : new_s + new_n]
                    )
                    if combined_norms is not None:
                        combined_norms[write_pos : write_pos + new_n] = (
                            new_norms_sorted[new_s : new_s + new_n]
                        )
                    if combined_full is not None:
                        combined_full[write_pos : write_pos + new_n] = (
                            new_full_sorted[new_s : new_s + new_n]
                        )
                    write_pos += new_n

            self._codes = combined_codes
            self._ids_by_row = combined_ids.tolist()
            if combined_norms is not None:
                self._norms = combined_norms
            if combined_full is not None:
                self._full_precision = combined_full

        self._offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._id_to_row = dict(zip(self._ids_by_row, range(len(self._ids_by_row))))

    def _cluster_ids_from_offsets(self) -> NDArray[np.int64]:
        """Reconstruct the per-row cluster id from ``_offsets``."""
        asn = np.empty(self._codes.shape[1], dtype=np.int64)
        for c in range(self.nlist):
            s, e = int(self._offsets[c]), int(self._offsets[c + 1])
            if e > s:
                asn[s:e] = c
        return asn

    def delete(self, id: Any) -> bool:
        """Remove a vector by id.  O(n) — rebuilds the contiguous layout."""
        self._check_not_frozen("delete")
        if id not in self._id_to_row:
            return False
        row = self._id_to_row[id]
        asn = self._cluster_ids_from_offsets()
        mask = np.ones(self._codes.shape[1], dtype=bool)
        mask[row] = False
        self._codes = self._codes[:, mask]
        asn = asn[mask]
        self._ids_by_row = [self._ids_by_row[i] for i in range(len(mask)) if mask[i]]
        if not self.normalized:
            self._norms = self._norms[mask]
        if self.keep_full_precision:
            self._full_precision = self._full_precision[mask]
        counts = np.bincount(asn, minlength=self.nlist)
        self._offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self._id_to_row = {v: i for i, v in enumerate(self._ids_by_row)}
        return True

    # ──────────────────────────────────────────────────────────────── #
    # search                                                            #
    # ──────────────────────────────────────────────────────────────── #

    def _resolve_filter(
        self, filter_ids: set[Any]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
        """Translate an external-id filter set into ``(filter_rows, allowed_clusters)``.

        ``filter_rows`` is a sorted int64 array of internal row ids; the
        gather path uses ``np.isin(row_idx, filter_rows, assume_unique=True)``
        to drop non-filter candidates.  Keeping it sparse (vs a dense
        N-bool mask) avoids an O(N) allocation per query, which matters
        at scale: at N=100M a dense bool is 100 MB per call.
        ``allowed_clusters`` is the set of coarse clusters that contain
        at least one filter row; used to restrict the probe ranking so
        no ``nprobe`` slot is wasted on a cluster that cannot contribute
        a hit.

        Returns ``None`` when the filter yields zero ids present in the
        index (either because the caller passed an empty set or because
        every id is unknown); the caller then short-circuits to ``[]``.
        """
        filter_rows = np.fromiter(
            (self._id_to_row[i] for i in filter_ids if i in self._id_to_row),
            dtype=np.int64,
        )
        if filter_rows.size == 0:
            return None
        filter_rows.sort()
        # Row r belongs to the smallest cluster c for which offsets[c+1] > r,
        # which is exactly searchsorted(offsets[1:], r, side='right').
        allowed_clusters = np.unique(
            np.searchsorted(self._offsets[1:], filter_rows, side="right")
        ).astype(np.int64)
        return filter_rows, allowed_clusters

    @staticmethod
    def _probe_topn(
        ranking: NDArray[np.float32],
        nprobe: int,
        allowed: NDArray[np.int64] | None = None,
    ) -> NDArray[np.int64]:
        """Top-``nprobe`` clusters by ``ranking``, optionally restricted to
        ``allowed``.  When ``allowed`` is shorter than ``nprobe`` we return
        all of it (every contributing cluster gets probed)."""
        if allowed is None:
            return np.argpartition(-ranking, nprobe - 1)[:nprobe].astype(np.int64)
        if len(allowed) <= nprobe:
            return allowed
        restricted = ranking[allowed]
        top = np.argpartition(-restricted, nprobe - 1)[:nprobe]
        return allowed[top].astype(np.int64)

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        nprobe: int | None = None,
        rerank_candidates: int | None = None,
        filter_ids: set[Any] | None = None,
    ) -> list[tuple[Any, float]]:
        """Approximate top-k via IVF probing + residual PQ ADC.

        Parameters
        ----------
        query : NDArray[np.float32]
        k : int, default 10
        nprobe : int | None, default ``max(1, nlist // 16)``
            Number of coarse clusters to visit.
        rerank_candidates : int | None, default None
            When set, the IVF-PQ pass returns the top-``rerank_candidates``
            instead of top-``k``, then those are re-scored against the
            stored full-precision vectors (kept as ``float16`` since
            v0.7 to halve disk + RAM footprint; the rerank matmul
            itself still runs in ``float32`` because ``q_pre`` is
            ``float32`` and NumPy type-promotion widens the result),
            and the top-``k`` of the reranked set is returned.  Lifts
            recall toward the float32 brute-force ceiling at the cost
            of one
            ``(rerank_candidates, dim_eff) @ (dim_eff,)`` matmul per
            query — typically <1 ms even at large nprobe.

            Requires ``keep_full_precision=True`` at index construction.
            Must be ``>= k``.
        filter_ids : set | None, default None
            When provided, restrict results to ids in the set.  The
            implementation is cluster- and pool-aware: the probe ranking
            is restricted to clusters that contain at least one filter
            row (so sparse filters skip clusters entirely), and the
            row-level mask is applied before the top-k / rerank-pool
            selection (so the rerank candidate pool is drawn from the
            filtered subset, not from the unfiltered probe output).

            Unknown ids in ``filter_ids`` are silently dropped.  An
            entirely-unknown filter returns ``[]``.  A very sparse
            filter may need a larger ``nprobe`` to surface ``k`` hits.
        """
        self._require_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if rerank_candidates is not None:
            if not self.keep_full_precision:
                raise ValueError(
                    "rerank_candidates requires keep_full_precision=True "
                    "at IVFPQSnapIndex construction time."
                )
            if rerank_candidates < k:
                raise ValueError(
                    f"rerank_candidates ({rerank_candidates}) must be >= k ({k})"
                )
        if nprobe is None:
            nprobe = self._default_nprobe
        if not (1 <= nprobe <= self.nlist):
            raise ValueError(
                f"nprobe must be in [1, nlist={self.nlist}]; got {nprobe}"
            )
        if len(self._ids_by_row) == 0:
            return []
        q = np.asarray(query, dtype=np.float32)
        # Optimized: ~1.5x speedup bypassing np.linalg.norm overhead
        if float(np.sqrt(np.inner(q, q))) < 1e-10:
            return []

        filter_rows: NDArray[np.int64] | None = None
        allowed_clusters: NDArray[np.int64] | None = None
        if filter_ids is not None:
            resolved = self._resolve_filter(filter_ids)
            if resolved is None:
                return []
            filter_rows, allowed_clusters = resolved

        q_pre = self._preprocess_single(q)

        # Rank clusters by L2-monotone score (matches the metric used
        # during assignment — plain ⟨q, c⟩ would be wrong because
        # coarse centroids are means of unit vectors so ‖c‖ varies).
        # For scoring probed vectors we still use ⟨q, centroid_c⟩ as
        # the additive offset, since the decoded vector is
        # centroid_c + decoded_residual and we score ⟨q, decoded⟩.
        probe_ranking = probe_scores_l2_monotone(self._coarse, q_pre)
        coarse_dot = self._coarse @ q_pre          # (nlist,)
        probe = self._probe_topn(probe_ranking, nprobe, allowed_clusters)
        if len(probe) == 0:
            return []

        # Per-subspace residual LUT -- single batched matmul.
        q_split = q_pre.reshape(self.M, self._d_sub, 1)   # (M, d_sub, 1)
        lut = np.matmul(self._codebooks, q_split)          # (M, K, 1)
        lut = lut.squeeze(-1)                              # (M, K)

        if rerank_candidates is None:
            return self._score_one(probe, coarse_dot, lut, k, filter_rows)
        return self._score_one_with_rerank(
            probe, coarse_dot, lut, q_pre, k, rerank_candidates, filter_rows,
        )

    def _score_one_with_rerank(
        self,
        probe: NDArray[np.int64],
        coarse_dot: NDArray[np.float32],
        lut: NDArray[np.float32],
        q_pre: NDArray[np.float32],
        k: int,
        rerank_candidates: int,
        filter_rows: NDArray[np.int64] | None = None,
        _parallel: bool = True,
    ) -> list[tuple[Any, float]]:
        """IVF-PQ → top-N candidates → float32 rerank → top-k.

        Uses the same ``_gather_pq_scores`` helper as ``_score_one`` so
        the candidate-selection metric is identical to the non-rerank
        path (crucially including the ``* self._norms`` scaling when
        ``normalized=False`` — otherwise large-norm vectors could miss
        the candidate pool entirely).  The only difference from the
        default path is that we then look up ``_full_precision`` for
        the winners and re-score them exactly.
        """
        pq_scores, row_idx = self._gather_pq_scores(
            probe, coarse_dot, lut, _parallel=_parallel,
        )
        if filter_rows is not None and len(row_idx) > 0:
            # Both arrays are unique (row_idx is a concat of contiguous
            # per-cluster arange() blocks, filter_rows was made unique
            # in _resolve_filter), so assume_unique=True enables the
            # faster sort-merge path in np.isin.
            keep = np.isin(row_idx, filter_rows, assume_unique=True)
            pq_scores = pq_scores[keep]
            row_idx = row_idx[keep]
        total = len(pq_scores)
        if total == 0:
            return []

        # Pick the top ``candidate_pool`` by PQ score.  When a filter
        # is active, the pool is already drawn from the filtered
        # subset so the rerank stays pool-aware (no wasted slots on
        # out-of-filter ids).
        pool = min(max(rerank_candidates, k), total)
        top = np.argpartition(-pq_scores, pool - 1)[:pool]
        cand_rows = row_idx[top]

        # Float32 rerank: exact dot products against the cached vectors.
        cand_full = self._full_precision[cand_rows]      # (pool, dim_eff)
        exact_scores = cand_full @ q_pre                 # (pool,)
        if not self.normalized:
            exact_scores *= self._norms[cand_rows]

        k_eff = min(k, pool)
        final = np.argpartition(-exact_scores, k_eff - 1)[:k_eff]
        final = final[np.argsort(-exact_scores[final])]
        return [
            (self._ids_by_row[int(cand_rows[i])], float(exact_scores[i]))
            for i in final
        ]

    def _gather_pq_scores(
        self,
        probe: NDArray[np.int64],
        coarse_dot: NDArray[np.float32],
        lut: NDArray[np.float32],
        _parallel: bool = True,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Gather probed clusters → PQ-ADC scores + internal row ids.

        Single source of truth for the contiguous gather and the
        per-subspace ADC sum.  When ``normalized=False`` the scores
        are scaled by the per-vector norms so any downstream top-k
        (whether the default path or the rerank path) ranks by an
        unbiased estimate of ``⟨q, v⟩``, not ``⟨q, v̂⟩`` where v̂ is
        unit-length.

        Returns ``(scores, row_idx)``; both empty when no probed
        cluster contains any vectors.
        """
        starts = self._offsets[probe]
        ends = self._offsets[probe + 1]
        counts = ends - starts
        total = int(counts.sum())
        if total == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )

        row_idx = np.empty(total, dtype=np.int64)
        coarse_offsets = np.empty(total, dtype=np.float32)
        cursor = 0
        for s, e, c in zip(starts.tolist(), ends.tolist(), probe.tolist()):
            n_c = e - s
            if n_c == 0:
                continue
            row_idx[cursor : cursor + n_c] = np.arange(s, e, dtype=np.int64)
            coarse_offsets[cursor : cursor + n_c] = coarse_dot[c]
            cursor += n_c

        scores = np.empty(total, dtype=np.float32)
        assert int(row_idx.max()) < self._codes.shape[1] if total > 0 else True
        fused_gather_adc(self._codes, row_idx, coarse_offsets,
                         lut, scores, parallel=_parallel)

        if not self.normalized:
            scores *= self._norms[row_idx]

        return scores, row_idx

    def _score_one(
        self,
        probe: NDArray[np.int64],
        coarse_dot: NDArray[np.float32],
        lut: NDArray[np.float32],
        k: int,
        filter_rows: NDArray[np.int64] | None = None,
        _parallel: bool = True,
    ) -> list[tuple[Any, float]]:
        """Gather → ADC sum → top-k for a single query.

        Hot path; used by both ``search()`` and ``search_batch()``.
        """
        scores, row_idx = self._gather_pq_scores(
            probe, coarse_dot, lut, _parallel=_parallel,
        )
        if filter_rows is not None and len(row_idx) > 0:
            keep = np.isin(row_idx, filter_rows, assume_unique=True)
            scores = scores[keep]
            row_idx = row_idx[keep]
        total = len(scores)
        if total == 0:
            return []

        k_eff = min(k, total)
        top = np.argpartition(-scores, k_eff - 1)[:k_eff]
        top = top[np.argsort(-scores[top])]
        return [
            (self._ids_by_row[int(row_idx[i])], float(scores[i]))
            for i in top
        ]

    # ──────────────────────────────────────────────────────────────── #
    # batched search                                                    #
    # ──────────────────────────────────────────────────────────────── #

    def search_batch(
        self,
        queries: NDArray[np.float32],
        k: int = 10,
        nprobe: int | None = None,
        num_threads: int = 1,
        filter_ids: set[Any] | None = None,
    ) -> list[list[tuple[Any, float]]]:
        """Approximate top-k for a batch of queries.

        Throughput-oriented sibling of ``search()``.  Two things move:

        * **Coarse probe + LUT build run as one BLAS call each across
          the whole batch** instead of B per-query matmuls.  At
          B = 128, M = 192, K = 256 this alone is ~5× faster than
          looping ``search()``.
        * **Per-query gather + scoring** can optionally fan out over
          ``num_threads`` worker threads.  Unlike single-query
          threading (which competes with NumPy's internal BLAS pool),
          batch-level threading hands each thread a *whole* query's
          worth of work, so Python overhead is amortised over the
          query and the speedup actually shows up.

        Parameters
        ----------
        queries : NDArray[np.float32]
            Shape ``(B, dim)``.  Need not be normalized.
        k, nprobe : as in ``search()``.
        num_threads : int, default 1
            Worker threads for per-query scoring.  ``1`` is sequential
            (still benefits from the batched coarse + LUT).  ``> 1``
            engages a lazily-created ``ThreadPoolExecutor``.  Validate
            against your laptop core count before going above 4.
        filter_ids : set | None, default None
            Optional id whitelist shared by every query in the batch
            (typical use: tenant / partition scoping).  Same cluster-
            and pool-aware semantics as ``search()``.  The filter is
            resolved once per batch, not per query.

        Returns
        -------
        list of length B; each entry is the per-query top-k list of
        ``(id, score)`` pairs (same shape as ``search()``).  Queries
        with zero norm return ``[]`` for that slot.
        """
        self._require_fitted()
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if num_threads < 1:
            raise ValueError(f"num_threads must be >= 1; got {num_threads}")
        if nprobe is None:
            nprobe = self._default_nprobe
        if not (1 <= nprobe <= self.nlist):
            raise ValueError(
                f"nprobe must be in [1, nlist={self.nlist}]; got {nprobe}"
            )
        Q = np.asarray(queries, dtype=np.float32)
        if Q.ndim != 2 or Q.shape[1] != self.dim:
            raise ValueError(
                f"queries must be shape (B, {self.dim}); got {Q.shape}"
            )
        B = len(Q)
        if B == 0:
            return []
        if len(self._ids_by_row) == 0:
            return [[] for _ in range(B)]

        filter_rows: NDArray[np.int64] | None = None
        allowed_clusters: NDArray[np.int64] | None = None
        if filter_ids is not None:
            resolved = self._resolve_filter(filter_ids)
            if resolved is None:
                return [[] for _ in range(B)]
            filter_rows, allowed_clusters = resolved

        # Per-query unit normalisation (skip preprocess_single's
        # one-by-one path).  Zero-norm queries get marked invalid.
        # Optimized: ~4x faster than np.linalg.norm(..., axis=1) via einsum
        q_norms = np.sqrt(np.einsum('ij,ij->i', Q, Q))
        valid = q_norms >= 1e-10
        safe_norms = np.where(valid, q_norms, np.float32(1.0))
        Q_unit: NDArray[np.float32] = cast(
            "NDArray[np.float32]", Q / safe_norms[:, None]
        )

        q_pre_all: NDArray[np.float32]
        if self.use_rht:
            padded = np.zeros((B, self._pdim), dtype=np.float32)
            padded[:, : self.dim] = Q_unit
            q_pre_all = rht(padded, self.seed)
            # Optimized: ~4x faster than np.linalg.norm(..., axis=1) via einsum
            q_pre_all /= (
                np.sqrt(np.einsum('ij,ij->i', q_pre_all, q_pre_all))[:, np.newaxis]
                + np.float32(1e-12)
            )
        elif self.use_opq and self._opq_rotation is not None:
            # Apply the learned rotation to every query in the batch
            # so search_batch stays consistent with search() -- the
            # single-query path rotates via _preprocess_single.
            q_pre_all = cast(
                "NDArray[np.float32]", Q_unit @ self._opq_rotation
            )
        else:
            q_pre_all = Q_unit

        # One matmul, the whole batch.
        coarse_dot_all = q_pre_all @ self._coarse.T            # (B, nlist)
        cnorms = (self._coarse * self._coarse).sum(1)          # (nlist,)
        probe_ranking_all = 2.0 * coarse_dot_all - cnorms[None, :]
        if allowed_clusters is None:
            probes = np.argpartition(
                -probe_ranking_all, nprobe - 1, axis=1
            )[:, :nprobe]                                      # (B, nprobe)
        elif len(allowed_clusters) <= nprobe:
            # Every query probes the same small set of allowed clusters
            # exactly once — no argpartition needed, just broadcast.
            probes = np.tile(allowed_clusters, (B, 1))         # (B, |allowed|)
        else:
            # Pick top-nprobe per query via one vectorised argpartition
            # over the allowed-column slice of the coarse score matrix.
            restricted = probe_ranking_all[:, allowed_clusters]   # (B, |allowed|)
            idx = np.argpartition(-restricted, nprobe - 1, axis=1)[:, :nprobe]
            probes = allowed_clusters[idx]                     # (B, nprobe)

        # Batched residual LUTs via matmul (faster than einsum).
        q_split = q_pre_all.reshape(B, self.M, self._d_sub)
        q_t = np.transpose(q_split, (1, 0, 2))                  # (M, B, d_sub)
        cb_t = np.transpose(self._codebooks, (0, 2, 1))         # (M, d_sub, K)
        lut_batch = np.transpose(q_t @ cb_t, (1, 0, 2))         # (B, M, K)

        results: list[list[tuple[Any, float]]] = [[] for _ in range(B)]

        def _process(b: int) -> None:
            if not bool(valid[b]):
                return
            results[b] = self._score_one(
                probes[b], coarse_dot_all[b], lut_batch[b], k, filter_rows,
                _parallel=False,
            )

        if num_threads > 1 and B >= num_threads:
            # Double-checked locking: the common case (executor already
            # initialised with the requested worker count) skips the
            # lock entirely.  Only the first caller — or a caller that
            # asks for a different ``num_threads`` after init — enters
            # the slow path.
            #
            # Once the executor is created we never shut it down here:
            # another concurrent ``search_batch()`` caller may be about
            # to submit work to it, and ``shutdown()`` would make those
            # submissions raise "cannot schedule new futures after
            # shutdown".  Lifecycle management belongs in ``close()``.
            if (
                self._executor is None
                or self._executor_workers != num_threads
            ):
                with self._executor_lock:
                    if self._executor is None:
                        self._executor = ThreadPoolExecutor(
                            max_workers=num_threads,
                        )
                        self._executor_workers = num_threads
                    elif self._executor_workers != num_threads:
                        raise ValueError(
                            f"search_batch() cannot change num_threads "
                            f"after the executor has been initialised; "
                            f"requested {num_threads}, existing "
                            f"{self._executor_workers}.  Reuse the "
                            f"same num_threads or call close() first."
                        )
            executor = self._executor
            # ex.map blocks until all complete; futures populate results
            # in-place via the closure capture.
            list(executor.map(_process, range(B)))
        else:
            for b in range(B):
                _process(b)

        return results

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
            "keep_full_precision": self.keep_full_precision,
            "fitted": self._fitted,
            "bytes_per_vec": (
                bytes_per_vec
                + (self._pdim * 2 if self.keep_full_precision else 0)
            ),
            "bytes_per_vec_codes_only": bytes_per_vec,
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
        flags = 0
        if self.normalized:
            flags |= _FLAG_NORMALIZED
        if self.use_rht:
            flags |= _FLAG_USE_RHT
        if self.keep_full_precision:
            flags |= _FLAG_KEEP_FULL_PRECISION
        if self.use_opq:
            flags |= _FLAG_USE_OPQ
        n = len(self._ids_by_row)

        def _write(f: "ChecksumWriter") -> None:
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
            if self.use_opq:
                # (pdim, pdim) float32 rotation, placed after the
                # codebooks and before offsets.  v5 readers gate the
                # block on _FLAG_USE_OPQ; v4 and older readers refuse
                # v5 files at the version check above, so they never
                # reach this position.
                assert self._opq_rotation is not None
                f.write(self._opq_rotation.tobytes())
            f.write(self._offsets.tobytes())
            if n > 0:
                # _codes is shape (M, n); ascontiguousarray ensures the
                # M-major byte layout regardless of any prior slicing
                # that left the array as a non-contiguous view.
                f.write(np.ascontiguousarray(self._codes).tobytes())
                if not self.normalized:
                    f.write(self._norms.tobytes())
                if self.keep_full_precision:
                    f.write(np.ascontiguousarray(self._full_precision).tobytes())
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

        save_with_checksum_atomic(path, _write)

    @classmethod
    def load(cls, path: str | Path) -> "IVFPQSnapIndex":
        path = Path(path)
        verify_checksum(path)  # no-op for legacy files without a trailer
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"bad magic {magic!r}, expected {_MAGIC!r}")
            (version, dim, pdim, nlist, M, K, d_sub, seed, n, flags) = struct.unpack(
                "<IIIIIIIIII", f.read(40)
            )
            if version != _VERSION and version not in _LEGACY_VERSIONS:
                supported = sorted({_VERSION, *_LEGACY_VERSIONS})
                raise ValueError(
                    f"unsupported .snpi version {version}; this build of "
                    f"snapvec supports versions {supported}.  If {version} > "
                    f"{_VERSION} this file was written by a newer snapvec -- "
                    f"upgrade via `pip install -U snapvec`."
                )
            normalized = bool(flags & _FLAG_NORMALIZED)
            use_rht = bool(flags & _FLAG_USE_RHT)
            keep_full_precision = bool(flags & _FLAG_KEEP_FULL_PRECISION)
            use_opq = bool(flags & _FLAG_USE_OPQ)
            idx = cls(
                dim=dim, nlist=nlist, M=M, K=K, seed=seed,
                normalized=normalized, use_rht=use_rht,
                keep_full_precision=keep_full_precision,
                use_opq=use_opq,
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
            if use_opq:
                idx._opq_rotation = (
                    np.frombuffer(f.read(pdim * pdim * 4), dtype=np.float32)
                    .reshape(pdim, pdim).copy()
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
                # v1 stored codes as (n, M) row-major; v2+ stores
                # (M, n) column-major.  In both cases ``readinto`` into
                # a pre-allocated buffer avoids the ~n*M byte transient
                # that the old ``frombuffer(f.read(...)).copy()`` path
                # held alongside the final array (peak ~2x file size
                # during load -- painful at n=1M).
                if version == 1:
                    # v1: stream file through a small staging buffer and
                    # copy each chunk into the final (M, n) array with
                    # the transpose fused in -- never materialises the
                    # full (n, M) byte buffer.
                    idx._codes = np.empty((M, n), dtype=np.uint8)
                    chunk_n = 16_384
                    staging = np.empty((chunk_n, M), dtype=np.uint8)
                    for start in range(0, n, chunk_n):
                        cur = min(chunk_n, n - start)
                        f.readinto(staging[:cur].data)
                        idx._codes[:, start : start + cur] = staging[:cur].T
                    del staging
                else:
                    idx._codes = np.empty((M, n), dtype=np.uint8)
                    f.readinto(idx._codes.data)
                if not normalized:
                    idx._norms = np.empty(n, dtype=np.float32)
                    f.readinto(idx._norms.data)
                if keep_full_precision:
                    # v3 stored the cache as float32 (4 bytes/value);
                    # v4+ uses float16 (2 bytes/value).  Cast the v3
                    # legacy payload down on load so the in-memory
                    # layout is always fp16 regardless of file age.
                    if version <= 3:
                        # Stream the cast in row-chunks so peak RAM
                        # stays bounded instead of materialising the
                        # full float32 payload (1.5 GB at N=1M, d=384)
                        # alongside the fp16 output.
                        idx._full_precision = np.empty(
                            (n, pdim), dtype=np.float16,
                        )
                        rows_per_chunk = max(
                            1, (1 << 20) // (pdim * 4),  # ≈ 1 MB of fp32 per chunk
                        )
                        for start in range(0, n, rows_per_chunk):
                            end = min(start + rows_per_chunk, n)
                            chunk = np.frombuffer(
                                f.read((end - start) * pdim * 4),
                                dtype=np.float32,
                            ).reshape(end - start, pdim)
                            idx._full_precision[start:end] = chunk
                    else:
                        idx._full_precision = np.empty(
                            (n, pdim), dtype=np.float16,
                        )
                        f.readinto(idx._full_precision.data)
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
