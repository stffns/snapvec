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
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from ._fast import fused_gather_adc
except ImportError:
    from ._fast_fallback import fused_gather_adc  # type: ignore[assignment]
from ._file_format import save_with_checksum_atomic, verify_checksum
from ._freezable import FreezableIndex
from ._kmeans import assign_l2, kmeans_mse, probe_scores_l2_monotone
from ._rotation import padded_dim, rht

_MAGIC = b"SNPI"
_VERSION = 4  # v4: rerank cache stored as float16 (half the storage of v3)
_LEGACY_VERSIONS = {1, 2, 3}
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
        self.keep_full_precision = keep_full_precision

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

        # Combine with existing state, sort once by cluster id.
        if self._codes.shape[1] == 0:
            combined_codes = new_codes
            combined_asn = new_asn
            combined_ids_seq: list[Any] = list(ids)
            combined_norms = new_norms
            combined_full = new_full
        else:
            combined_codes = np.concatenate([self._codes, new_codes], axis=1)
            combined_asn = np.concatenate(
                [self._cluster_ids_from_offsets(), new_asn]
            )
            combined_ids_seq = self._ids_by_row + list(ids)
            combined_norms = (
                new_norms if self.normalized
                else np.concatenate([self._norms, new_norms])
            )
            combined_full = (
                np.concatenate([self._full_precision, new_full], axis=0)
                if self.keep_full_precision else new_full
            )

        order = np.argsort(combined_asn, kind="stable")
        self._codes = combined_codes[:, order]
        # Reorder ids via numpy object array → bulk gather in C, no
        # Python list comprehension over N elements.
        ids_arr = np.array(combined_ids_seq, dtype=object)
        self._ids_by_row = ids_arr[order].tolist()
        if not self.normalized:
            self._norms = combined_norms[order]
        if self.keep_full_precision:
            self._full_precision = combined_full[order]
        counts = np.bincount(combined_asn, minlength=self.nlist)
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
        if float(np.linalg.norm(q)) < 1e-10:
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
        q_norms = np.linalg.norm(Q, axis=1)
        valid = q_norms >= 1e-10
        safe_norms = np.where(valid, q_norms, 1.0).astype(np.float32)
        Q_unit = Q / safe_norms[:, None]

        if self.use_rht:
            padded = np.zeros((B, self._pdim), dtype=np.float32)
            padded[:, : self.dim] = Q_unit
            q_pre_all = rht(padded, self.seed)
            q_pre_all /= np.linalg.norm(q_pre_all, axis=1, keepdims=True) + 1e-12
        else:
            q_pre_all = Q_unit.astype(np.float32)

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
        n = len(self._ids_by_row)

        def _write(f):
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
            idx = cls(
                dim=dim, nlist=nlist, M=M, K=K, seed=seed,
                normalized=normalized, use_rht=use_rht,
                keep_full_precision=keep_full_precision,
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
                # v1 stored codes as (n, M) row-major; v2 stores
                # (M, n) column-major.  Transparently transpose v1 on
                # load so search/add see the v2 layout.
                if version == 1:
                    idx._codes = (
                        np.frombuffer(f.read(n * M), dtype=np.uint8)
                        .reshape(n, M).T.copy()
                    )
                else:
                    idx._codes = (
                        np.frombuffer(f.read(M * n), dtype=np.uint8)
                        .reshape(M, n).copy()
                    )
                if not normalized:
                    idx._norms = np.frombuffer(f.read(n * 4), dtype=np.float32).copy()
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
                        idx._full_precision = (
                            np.frombuffer(f.read(n * pdim * 2), dtype=np.float16)
                            .reshape(n, pdim).copy()
                        )
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
