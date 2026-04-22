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

Memory layout
-------------
``_indices``  — bit-packed uint8 for all bit widths (both RAM and on disk).
               At 4-bit: 2 indices per byte.  At 2-bit: 4 per byte.
               At 3-bit: 8 indices are packed tightly across 3 bytes
               (0.375 bytes/coord).  Since ``pdim`` is a power of 2 ≥ 8,
               each row is byte-aligned and RAM layout == disk layout.
``_cache``    — (N, pdim) float16, lazy centroid expansion.  Evicted on writes.
               For N > ~500k the cache dominates RAM; use ``chunk_size`` to
               search without materialising it (streaming mode).

Reference: Zandieh, Daliri, Hadian, Mirrokni (2025).
    "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."
    arXiv:2504.19874.  ICLR 2026.

File format
-----------
v1 (legacy)  — mse only, 20-byte header
v2           — adds flags field (bit-0 = use_prod, bit-1 = normalized)
               + QJL arrays.  3-bit indices are byte-aligned (wastes 2 bits
               per byte: 0.5 bytes/coord instead of the theoretical 0.375).
v3           — same layout as v2, but 3-bit indices are tightly packed
               (8 indices → 3 bytes = 0.375 bytes/coord).  2-bit and 4-bit
               layouts are identical in v2/v3 and remain round-trip
               compatible.
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

_MAGIC = b"SNPV"
_VERSION = 3  # v3: tight 3-bit packing (0.375 bytes/coord); see File format notes
_LEGACY_VERSIONS: tuple[int, ...] = (1, 2)
_FLAG_PROD = 0x1
_FLAG_NORMALIZED = 0x2


class SnapIndex(FreezableIndex):
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
    chunk_size : int | None
        If set, search processes ``chunk_size`` rows at a time without
        materialising the full float16 cache.  Use for N > 500k to trade
        compute for memory.  ``None`` (default) uses the cached matmul.
    normalized : bool
        If True, input vectors are assumed to already be unit-length.
        Skips norm computation in add/add_batch.

    Examples
    --------
    >>> import numpy as np
    >>> from snapvec import SnapIndex
    >>> idx = SnapIndex(dim=384, bits=4)
    >>> vecs = np.random.randn(1000, 384).astype(np.float32)
    >>> idx.add_batch(list(range(1000)), vecs)
    >>> results = idx.search(vecs[0], k=5)
    >>> results[0][0]  # top match is the vector itself
    0
    """

    def __init__(
        self,
        dim: int,
        bits: int = 4,
        seed: int = 0,
        use_prod: bool = False,
        chunk_size: int | None = None,
        normalized: bool = False,
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
        self.chunk_size = chunk_size
        self.normalized = normalized

        self._pdim: int = padded_dim(dim)
        self._mse_bits: int = bits - 1 if use_prod else bits
        self._centroids: NDArray[np.float32]
        self._centroids, _ = get_codebook(self._mse_bits)

        # RAM bit-packing.  We pack whenever (pdim * mse_bits) is a multiple
        # of 8 — always true for pdim = 2^k with k >= 3 (pdim >= 8).
        # For mse_bits ∈ {2, 4} we use byte-aligned packing (ipb per byte).
        # For mse_bits = 3 we use tight packing (8 indices → 3 bytes).  Both
        # layouts are row-contiguous, so the RAM buffer matches the on-disk
        # bit-packed stream byte-for-byte (enabling zero-copy save/load).
        self._can_pack: bool = (self._pdim * self._mse_bits) % 8 == 0
        # _ipb only applies to byte-aligned modes; 0 marks the tight 3-bit path
        self._ipb: int = 8 // self._mse_bits if (8 % self._mse_bits == 0) else 0
        _pcols = (self._pdim * self._mse_bits) // 8 if self._can_pack else self._pdim

        # Core storage
        self._ids: list[Any] = []
        self._id_to_pos: dict[Any, int] = {}
        self._indices: NDArray[np.uint8] = np.zeros((0, _pcols), dtype=np.uint8)
        self._norms: NDArray[np.float32] = np.zeros(0, dtype=np.float32)

        # TurboQuant_prod arrays (None when use_prod=False)
        self._qjl: NDArray[np.int8] | None = None
        self._rnorms: NDArray[np.float32] | None = None
        self._S: NDArray[np.float32] | None = None

        if use_prod:
            rng = np.random.default_rng(seed + 1337)
            self._S = rng.standard_normal((self._pdim, self._pdim)).astype(np.float32)

        # float16 centroid cache — None until first search, evicted on writes
        self._cache: NDArray[np.float16] | None = None

    def freeze(self) -> None:
        """Freeze + pre-warm the lazy centroid cache.

        ``_search_cached`` materialises ``self._cache`` on first call
        (see the non-chunked default path).  If the caller froze the
        index without doing a warm-up query, two concurrent searches
        would race on that assignment — breaking the thread-safety
        contract ``FreezableIndex`` documents.  We pre-warm here so
        every post-freeze ``search()`` only reads the cache.

        Chunked mode (``chunk_size is not None``) never touches
        ``self._cache`` and the filtered-subset path builds per-query
        work into local arrays, so those paths are already safe
        without pre-warming.
        """
        if (
            self.chunk_size is None
            and self._cache is None
            and len(self._ids) > 0
        ):
            indices = self._unpack_to_indices()
            self._cache = self._centroids[indices].astype(np.float16)
        super().freeze()

    # ──────────────────────────────────────────────────────────────────── #
    # RAM bit-packing helpers                                               #
    # ──────────────────────────────────────────────────────────────────── #

    def _pack_matrix(self, mat: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Pack (N, pdim) uint8 indices → row-packed uint8 for RAM.

        4-bit: 2 indices per byte.  2-bit: 4 per byte.  3-bit: 8 indices
        across 3 bytes (tight, 0.375 bytes/coord).  Each row is byte-
        aligned (``pdim * mse_bits`` is divisible by 8 for pdim = 2^k ≥ 8),
        so the flat layout = per-row layout — we reuse the disk helpers.
        """
        bits = self._mse_bits
        n = len(mat)
        if bits == 3:
            return _pack_3bit_tight(mat.ravel()).reshape(n, -1)
        ipb = self._ipb
        mask = (1 << bits) - 1
        packed: NDArray[np.uint8] = np.zeros(
            (n, self._pdim // ipb), dtype=np.uint8,
        )
        for i in range(ipb):
            packed |= (mat[:, i::ipb] & mask).astype(np.uint8) << (i * bits)
        return packed

    def _unpack_to_indices(
        self, packed: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Unpack RAM-packed indices to (N, pdim) uint8.

        Returns the input unchanged when RAM packing is not active.
        """
        if packed is None:
            packed = self._indices
        if not self._can_pack:
            return packed
        bits = self._mse_bits
        n = len(packed)
        if bits == 3:
            return _unpack_3bit_tight(packed.ravel()).reshape(n, -1)
        ipb = self._ipb
        mask = (1 << bits) - 1
        mat = np.empty((n, packed.shape[1] * ipb), dtype=np.uint8)
        for i in range(ipb):
            mat[:, i::ipb] = ((packed >> (i * bits)) & mask).astype(np.uint8)
        return mat

    # ──────────────────────────────────────────────────────────────────── #
    # Core API                                                              #
    # ──────────────────────────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        mode = "prod" if self.use_prod else "mse"
        extra = ", normalized" if self.normalized else ""
        return (
            f"SnapIndex(dim={self.dim}, bits={self.bits}, "
            f"mode={mode}, n={len(self)}{extra})"
        )

    def add(self, id: Any, vector: NDArray[np.float32]) -> None:
        """Add a single vector.  Prefer :meth:`add_batch` for bulk inserts."""
        self._check_not_frozen("add")
        self.add_batch([id], np.asarray(vector, dtype=np.float32).reshape(1, -1))

    def add_batch(self, ids: list[Any], vectors: NDArray[np.float32]) -> None:
        """Add vectors in bulk — ~50x faster than repeated :meth:`add`.

        Parameters
        ----------
        ids : list[Any]
            Identifier for each vector (int, str, …).  Must be unique.
        vectors : NDArray[np.float32], shape (n, dim)
            Raw embedding vectors (need not be normalized).
        """
        self._check_not_frozen("add_batch")
        arr = np.asarray(vectors, dtype=np.float32)
        n = len(arr)
        if n == 0:
            return

        # 1. Normalize to unit sphere (skipped when self.normalized=True,
        #    i.e. the caller guarantees inputs already have unit length)
        if self.normalized:
            units: NDArray[np.float32] = arr
            batch_norms: NDArray[np.float32] = np.ones(n, dtype=np.float32)
        else:
            # Optimized: ~4x faster than np.linalg.norm(..., axis=1) via einsum
            raw_norms: NDArray[np.float32] = np.sqrt(np.einsum('ij,ij->i', arr, arr))
            safe: NDArray[np.float32] = np.where(raw_norms > 1e-10, raw_norms, 1.0)
            units = arr / safe[:, None]
            batch_norms = np.where(
                raw_norms > 1e-10, raw_norms, 0.0
            ).astype(np.float32)

        # 2. Zero-pad to power of 2, batch RHT — O(n·d·log d)
        pdim = self._pdim
        padded: NDArray[np.float32] = np.zeros((n, pdim), dtype=np.float32)
        padded[:, : self.dim] = units
        rotated: NDArray[np.float32] = rht(padded, self.seed)
        # ``np.sqrt(pdim)`` returns a numpy float64 scalar (even on an int
        # input), which would promote the float32 ``rotated`` to float64
        # -- silently contradicting the NDArray[np.float32] annotation.
        # Wrap in ``np.float32`` so the multiplication stays in float32.
        scaled: NDArray[np.float32] = rotated * np.float32(np.sqrt(pdim))

        # 3. MSE quantization at _mse_bits
        _, boundaries = get_codebook(self._mse_bits)
        flat_idx: NDArray[np.uint8] = np.searchsorted(
            boundaries, scaled.ravel()
        ).astype(np.uint8)
        batch_idx: NDArray[np.uint8] = flat_idx.reshape(n, pdim)

        # 4. QJL residual (prod mode only)
        qjl_signs: NDArray[np.int8] | None = None
        residual_norms: NDArray[np.float32] | None = None
        if self.use_prod:
            assert self._S is not None
            reconstructed: NDArray[np.float32] = self._centroids[batch_idx]
            r_scaled: NDArray[np.float32] = scaled - reconstructed
            # sign(S·r_rot) = sign(S·r_scaled) — scale-invariant
            S_r: NDArray[np.float32] = (self._S @ r_scaled.T).T
            qjl_signs = np.sign(S_r).astype(np.int8)
            qjl_signs[qjl_signs == 0] = 1
            # Store ‖r_rot‖ = ‖r_scaled‖/√pdim (unscaled space norm)
            # Optimized: ~4x faster than np.linalg.norm(..., axis=1) via einsum
            residual_norms = np.sqrt(
                np.einsum('ij,ij->i', r_scaled, r_scaled)
            ) / np.float32(np.sqrt(pdim))

        # 5. RAM bit-pack indices when possible
        if self._can_pack:
            batch_idx = self._pack_matrix(batch_idx)

        # 6. Append to storage arrays
        start = len(self._ids)
        self._ids.extend(ids)
        for i, id_val in enumerate(ids):
            self._id_to_pos[id_val] = start + i

        self._indices = (
            batch_idx
            if len(self._indices) == 0
            else np.vstack([self._indices, batch_idx])
        )
        self._norms = (
            batch_norms
            if len(self._norms) == 0
            else np.concatenate([self._norms, batch_norms])
        )

        if self.use_prod:
            assert qjl_signs is not None and residual_norms is not None
            self._qjl = (
                qjl_signs
                if self._qjl is None
                else np.vstack([self._qjl, qjl_signs])
            )
            self._rnorms = (
                residual_norms
                if self._rnorms is None
                else np.concatenate([self._rnorms, residual_norms])
            )

        self._cache = None  # evict

    def delete(self, id: Any) -> bool:
        """Remove a vector by id.  O(1) lookup, O(1) delete via swap-with-last.

        Returns True if the id was found and removed, False otherwise.
        """
        self._check_not_frozen("delete")
        if id not in self._id_to_pos:
            return False

        pos = self._id_to_pos.pop(id)
        last_pos = len(self._ids) - 1

        if pos != last_pos:
            last_id = self._ids[last_pos]
            self._ids[pos] = last_id
            self._id_to_pos[last_id] = pos

            self._indices[pos] = self._indices[last_pos]
            self._norms[pos] = self._norms[last_pos]
            if self._qjl is not None:
                self._qjl[pos] = self._qjl[last_pos]
                assert self._rnorms is not None
                self._rnorms[pos] = self._rnorms[last_pos]

        self._ids.pop()
        self._indices = self._indices[:-1]
        self._norms = self._norms[:-1]
        if self._qjl is not None:
            self._qjl = self._qjl[:-1]
            assert self._rnorms is not None
            self._rnorms = self._rnorms[:-1]

        self._cache = None
        return True

    def search(
        self,
        query: NDArray[np.float32],
        k: int = 10,
        filter_ids: set[Any] | None = None,
    ) -> list[tuple[Any, float]]:
        """Find k nearest neighbors by approximate cosine similarity.

        Parameters
        ----------
        query : NDArray[np.float32], shape (dim,)
            Query vector (raw, need not be normalized).
        k : int
            Number of results to return.
        filter_ids : set | None
            If provided, restrict search to this subset of ids.
            Uses O(1) dict lookups — cost is O(|filter_ids| · d) instead
            of O(N · d).  Useful for collection/partition filtering.

        Returns
        -------
        list of (id, score) sorted by descending similarity.
        Score is an approximation of cosine similarity in [-1, 1].

        Notes
        -----
        When ``chunk_size`` is set, search processes rows in chunks without
        materialising the full float16 cache.  This trades peak RAM for
        additional compute — useful when N > ~500k vectors.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1; got {k}")
        if not self._ids:
            return []

        q = np.asarray(query, dtype=np.float32)
        q_norm: float = float(np.linalg.norm(q))
        if q_norm < 1e-10:
            return []

        pdim = self._pdim
        # Python float ``q_norm`` is float64 under pre-NEP-50 numpy; wrap
        # so ``q_unit`` stays in float32.
        q_unit: NDArray[np.float32] = cast(
            "NDArray[np.float32]", q / np.float32(q_norm)
        )
        q_padded: NDArray[np.float32] = np.zeros(pdim, dtype=np.float32)
        q_padded[: self.dim] = q_unit
        q_rot: NDArray[np.float32] = rht(q_padded, self.seed)
        # np.sqrt(int) returns a numpy float64 scalar; wrap in np.float32
        # so the multiply stays in float32 (matches the type annotation).
        q_scaled: NDArray[np.float32] = q_rot * np.float32(np.sqrt(pdim))

        # Resolve filter_ids → sorted row positions (O(|filter_ids|))
        if filter_ids is not None:
            rows: NDArray[np.intp] = np.array(
                [self._id_to_pos[i] for i in filter_ids if i in self._id_to_pos],
                dtype=np.intp,
            )
            if len(rows) == 0:
                return []
            rows.sort()  # contiguous access is faster for cache/chunked
            active_ids = [self._ids[r] for r in rows]
            packed_slice = self._indices[rows]
            qjl = self._qjl[rows] if self._qjl is not None else None
            rnorms = self._rnorms[rows] if self._rnorms is not None else None
        else:
            active_ids = self._ids
            packed_slice = self._indices
            qjl = self._qjl
            rnorms = self._rnorms

        scores: NDArray[np.float32]
        if self.chunk_size is not None:
            scores = self._search_chunked(packed_slice, q_scaled)
        elif filter_ids is None:
            # Full scan — use lazy float16 cache (built once, reused across queries)
            scores = self._search_cached(q_scaled)
        else:
            # Filtered subset — skip cache, compute directly on the slice
            indices = self._unpack_to_indices(packed_slice)
            expanded: NDArray[np.float16] = self._centroids[indices].astype(np.float16)
            # fp16 @ fp32 matmul already yields float32; divide by a
            # Python int preserves float32 under every supported numpy
            # version.  No astype needed.
            raw = expanded @ q_scaled
            scores = cast("NDArray[np.float32]", raw / pdim)

        # QJL correction (prod mode)
        if self.use_prod and qjl is not None and rnorms is not None:
            scores = self._apply_qjl_arrays(scores, q_scaled, qjl, rnorms)

        actual_k = min(k, len(active_ids))
        top_idx: NDArray[np.intp] = np.argpartition(scores, -actual_k)[-actual_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(active_ids[int(i)], float(scores[i])) for i in top_idx]

    # ──────────────────────────────────────────────────────────────────── #
    # Search internals                                                      #
    # ──────────────────────────────────────────────────────────────────── #

    def _search_cached(self, q_scaled: NDArray[np.float32]) -> NDArray[np.float32]:
        """Single float16 matmul using the lazy centroid cache (full index).

        Builds ``_cache`` once and reuses it across queries until a write
        invalidates it.  Best for repeated queries on a stable index.
        Peak RAM: O(N·d·2 bytes) for the float16 cache.
        """
        if self._cache is None:
            indices = self._unpack_to_indices()
            self._cache = self._centroids[indices].astype(np.float16)
        # fp16 @ fp32 matmul already yields float32; divide by a Python
        # int preserves float32.  No astype needed.
        raw = self._cache @ q_scaled
        return cast("NDArray[np.float32]", raw / self._pdim)

    def _search_chunked(
        self,
        packed_indices: NDArray[np.uint8],
        q_scaled: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Chunked search — unpacks per chunk, no full cache materialised.

        Processes ``chunk_size`` rows at a time.  Works for both the full
        index and filtered subsets (``filter_ids``).
        """
        assert self.chunk_size is not None
        n = len(packed_indices)
        scores = np.empty(n, dtype=np.float32)
        cs = self.chunk_size
        for start in range(0, n, cs):
            end = min(start + cs, n)
            chunk_idx = self._unpack_to_indices(packed_indices[start:end])
            chunk: NDArray[np.float16] = self._centroids[
                chunk_idx
            ].astype(np.float16)
            scores[start:end] = (chunk @ q_scaled) / self._pdim
        return scores

    def _apply_qjl_arrays(
        self,
        scores: NDArray[np.float32],
        q_scaled: NDArray[np.float32],
        qjl: NDArray[np.int8],
        rnorms: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Add QJL unbiased correction term (prod mode only).

        Formula (Zandieh et al., Lemma 4):
            correctionᵢ = √(π/2) / d · ‖rᵢ‖ · dot(S·q̂, sign(S·rᵢ))

        Accepts explicit qjl/rnorms arrays to support filtered subsets.
        """
        assert self._S is not None
        q_unit_rot: NDArray[np.float32] = q_scaled / np.sqrt(self._pdim)
        S_q: NDArray[np.float32] = self._S @ q_unit_rot
        qjl_dots: NDArray[np.float32] = np.dot(qjl, S_q)
        correction: NDArray[np.float32] = (
            np.sqrt(np.pi / 2.0) / self._pdim * rnorms * qjl_dots
        )
        return scores + correction

    # ──────────────────────────────────────────────────────────────────── #
    # Persistence                                                           #
    # ──────────────────────────────────────────────────────────────────── #

    def save(self, path: str | Path) -> None:
        """Persist index to a binary ``.snpv`` file (atomic write).

        Writes to ``<path>.tmp`` first, then renames atomically.
        Indices are bit-packed on disk (b/8 bytes per coordinate).
        An 8-byte CRC32 trailer is appended so silent disk / transport
        corruption is caught at ``load()`` time.
        """
        n = len(self._ids)
        flags = 0
        if self.use_prod:
            flags |= _FLAG_PROD
        if self.normalized:
            flags |= _FLAG_NORMALIZED

        # When RAM is bit-packed, the layout already matches disk bit-packing
        # byte-for-byte (C-order ravel of (N, pdim/ipb) uint8) — skip the
        # unpack/repack round-trip entirely.
        if self._can_pack:
            packed = self._indices.tobytes()
        else:
            packed = _pack(self._indices, self._mse_bits)

        def _write(f: "ChecksumWriter") -> None:
            f.write(_MAGIC)
            f.write(struct.pack("<IIIIII", _VERSION, self.dim, self.bits, self.seed, n, flags))
            f.write(struct.pack("<I", len(packed)))
            f.write(packed)
            f.write(self._norms.tobytes())
            if self.use_prod and self._qjl is not None:
                assert self._rnorms is not None
                f.write(self._qjl.tobytes())
                f.write(self._rnorms.tobytes())
            for id_val in self._ids:
                enc = str(id_val).encode("utf-8")
                f.write(struct.pack("<H", len(enc)))
                f.write(enc)

        save_with_checksum_atomic(path, _write)

    @classmethod
    def load(cls, path: str | Path) -> "SnapIndex":
        """Load index from a ``.snpv`` file.

        Supports v1 (mse-only legacy) and v2 (prod/flags) formats.
        Verifies the CRC32 trailer when present (files saved with
        snapvec ≥ 0.7); legacy files without a trailer load without
        integrity checking for backward compat.
        """
        path = Path(path)
        verify_checksum(path)
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != _MAGIC:
                raise ValueError(f"Invalid file: expected {_MAGIC!r} magic, got {magic!r}")

            (version,) = struct.unpack("<I", f.read(4))
            if version == 1:
                dim, bits, seed, n = struct.unpack("<IIII", f.read(16))
                use_prod = False
                normalized = False
            elif version in (2, 3):
                dim, bits, seed, n, flags = struct.unpack("<IIIII", f.read(20))
                use_prod = bool(flags & _FLAG_PROD)
                normalized = bool(flags & _FLAG_NORMALIZED)
            else:
                supported = sorted({_VERSION, *_LEGACY_VERSIONS})
                raise ValueError(
                    f"unsupported .snpv version {version}; this build of "
                    f"snapvec supports versions {supported}.  If {version} > "
                    f"{_VERSION} this file was written by a newer snapvec -- "
                    f"upgrade via `pip install -U snapvec`."
                )

            idx = cls(dim=dim, bits=bits, seed=seed, use_prod=use_prod,
                      normalized=normalized)
            pdim = idx._pdim
            mse_bits = idx._mse_bits
            # v1/v2 used byte-aligned 3-bit packing; v3 is tight.
            legacy_3bit = (version < 3 and mse_bits == 3)

            (packed_len,) = struct.unpack("<I", f.read(4))
            if idx._can_pack and not legacy_3bit:
                # Disk layout matches RAM layout -- read directly into a
                # pre-allocated buffer.  Avoids the transient ``data =
                # f.read(packed_len)`` bytes object holding a second copy
                # of the payload alongside the final array.
                packed_cols = (pdim * mse_bits) // 8
                idx._indices = np.empty((n, packed_cols), dtype=np.uint8)
                f.readinto(idx._indices.data)
            else:
                data = f.read(packed_len)
                unpacked = _unpack(
                    data, n, pdim, mse_bits, legacy_3bit=legacy_3bit,
                )
                idx._indices = (
                    idx._pack_matrix(unpacked) if idx._can_pack else unpacked
                )
            idx._norms = np.empty(n, dtype=np.float32)
            f.readinto(idx._norms.data)

            if use_prod and n > 0:
                idx._qjl = np.empty((n, pdim), dtype=np.int8)
                f.readinto(idx._qjl.data)
                idx._rnorms = np.empty(n, dtype=np.float32)
                f.readinto(idx._rnorms.data)

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
        """Return memory and compression statistics."""
        n = len(self._ids)
        orig = n * self.dim * 4
        mse_b = int(self._indices.nbytes + self._norms.nbytes)
        qjl_b = int(
            (self._qjl.nbytes + self._rnorms.nbytes)
            if self._qjl is not None and self._rnorms is not None else 0
        )
        cache_b = int(self._cache.nbytes) if self._cache is not None else 0
        total = mse_b + qjl_b
        return {
            "n": n,
            "dim": self.dim,
            "padded_dim": self._pdim,
            "bits": self.bits,
            "mse_bits": self._mse_bits,
            "use_prod": self.use_prod,
            "normalized": self.normalized,
            "ram_packed": self._can_pack,
            "chunk_size": self.chunk_size,
            "float32_bytes": orig,
            "compressed_bytes": mse_b,
            "qjl_bytes": qjl_b,
            "cache_bytes": cache_b,
            "total_bytes": total,
            "compression_ratio": round(orig / max(total, 1), 2),
        }


# ──────────────────────────────────────────────────────────────────────────── #
# Bit-packing helpers                                                          #
#                                                                              #
# Packing scheme per ``mse_bits`` (b):                                         #
#   b = 2 : 4 indices / byte, byte-aligned          (0.25 bytes/coord)         #
#   b = 3 : 8 indices / 3 bytes, cross-byte tight   (0.375 bytes/coord)        #
#   b = 4 : 2 indices / byte, byte-aligned          (0.50 bytes/coord)         #
#                                                                              #
# Since ``pdim`` is always a power of 2 ≥ 8, ``pdim * b`` is divisible by 8    #
# for all three cases — so rows are byte-aligned and the RAM-packed matrix    #
# ``_indices`` has the same byte layout as the on-disk bit-packed stream.     #
# ``save`` and ``load`` exploit this: indices are written / read with         #
# ``tobytes()`` / ``np.frombuffer`` directly, with no unpack+repack trip.      #
#                                                                              #
# Unpack-to-uint8 happens in three places:                                     #
#   (a) first cached search, when building the lazy float16 centroid matrix;  #
#   (b) ``_search_chunked``, once per chunk of ``chunk_size`` rows;           #
#   (c) ``search(filter_ids=...)``, once per query on the filtered slice.    #
#                                                                              #
# So the unpack is OFF the hot matmul path in the cached full-scan mode;      #
# chunked and filtered paths pay it per chunk/query (still cheap: a few       #
# vectorised NumPy shifts on small slices).                                   #
#                                                                              #
# The legacy (v1/v2) disk format used byte-aligned 3-bit packing (2 indices   #
# per byte, wasting 2 bits per byte). ``_unpack(..., legacy_3bit=True)``     #
# decodes that layout so pre-v0.3 files remain loadable.                     #
# ──────────────────────────────────────────────────────────────────────────── #

def _pack_3bit_tight(flat: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack a 1-D array of 3-bit indices into tight bytes (8 values → 3 bytes).

    ``len(flat)`` must be a multiple of 8.  Returned array has length
    ``3 * len(flat) / 8``.  Single implementation reused by both the RAM
    row-packer and the on-disk flat packer.
    """
    chunks = flat.reshape(-1, 8).astype(np.uint32)
    combined = (
        chunks[:, 0]
        | (chunks[:, 1] << 3)
        | (chunks[:, 2] << 6)
        | (chunks[:, 3] << 9)
        | (chunks[:, 4] << 12)
        | (chunks[:, 5] << 15)
        | (chunks[:, 6] << 18)
        | (chunks[:, 7] << 21)
    )
    out = np.empty((len(combined), 3), dtype=np.uint8)
    out[:, 0] = combined & 0xFF
    out[:, 1] = (combined >> 8) & 0xFF
    out[:, 2] = (combined >> 16) & 0xFF
    return out.ravel()


def _unpack_3bit_tight(packed: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Inverse of :func:`_pack_3bit_tight`.

    ``len(packed)`` must be a multiple of 3.  Returned array has length
    ``8 * len(packed) / 3`` — callers slice off any trailing padding.
    """
    chunks = packed.reshape(-1, 3).astype(np.uint32)
    combined = chunks[:, 0] | (chunks[:, 1] << 8) | (chunks[:, 2] << 16)
    out = np.empty((len(combined), 8), dtype=np.uint8)
    for i in range(8):
        out[:, i] = ((combined >> (i * 3)) & 0x7).astype(np.uint8)
    return out.ravel()


def _pack(mat: NDArray[np.uint8], bits: int) -> bytes:
    """Pack uint8 index matrix into a bit-packed byte string.

    2-bit and 4-bit: byte-aligned (ipb = 8 // bits indices per byte).
    3-bit: tight packing (8 indices → 3 bytes = 24 bits), v3 format.
    """
    if bits == 8:
        return mat.tobytes()
    if bits == 3:
        flat = mat.ravel()
        pad = (-len(flat)) % 8
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
        return _pack_3bit_tight(flat).tobytes()
    ipb = 8 // bits
    mask = (1 << bits) - 1
    flat = mat.ravel()
    pad = (-len(flat)) % ipb
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    out = np.zeros(len(flat) // ipb, dtype=np.uint8)
    for i in range(ipb):
        out |= (flat[i::ipb] & mask).astype(np.uint8) << (i * bits)
    return out.tobytes()


def _unpack(
    data: bytes, n_rows: int, n_cols: int, bits: int, *, legacy_3bit: bool = False,
) -> NDArray[np.uint8]:
    """Unpack bit-packed bytes back to uint8 matrix.

    ``legacy_3bit=True`` decodes the byte-aligned 3-bit layout used by the
    v1 / v2 file formats (ipb = 2, wastes 2 bits per byte).  v3 files use
    the default tight 3-bit layout.
    """
    total = n_rows * n_cols
    if bits == 8:
        result: NDArray[np.uint8] = (
            np.frombuffer(data, dtype=np.uint8).reshape(n_rows, n_cols).copy()
        )
        return result
    if bits == 3 and not legacy_3bit:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cast(
            "NDArray[np.uint8]",
            _unpack_3bit_tight(arr)[:total].reshape(n_rows, n_cols).copy(),
        )
    # Byte-aligned path: 2-bit, 4-bit, and legacy 3-bit (v1/v2)
    ipb = 8 // bits
    mask = (1 << bits) - 1
    packed: NDArray[np.uint8] = np.frombuffer(data, dtype=np.uint8)
    out: NDArray[np.uint8] = np.zeros(len(packed) * ipb, dtype=np.uint8)
    for i in range(ipb):
        out[i::ipb] = (packed >> (i * bits)) & mask
    unpacked: NDArray[np.uint8] = out[:total].reshape(n_rows, n_cols).copy()
    return unpacked
