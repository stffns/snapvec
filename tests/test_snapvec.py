"""Tests for snapvec — SnapIndex correctness, persistence, and compression."""
from __future__ import annotations

import numpy as np
import pytest

from snapvec import SnapIndex

RNG = np.random.default_rng(42)
DIM = 128


def _rand(n: int, dim: int = DIM) -> np.ndarray:
    v = RNG.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


# ─────────────────────────────────────────────── Basic API ──────────────────

class TestBasic:
    def test_add_len(self):
        idx = SnapIndex(dim=DIM, bits=4)
        assert len(idx) == 0
        idx.add_batch(list(range(10)), _rand(10))
        assert len(idx) == 10

    def test_search_returns_self(self):
        idx = SnapIndex(dim=DIM, bits=4)
        vecs = _rand(100)
        idx.add_batch(list(range(100)), vecs)
        results = idx.search(vecs[0], k=5)
        ids = [r[0] for r in results]
        assert 0 in ids, "top-5 should include the query vector itself"

    def test_search_scores_descending(self):
        idx = SnapIndex(dim=DIM, bits=4)
        vecs = _rand(50)
        idx.add_batch(list(range(50)), vecs)
        results = idx.search(vecs[5], k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_single(self):
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add(99, _rand(1)[0])
        assert len(idx) == 1

    def test_repr(self):
        idx = SnapIndex(dim=64, bits=3)
        assert "64" in repr(idx)
        assert "mse" in repr(idx)

    def test_string_ids(self):
        idx = SnapIndex(dim=DIM, bits=4)
        vecs = _rand(5)
        ids = [f"doc_{i}" for i in range(5)]
        idx.add_batch(ids, vecs)
        results = idx.search(vecs[0], k=3)
        assert all(isinstance(r[0], str) for r in results)


# ─────────────────────────────────────────────── Recall ─────────────────────

class TestRecall:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_recall_at_10(self, bits):
        """Recall@10 should be reasonable across all bit widths."""
        n, k = 500, 10
        vecs = _rand(n)
        idx = SnapIndex(dim=DIM, bits=bits)
        idx.add_batch(list(range(n)), vecs)

        hits = 0
        sample = 50
        for i in range(sample):
            results = idx.search(vecs[i], k=k)
            if i in [r[0] for r in results]:
                hits += 1

        recall = hits / sample
        # Conservative threshold: 2-bit is coarser
        threshold = {2: 0.60, 3: 0.80, 4: 0.85}[bits]
        assert recall >= threshold, f"recall@{k} = {recall:.2f} < {threshold} at {bits}-bit"

    def test_prod_mode_works(self):
        idx = SnapIndex(dim=DIM, bits=4, use_prod=True)
        vecs = _rand(100)
        idx.add_batch(list(range(100)), vecs)
        results = idx.search(vecs[0], k=5)
        assert 0 in [r[0] for r in results]


# ─────────────────────────────────────────────── Delete ─────────────────────

class TestDelete:
    def test_delete_removes_from_results(self):
        idx = SnapIndex(dim=DIM, bits=4)
        vecs = _rand(50)
        idx.add_batch(list(range(50)), vecs)
        assert idx.delete(0) is True
        assert len(idx) == 49
        results = idx.search(vecs[0], k=10)
        assert 0 not in [r[0] for r in results]

    def test_delete_nonexistent(self):
        idx = SnapIndex(dim=DIM, bits=4)
        assert idx.delete(999) is False

    def test_delete_keeps_id_to_pos_valid(self):
        idx = SnapIndex(dim=DIM, bits=4)
        n = 30
        idx.add_batch(list(range(n)), _rand(n))
        idx.delete(10)
        for pos, id_val in enumerate(idx._ids):
            assert idx._id_to_pos[id_val] == pos

    def test_delete_all(self):
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch([1, 2, 3], _rand(3))
        for i in [1, 2, 3]:
            idx.delete(i)
        assert len(idx) == 0
        assert idx.search(_rand(1)[0], k=5) == []


# ─────────────────────────────────────────────── Persistence ────────────────

class TestPersistence:
    def test_save_load_count(self, tmp_path):
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(20)), _rand(20))
        path = tmp_path / "test.snpv"
        idx.save(path)
        idx2 = SnapIndex.load(path)
        assert len(idx2) == 20

    def test_save_load_recall(self, tmp_path):
        vecs = _rand(50)
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(50)), vecs)
        path = tmp_path / "test.snpv"
        idx.save(path)
        idx2 = SnapIndex.load(path)
        results = idx2.search(vecs[7], k=5)
        assert 7 in [r[0] for r in results]

    def test_atomic_no_tmp_leftover(self, tmp_path):
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add(1, _rand(1)[0])
        path = tmp_path / "test.snpv"
        idx.save(path)
        assert not (tmp_path / "test.snpv.tmp").exists()
        assert path.exists()

    def test_save_load_prod_mode(self, tmp_path):
        idx = SnapIndex(dim=DIM, bits=4, use_prod=True)
        vecs = _rand(20)
        idx.add_batch(list(range(20)), vecs)
        path = tmp_path / "test.snpv"
        idx.save(path)
        idx2 = SnapIndex.load(path)
        assert idx2.use_prod is True
        results = idx2.search(vecs[3], k=5)
        assert 3 in [r[0] for r in results]

    def test_legacy_v2_3bit_file_loads_via_compat_decoder(self, tmp_path):
        """Pre-v0.3 files wrote 3-bit indices in byte-aligned form (ipb=2).

        The v0.3 reader must detect version < 3 and decode with the legacy
        path, then re-pack into the new tight RAM layout.
        """
        import struct
        from snapvec._index import _MAGIC

        idx = SnapIndex(dim=128, bits=3)
        vecs = _rand(15, dim=128)
        idx.add_batch(list(range(15)), vecs)
        unpacked = idx._unpack_to_indices()

        # Re-pack using the legacy byte-aligned algorithm (what v0.2.x wrote).
        flat = unpacked.ravel()
        ipb, mask = 2, 7
        packed_legacy = np.zeros(len(flat) // ipb, dtype=np.uint8)
        for i in range(ipb):
            packed_legacy |= (flat[i::ipb] & mask).astype(np.uint8) << (i * 3)

        path = tmp_path / "legacy.snpv"
        with open(path, "wb") as f:
            f.write(_MAGIC)
            f.write(struct.pack("<IIIIII", 2, 128, 3, 0, 15, 0))  # version=2
            f.write(struct.pack("<I", len(packed_legacy)))
            f.write(packed_legacy.tobytes())
            f.write(idx._norms.tobytes())
            for i in range(15):
                enc = str(i).encode("utf-8")
                f.write(struct.pack("<H", len(enc)))
                f.write(enc)

        loaded = SnapIndex.load(path)
        np.testing.assert_array_equal(unpacked, loaded._unpack_to_indices())
        results = loaded.search(vecs[5], k=3)
        assert results[0][0] == 5

    def test_wrong_magic_raises(self, tmp_path):
        path = tmp_path / "bad.snpv"
        path.write_bytes(b"XXXX" + b"\x00" * 20)
        with pytest.raises(ValueError, match="magic"):
            SnapIndex.load(path)


# ─────────────────────────────────────────────── Validation ─────────────────

class TestValidation:
    def test_bits_must_be_valid(self):
        with pytest.raises(ValueError):
            SnapIndex(dim=64, bits=5)

    def test_prod_requires_bits_ge_3(self):
        with pytest.raises(ValueError):
            SnapIndex(dim=64, bits=2, use_prod=True)

    def test_prod_bits_3_ok(self):
        idx = SnapIndex(dim=64, bits=3, use_prod=True)
        assert idx._mse_bits == 2


# ─────────────────────────────────────────────── Compression ────────────────

class TestStats:
    def test_compression_ratio_4bit(self):
        idx = SnapIndex(dim=384, bits=4)
        idx.add_batch(list(range(1000)), _rand(1000, dim=384))
        s = idx.stats()
        # In-memory: indices stored as uint8 (1 byte/coord), padded to next pow2.
        # d=384 → pdim=512; stored = 512 + 4 norm = 516 bytes vs 1536 float32 → ~3x.
        # Bit-packing (0.5 bytes/coord) applies only on disk, not in RAM.
        # On disk the ratio is ~6x; in memory ~3x — both are meaningful.
        assert s["compression_ratio"] > 2.5, s

    def test_stats_fields(self):
        idx = SnapIndex(dim=64, bits=4)
        idx.add_batch([0, 1], _rand(2, dim=64))
        s = idx.stats()
        assert all(k in s for k in [
            "n", "dim", "padded_dim", "bits", "mse_bits",
            "use_prod", "chunk_size", "float32_bytes", "compressed_bytes",
            "qjl_bytes", "cache_bytes", "total_bytes", "compression_ratio",
        ])


# ─────────────────────────────────────────────── Chunked search ─────────────

class TestChunkedSearch:
    def test_chunked_same_results_as_cached(self):
        """chunk_size produces identical top-k results as the float16 cache path."""
        vecs = _rand(200)
        ids = list(range(200))

        idx_cached = SnapIndex(dim=DIM, bits=4)
        idx_cached.add_batch(ids, vecs)

        idx_chunked = SnapIndex(dim=DIM, bits=4, chunk_size=32)
        idx_chunked.add_batch(ids, vecs)

        for i in range(10):
            r_cached = [x[0] for x in idx_cached.search(vecs[i], k=5)]
            r_chunked = [x[0] for x in idx_chunked.search(vecs[i], k=5)]
            assert r_cached == r_chunked, f"mismatch at query {i}"

    def test_chunked_no_cache_allocated(self):
        """Chunked mode must not populate _cache."""
        idx = SnapIndex(dim=DIM, bits=4, chunk_size=10)
        idx.add_batch(list(range(50)), _rand(50))
        idx.search(_rand(1)[0], k=5)
        assert idx._cache is None, "_cache must stay None in chunked mode"

    def test_cached_mode_builds_cache(self):
        """Default mode builds _cache after first search."""
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(50)), _rand(50))
        assert idx._cache is None
        idx.search(_rand(1)[0], k=5)
        assert idx._cache is not None

    def test_chunked_recall(self):
        """Chunked search achieves same recall as cached."""
        n, k = 300, 10
        vecs = _rand(n)
        idx = SnapIndex(dim=DIM, bits=4, chunk_size=50)
        idx.add_batch(list(range(n)), vecs)

        hits = sum(
            i in [r[0] for r in idx.search(vecs[i], k=k)]
            for i in range(30)
        )
        assert hits / 30 >= 0.80


# ─────────────────────────────────────────────── RAM bit-packing ────────────

class TestRamPacking:
    """RAM bit-packing: indices stored at 4 bits/coord (2 per byte) at 4-bit."""

    def test_packed_indices_at_4bit(self):
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(10)), _rand(10))
        assert idx._indices.shape == (10, idx._pdim // 2)
        assert idx._can_pack is True

    def test_packed_indices_at_2bit(self):
        idx = SnapIndex(dim=DIM, bits=2)
        idx.add_batch(list(range(5)), _rand(5))
        assert idx._indices.shape == (5, idx._pdim // 4)

    def test_tight_packed_indices_at_3bit(self):
        """3-bit uses tight packing in RAM: 8 indices → 3 bytes."""
        idx = SnapIndex(dim=DIM, bits=3)
        idx.add_batch(list(range(5)), _rand(5))
        assert idx._can_pack is True
        # Each row: pdim indices × 3 bits / 8 = 3*pdim/8 bytes
        assert idx._indices.shape == (5, (idx._pdim * 3) // 8)

    def test_tight_3bit_roundtrip_correctness(self):
        """Pack → unpack must recover the original indices exactly."""
        idx = SnapIndex(dim=DIM, bits=3)
        idx.add_batch(list(range(20)), _rand(20))
        unpacked = idx._unpack_to_indices()
        repacked = idx._pack_matrix(unpacked)
        np.testing.assert_array_equal(idx._indices, repacked)
        # And disk-level round-trip
        from snapvec._index import _pack, _unpack
        raw = _pack(unpacked, 3)
        recovered = _unpack(raw, 20, idx._pdim, 3)
        np.testing.assert_array_equal(unpacked, recovered)

    def test_packed_search_top1_is_self(self):
        """Packed-index search should still find query as nearest neighbor."""
        vecs = _rand(100)
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(100)), vecs)
        results = idx.search(vecs[3], k=5)
        assert results[0][0] == 3

    def test_packed_save_load_roundtrip(self, tmp_path):
        vecs = _rand(30)
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(30)), vecs)
        path = tmp_path / "packed.snpv"
        idx.save(path)
        idx2 = SnapIndex.load(path)
        assert idx2._can_pack is True
        assert idx2._indices.shape == idx._indices.shape
        np.testing.assert_array_equal(
            idx._unpack_to_indices(), idx2._unpack_to_indices()
        )

    def test_compression_ratio_improved_4bit(self):
        """RAM compression should be ~6x at 4-bit (vs ~3x before packing)."""
        idx = SnapIndex(dim=384, bits=4)
        idx.add_batch(list(range(1000)), _rand(1000, dim=384))
        s = idx.stats()
        assert s["compression_ratio"] > 5.0, s
        assert s["ram_packed"] is True


# ─────────────────────────────────────────────── Normalized mode ────────────

class TestNormalized:
    """normalized=True: skip norm computation for pre-normalized inputs."""

    def test_normalized_flag_in_repr(self):
        idx = SnapIndex(dim=DIM, bits=4, normalized=True)
        assert "normalized" in repr(idx)

    def test_normalized_search_works(self):
        vecs = _rand(50)  # _rand already produces unit vectors
        idx = SnapIndex(dim=DIM, bits=4, normalized=True)
        idx.add_batch(list(range(50)), vecs)
        results = idx.search(vecs[7], k=5)
        assert 7 in [r[0] for r in results]

    def test_normalized_norms_are_one(self):
        idx = SnapIndex(dim=DIM, bits=4, normalized=True)
        idx.add_batch(list(range(10)), _rand(10))
        np.testing.assert_array_equal(idx._norms, np.ones(10, dtype=np.float32))

    def test_normalized_save_load(self, tmp_path):
        vecs = _rand(20)
        idx = SnapIndex(dim=DIM, bits=4, normalized=True)
        idx.add_batch(list(range(20)), vecs)
        path = tmp_path / "norm.snpv"
        idx.save(path)
        idx2 = SnapIndex.load(path)
        assert idx2.normalized is True
        results = idx2.search(vecs[5], k=5)
        assert 5 in [r[0] for r in results]

    def test_normalized_matches_non_normalized_results(self):
        """For unit inputs, normalized=True must give identical top-k."""
        vecs = _rand(100)
        ids = list(range(100))
        a = SnapIndex(dim=DIM, bits=4, normalized=False)
        a.add_batch(ids, vecs)
        b = SnapIndex(dim=DIM, bits=4, normalized=True)
        b.add_batch(ids, vecs)
        for i in range(0, 20, 5):
            ra = [x[0] for x in a.search(vecs[i], k=5)]
            rb = [x[0] for x in b.search(vecs[i], k=5)]
            assert ra == rb


# ─────────────────────────────────────────────── Pre-filtering ───────────────

class TestFilterIds:
    def test_filter_restricts_results(self):
        """Results must only contain ids from filter_ids."""
        vecs = _rand(100)
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(100)), vecs)

        allowed = set(range(0, 50))
        results = idx.search(vecs[0], k=10, filter_ids=allowed)
        assert all(r[0] in allowed for r in results)

    def test_filter_finds_target_in_minority(self):
        """Even with 1 valid id in 200, it must be found."""
        vecs = _rand(200)
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(200)), vecs)

        results = idx.search(vecs[42], k=5, filter_ids={42})
        assert len(results) == 1
        assert results[0][0] == 42

    def test_filter_empty_set_returns_empty(self):
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(10)), _rand(10))
        results = idx.search(_rand(1)[0], k=5, filter_ids=set())
        assert results == []

    def test_filter_unknown_ids_ignored(self):
        """IDs not in the index are silently ignored."""
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch([1, 2, 3], _rand(3))
        results = idx.search(_rand(1)[0], k=5, filter_ids={1, 999, 1000})
        assert all(r[0] in {1} for r in results)

    def test_filter_none_is_full_scan(self):
        """filter_ids=None must behave identically to no filter."""
        vecs = _rand(50)
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(list(range(50)), vecs)
        r1 = idx.search(vecs[0], k=5)
        r2 = idx.search(vecs[0], k=5, filter_ids=None)
        assert r1 == r2

    def test_filter_with_string_ids(self):
        """filter_ids works with string ids."""
        vecs = _rand(20)
        ids = [f"doc_{i}" for i in range(20)]
        idx = SnapIndex(dim=DIM, bits=4)
        idx.add_batch(ids, vecs)
        allowed = {f"doc_{i}" for i in range(10)}
        results = idx.search(vecs[0], k=5, filter_ids=allowed)
        assert all(r[0] in allowed for r in results)

    def test_filter_with_chunk_size(self):
        """filter_ids works correctly in chunked mode."""
        vecs = _rand(100)
        idx = SnapIndex(dim=DIM, bits=4, chunk_size=20)
        idx.add_batch(list(range(100)), vecs)
        allowed = set(range(50, 100))
        results = idx.search(vecs[75], k=5, filter_ids=allowed)
        assert all(r[0] in allowed for r in results)
