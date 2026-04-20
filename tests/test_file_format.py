"""Tests for the shared CRC32 file-format helpers and their
integration with each index type's save/load."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from snapvec import (
    IVFPQSnapIndex,
    PQSnapIndex,
    ResidualSnapIndex,
    SnapIndex,
)
from snapvec._file_format import (
    _TRAILER_MAGIC,
    _TRAILER_SIZE,
    has_trailer,
    verify_checksum,
)


def _unit_gaussian(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ──────────────────────────────────────────────────────────────────── #
# Helper-level tests                                                   #
# ──────────────────────────────────────────────────────────────────── #

def test_trailer_magic_is_four_bytes() -> None:
    assert len(_TRAILER_MAGIC) == 4
    assert _TRAILER_SIZE == 8


def test_finalise_is_idempotent(tmp_path: Path) -> None:
    """Second call to ``finalise()`` must be a no-op rather than
    appending a second (corrupting) trailer.  Without the idempotency
    guard this is the exact class of bug that would produce a file
    whose CRC digests over its own trailer."""
    import io

    from snapvec._file_format import ChecksumWriter

    buf = io.BytesIO()
    cw = ChecksumWriter(buf)
    cw.write(b"hello world")
    cw.finalise()
    size_after_first = len(buf.getvalue())
    cw.finalise()
    size_after_second = len(buf.getvalue())
    assert size_after_first == size_after_second, (
        "second finalise() appended extra bytes — trailer not idempotent"
    )


def test_write_after_finalise_raises() -> None:
    """Writing to a ChecksumWriter after finalise would silently
    invalidate the trailer (the CRC no longer covers the appended
    bytes).  Guard against that by raising."""
    import io

    from snapvec._file_format import ChecksumWriter

    buf = io.BytesIO()
    cw = ChecksumWriter(buf)
    cw.write(b"payload")
    cw.finalise()
    with pytest.raises(RuntimeError, match="after finalise"):
        cw.write(b"oops")


def test_verify_checksum_is_noop_on_legacy_files(tmp_path: Path) -> None:
    """Files without our trailer must load fine — legacy v0.4-shipped
    .snpv files have no trailer and must keep working."""
    p = tmp_path / "legacy.bin"
    p.write_bytes(b"arbitrary payload with no trailer")
    # has_trailer should return False, and verify_checksum should
    # silently accept.
    assert not has_trailer(p)
    verify_checksum(p)


def test_verify_checksum_raises_on_mutated_body(tmp_path: Path) -> None:
    """Flip a bit in the saved body and prove load-time detection
    triggers.  Uses SnapIndex as the smoke-test vehicle — same file-
    format helpers are shared across all four index types."""
    corpus = _unit_gaussian(50, 32, seed=0)
    idx = SnapIndex(dim=32, bits=4, normalized=True, seed=0)
    idx.add_batch(list(range(50)), corpus)
    path = tmp_path / "corrupt.snpv"
    idx.save(path)
    assert has_trailer(path)

    buf = bytearray(path.read_bytes())
    # Flip a bit in the middle of the body (not in the trailer).
    buf[len(buf) // 2] ^= 0x01
    path.write_bytes(bytes(buf))

    with pytest.raises(ValueError, match="CRC32 mismatch"):
        SnapIndex.load(path)


def test_truncated_trailer_falls_back_to_legacy_mode(tmp_path: Path) -> None:
    """Chop off the entire 8-byte trailer from a freshly-saved file.
    With the magic gone, ``has_trailer`` flips to False and
    ``verify_checksum`` becomes a no-op — which is the correct
    conservative contract: we can't verify something that isn't
    there.

    Also prove that the downstream loader treats the file as a
    "legacy no-trailer" file — i.e. it still reads successfully.
    That's important because a partial save or a mid-transfer
    truncation shouldn't spuriously fail load() with a CRC error
    when the actual message is "trailer is gone entirely".  A
    missing trailer is a *weaker* guarantee than a mismatched CRC,
    but it's a well-defined weaker guarantee.
    """
    corpus = _unit_gaussian(30, 32, seed=0)
    idx = SnapIndex(dim=32, bits=4, normalized=True, seed=0)
    idx.add_batch(list(range(30)), corpus)
    path = tmp_path / "truncated.snpv"
    idx.save(path)
    original_bytes = path.read_bytes()

    # Remove the full 8-byte trailer (_TRAILER_SIZE).
    path.write_bytes(original_bytes[:-_TRAILER_SIZE])
    assert not has_trailer(path)

    # verify_checksum must silently pass.
    verify_checksum(path)

    # And SnapIndex.load must still read the file successfully and
    # produce a usable index — this is the no-regression contract
    # for legacy v0.4-shipped files.
    reloaded = SnapIndex.load(path)
    assert len(reloaded) == len(idx)
    # Same search result as the original in-memory index.
    q = corpus[3]
    assert [h[0] for h in idx.search(q, k=3)] == [
        h[0] for h in reloaded.search(q, k=3)
    ]


# ──────────────────────────────────────────────────────────────────── #
# Per-index-type round trips                                           #
# ──────────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize("index_cls, ctor_kwargs, suffix", [
    (SnapIndex,          dict(dim=32, bits=4, normalized=True),           ".snpv"),
    (ResidualSnapIndex,  dict(dim=32, b1=3, b2=3, normalized=True),       ".snpr"),
])
def test_trailing_crc_roundtrip_trainingfree(
    index_cls, ctor_kwargs, suffix, tmp_path,
) -> None:
    corpus = _unit_gaussian(60, 32, seed=1)
    idx = index_cls(**ctor_kwargs)
    idx.add_batch(list(range(60)), corpus)
    path = tmp_path / f"roundtrip{suffix}"
    idx.save(path)
    assert has_trailer(path)
    # Integrity check passes on a freshly-saved file.
    verify_checksum(path)
    reloaded = index_cls.load(path)
    q = corpus[5]
    assert [h[0] for h in idx.search(q, k=3)] == [
        h[0] for h in reloaded.search(q, k=3)
    ]


def test_trailing_crc_roundtrip_pq(tmp_path: Path) -> None:
    corpus = _unit_gaussian(80, 32, seed=2)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(80)), corpus)
    path = tmp_path / "roundtrip.snpq"
    idx.save(path)
    assert has_trailer(path)
    verify_checksum(path)
    reloaded = PQSnapIndex.load(path)
    q = corpus[3]
    assert [h[0] for h in idx.search(q, k=3)] == [
        h[0] for h in reloaded.search(q, k=3)
    ]


def test_trailing_crc_roundtrip_ivfpq(tmp_path: Path) -> None:
    corpus = _unit_gaussian(200, 32, seed=3)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(200)), corpus)
    path = tmp_path / "roundtrip.snpi"
    idx.save(path)
    assert has_trailer(path)
    verify_checksum(path)
    reloaded = IVFPQSnapIndex.load(path)
    q = corpus[7]
    assert [h[0] for h in idx.search(q, k=3, nprobe=4)] == [
        h[0] for h in reloaded.search(q, k=3, nprobe=4)
    ]


@pytest.mark.parametrize("builder, suffix, loader", [
    (
        lambda: _build_snap(),
        ".snpv",
        lambda p: SnapIndex.load(p),
    ),
    (
        lambda: _build_residual(),
        ".snpr",
        lambda p: ResidualSnapIndex.load(p),
    ),
    (
        lambda: _build_pq(),
        ".snpq",
        lambda p: PQSnapIndex.load(p),
    ),
    (
        lambda: _build_ivfpq(),
        ".snpi",
        lambda p: IVFPQSnapIndex.load(p),
    ),
])
def test_body_corruption_caught_at_load(
    builder, suffix, loader, tmp_path: Path,
) -> None:
    """Flip one bit in every index type's body, confirm each
    load() path raises.  This is the real production-integrity test:
    every format must have working corruption detection."""
    idx = builder()
    path = tmp_path / f"corrupt{suffix}"
    idx.save(path)
    buf = bytearray(path.read_bytes())
    # Middle of the payload — not in trailer (last 8 bytes).
    buf[len(buf) // 2] ^= 0x10
    path.write_bytes(bytes(buf))
    with pytest.raises(ValueError, match="CRC32 mismatch"):
        loader(path)


# Builders kept at bottom so the parametrize table above stays readable.
def _build_snap() -> SnapIndex:
    idx = SnapIndex(dim=32, bits=4, normalized=True, seed=0)
    idx.add_batch(list(range(40)), _unit_gaussian(40, 32, seed=4))
    return idx


def _build_residual() -> ResidualSnapIndex:
    idx = ResidualSnapIndex(dim=32, b1=3, b2=3, normalized=True, seed=0)
    idx.add_batch(list(range(40)), _unit_gaussian(40, 32, seed=5))
    return idx


def _build_pq() -> PQSnapIndex:
    corpus = _unit_gaussian(80, 32, seed=6)
    idx = PQSnapIndex(dim=32, M=8, K=16, normalized=True, seed=0)
    idx.fit(corpus)
    idx.add_batch(list(range(80)), corpus)
    return idx


def _build_ivfpq() -> IVFPQSnapIndex:
    corpus = _unit_gaussian(200, 32, seed=7)
    idx = IVFPQSnapIndex(
        dim=32, nlist=4, M=8, K=16, normalized=True, seed=0,
    )
    idx.fit(corpus)
    idx.add_batch(list(range(200)), corpus)
    return idx


# ──────────────────────────────────────────────────────────────────── #
# Forward-compatibility: future-version files must raise with a        #
# clear upgrade hint so the user knows what to do.                     #
# ──────────────────────────────────────────────────────────────────── #


def _patch_version(path: Path, new_version: int) -> None:
    """Overwrite the 4-byte little-endian version field that lives
    immediately after the 4-byte magic in every snapvec index file."""
    import struct
    import zlib

    buf = bytearray(path.read_bytes())
    buf[4:8] = new_version.to_bytes(4, "little")
    # Re-compute the CRC32 trailer so we don't fail on corruption
    # before the version check fires.  Use the module constants so
    # this helper tracks the production trailer layout automatically.
    magic_size = len(_TRAILER_MAGIC)
    payload = bytes(buf[:-_TRAILER_SIZE])
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    buf[-_TRAILER_SIZE : -(_TRAILER_SIZE - magic_size)] = _TRAILER_MAGIC
    buf[-(_TRAILER_SIZE - magic_size) :] = struct.pack("<I", crc)
    path.write_bytes(bytes(buf))


@pytest.mark.parametrize("builder, suffix, loader", [
    (_build_snap,     ".snpv", lambda p: SnapIndex.load(p)),
    (_build_residual, ".snpr", lambda p: ResidualSnapIndex.load(p)),
    (_build_pq,       ".snpq", lambda p: PQSnapIndex.load(p)),
    (_build_ivfpq,    ".snpi", lambda p: IVFPQSnapIndex.load(p)),
])
def test_future_version_raises_with_upgrade_hint(
    builder, suffix, loader, tmp_path: Path,
) -> None:
    """A file whose version byte is far in the future must raise with
    a message that points the user at `pip install -U snapvec`.  The
    exact error wording is free, but the 'upgrade' hint has to be
    there so end users know what to do."""
    idx = builder()
    path = tmp_path / f"future{suffix}"
    idx.save(path)
    _patch_version(path, new_version=999)
    with pytest.raises(ValueError, match="upgrade"):
        loader(path)
