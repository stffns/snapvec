"""Shared integrity helpers for the ``.snpv`` / ``.snpq`` / ``.snpr`` /
``.snpi`` file formats.

Design goal: catch silent corruption (bit flip on disk, partial write,
cloud-storage re-encoding) at load time, with no changes to the
existing payload layout.  Legacy files stay forward-compatible.

Mechanism: append a 4-byte little-endian ``zlib.crc32`` of the
everything-up-to-this-point payload at the end of the file.  Each
writer is wrapped in ``ChecksumWriter`` that intercepts ``write``
calls, maintains a running CRC32, and emits the trailer when closed.

The trailer is an 8-byte tail::

    [0..4)  magic b"CRC2"       - separator / version of trailer scheme
    [4..8)  uint32 le           - CRC32 of all preceding bytes

``CRC2`` (not ``CRC1``) leaves room to extend later (e.g. switch to
xxHash3) without breaking old readers — we key off the 4-byte magic
at position ``len(file) - 8``.

Readers opt in via ``verify_checksum(path)``.  Legacy files (no
trailer) return ``None`` from ``trailer_offset`` and skip the check.
That keeps backward compat with every file written before v0.7.
"""
from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import IO


_TRAILER_MAGIC = b"CRC2"
_TRAILER_SIZE = 8  # 4 bytes magic + 4 bytes uint32 CRC


class ChecksumWriter:
    """File-like wrapper that maintains a running CRC32 on every write.

    Used by ``save()`` implementations to build a checksummed file
    without threading the CRC through every ``f.write`` call::

        with open(path, "wb") as raw, ChecksumWriter(raw) as cw:
            cw.write(_MAGIC)
            cw.write(struct.pack(...))
            cw.write(codes_bytes)
            # trailer is appended automatically on context exit

    ``flush`` and ``close`` are forwarded to the underlying file.
    """

    def __init__(self, f: IO[bytes]) -> None:
        self._f = f
        self._crc = 0

    def write(self, data: bytes) -> int:
        self._crc = zlib.crc32(data, self._crc)
        return self._f.write(data)

    def finalise(self) -> None:
        """Write the trailer.  Safe to call multiple times (idempotent)."""
        self._f.write(_TRAILER_MAGIC)
        self._f.write(struct.pack("<I", self._crc & 0xFFFFFFFF))
        self._crc = 0  # guard against double-write

    def __enter__(self) -> "ChecksumWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalise()


def has_trailer(path: str | Path) -> bool:
    """Return True if ``path`` ends with our 8-byte CRC trailer."""
    path = Path(path)
    size = path.stat().st_size
    if size < _TRAILER_SIZE:
        return False
    with open(path, "rb") as f:
        f.seek(size - _TRAILER_SIZE)
        return f.read(4) == _TRAILER_MAGIC


def verify_checksum(path: str | Path) -> None:
    """Check the CRC32 of ``path``'s payload matches the trailer.

    No-op for legacy files without a trailer.  Raises ``ValueError``
    on mismatch with a pointer to which file failed (so users with
    many indices can identify the bad one).

    This is deliberately a separate entrypoint from the format
    readers: tests and tooling can verify without touching the parser,
    and the parser stays simple.
    """
    path = Path(path)
    if not has_trailer(path):
        return
    size = path.stat().st_size
    expected = 0
    with open(path, "rb") as f:
        remaining = size - _TRAILER_SIZE
        # Stream 1 MB at a time so we don't materialise huge files.
        while remaining > 0:
            chunk = f.read(min(1 << 20, remaining))
            if not chunk:
                break
            expected = zlib.crc32(chunk, expected)
            remaining -= len(chunk)
        f.seek(size - _TRAILER_SIZE + 4)
        (stored,) = struct.unpack("<I", f.read(4))
    if (expected & 0xFFFFFFFF) != stored:
        raise ValueError(
            f"CRC32 mismatch in {path}: payload digests to "
            f"{expected & 0xFFFFFFFF:#010x}, trailer says "
            f"{stored:#010x}.  File is corrupted or was modified "
            f"after save()."
        )


def trailer_len(path: str | Path) -> int:
    """Bytes occupied by the trailer, or 0 for legacy files."""
    return _TRAILER_SIZE if has_trailer(path) else 0


def save_with_checksum_atomic(
    path: str | Path,
    writer_fn,
) -> None:
    """Run ``writer_fn(cw)`` against a ``ChecksumWriter`` and write to
    ``path`` atomically (``.tmp`` then ``os.replace``).

    Keeps the save() implementations in each index module tiny — they
    pass a closure that writes their payload, and this helper handles
    the trailer + atomic rename.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as raw:
        with ChecksumWriter(raw) as cw:
            writer_fn(cw)
    os.replace(tmp, path)


__all__ = [
    "ChecksumWriter",
    "has_trailer",
    "verify_checksum",
    "trailer_len",
    "save_with_checksum_atomic",
]
