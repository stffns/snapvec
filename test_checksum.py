import os
import struct
import zlib
from pathlib import Path
from types import TracebackType
from typing import IO, Callable
import typing

_TRAILER_MAGIC = b"CRC2"
_TRAILER_SIZE = 8

class ChecksumWriter:
    def __init__(self, f: IO[bytes]) -> None:
        self._f = f
        self._crc = 0
        self._finalised = False
        self._buffer = bytearray()
        self._buffer_size = 65536

    def write(self, data: typing.Union[bytes, bytearray]) -> int:
        if self._finalised:
            raise RuntimeError(
                "ChecksumWriter.write called after finalise(); the "
                "trailer has already been emitted."
            )
        self._buffer.extend(data)
        if len(self._buffer) >= self._buffer_size:
            self._flush()
        return len(data)

    def _flush(self) -> None:
        if self._buffer:
            self._crc = zlib.crc32(self._buffer, self._crc)
            self._f.write(self._buffer)
            self._buffer.clear()

    def finalise(self) -> None:
        if self._finalised:
            return
        self._flush()
        self._f.write(_TRAILER_MAGIC)
        self._f.write(struct.pack("<I", self._crc & 0xFFFFFFFF))
        self._finalised = True

    def __enter__(self) -> "ChecksumWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.finalise()
