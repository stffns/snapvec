import time
import struct
from snapvec._file_format import ChecksumWriter
import io
import zlib
import typing

class ChecksumWriterFast:
    def __init__(self, f: typing.IO[bytes]) -> None:
        self._f = f
        self._crc = 0
        self._finalised = False
        self._buffer = bytearray()

    def write(self, data: typing.Union[bytes, bytearray]) -> int:
        if self._finalised:
            raise RuntimeError(
                "ChecksumWriter.write called after finalise(); the "
                "trailer has already been emitted."
            )
        self._buffer.extend(data)
        if len(self._buffer) >= 65536:
            self.flush()
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._crc = zlib.crc32(self._buffer, self._crc)
            self._f.write(self._buffer)
            self._buffer.clear()

    def finalise(self) -> None:
        if self._finalised:
            return
        self.flush()
        self._f.write(b"CRC2")
        self._f.write(struct.pack("<I", self._crc & 0xFFFFFFFF))
        self._finalised = True

    def __enter__(self) -> "ChecksumWriterFast":
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        if exc_type is None:
            self.finalise()

class MockFile(io.BytesIO):
    def write(self, data):
        return super().write(data)

def run_test(cls):
    f = MockFile()
    start = time.time()
    with cls(f) as cw:
        for _ in range(100000):
            cw.write(b"hello ")
            cw.write(b"world!")
    end = time.time()
    return end - start

t1 = run_test(ChecksumWriter)
t2 = run_test(ChecksumWriterFast)
print(f"Old: {t1:.4f}s")
print(f"New: {t2:.4f}s")
