import time
import tempfile
import struct
import zlib
from pathlib import Path

class ChecksumWriter:
    def __init__(self, f):
        self._f = f
        self._crc = 0
        self._finalised = False
    def write(self, data: bytes) -> int:
        self._crc = zlib.crc32(data, self._crc)
        return self._f.write(data)

def run():
    ids = [f"id_{i}" for i in range(1_000_000)]

    with tempfile.TemporaryFile() as f:
        cw = ChecksumWriter(f)
        start = time.time()
        for id_val in ids:
            enc = str(id_val).encode("utf-8")
            cw.write(struct.pack("<H", len(enc)))
            cw.write(enc)
        print("Unbatched:", time.time() - start)

    with tempfile.TemporaryFile() as f:
        cw = ChecksumWriter(f)
        start = time.time()
        buf = bytearray()
        for id_val in ids:
            enc = str(id_val).encode("utf-8")
            buf.extend(struct.pack("<H", len(enc)))
            buf.extend(enc)
        cw.write(buf)
        print("Batched bytearray:", time.time() - start)

run()
