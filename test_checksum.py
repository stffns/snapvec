import struct
import zlib
from snapvec._file_format import ChecksumWriter
import io

class MockFile(io.BytesIO):
    def write(self, data):
        return super().write(data)

def test_writer():
    f = MockFile()
    with ChecksumWriter(f) as cw:
        cw.write(b"hello ")
        cw.write(b"world!")

    f.seek(0)
    res = f.read()
    print("result len:", len(res))
    assert res[:12] == b"hello world!"

test_writer()
print("Success")
