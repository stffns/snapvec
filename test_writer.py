import tempfile
from pathlib import Path
import zlib
import struct
import typing
import os
from snapvec._file_format import ChecksumWriter

print("Testing ChecksumWriter changes...")
