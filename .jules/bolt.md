## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-07-13 - Batching file writes via bytearray
**Learning:** In `ChecksumWriter`, frequent small file writes combined with continuous `zlib.crc32` updates caused significant overhead. Batching these small chunks into a `bytearray` and only computing the checksum and flushing to disk at a 64KB threshold yielded an approximate 1.4x speedup.
**Action:** Use a bounded `bytearray` batching strategy when dealing with many small file writes that require incremental checksum calculations to reduce system calls and library overhead without causing unbounded memory growth.
