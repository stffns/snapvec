## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-23 - Fast chunked batching for file writes
**Learning:** Batching multiple small file writes into a single `bytearray` before calling `f.write()` significantly improves serialization performance (approx. 1.4x speedup) by reducing system call overhead and frequent `zlib.crc32` updates. Implementing a chunked batching strategy (flushing at 64KB) prevents unbounded memory usage, while allowing large incoming data chunks (>= 64KB) to bypass the buffer to prevent unnecessary memory allocations.
**Action:** Use chunked `bytearray` batching when performing many small file writes to reduce I/O and CPU overhead.
