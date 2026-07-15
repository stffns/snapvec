## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-23 - Batch file writes in Python
**Learning:** Batching multiple small file writes into a single `bytearray` before writing to disk and updating checksums (e.g. `zlib.crc32`) reduces overhead significantly (~1.4x speedup for saving models with many strings), but care must be taken to flush the buffer and skip batching for large blocks to avoid unbounded memory allocation.
**Action:** Implement chunked batching via `bytearray` in high-volume, small-payload write operations to minimize syscalls and iterative CRC updates.
