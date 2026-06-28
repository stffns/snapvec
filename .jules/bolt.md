## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.
## 2024-06-28 - Fast serialization with batched writes
**Learning:** Batching multiple small file writes into a single `bytearray` before calling `f.write()` significantly improves serialization performance (approx. 2.4x speedup) in `save` methods by reducing system call overhead and frequent `zlib.crc32` updates in the `ChecksumWriter` loop.
**Action:** Always accumulate many small sequential items (like string lengths and encoded bytes) into a single `bytearray` buffer before writing to a checksum-wrapped file object.
