## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-27 - Fast 1D vector norm in pure NumPy
**Learning:** In NumPy, calculating the L2 norm of small 1D arrays (like single query vectors) using `np.linalg.norm(q)` incurs significant Python-level dispatch and dimension-checking overhead. Using `np.sqrt(np.inner(q, q))` achieves the exact same result but avoids this overhead, yielding a ~1.5x speedup for 1D arrays.
**Action:** Always prefer `np.sqrt(np.inner(q, q))` (or `vdot`) over `np.linalg.norm(q)` when computing the norm of 1D arrays in performance-critical sections of NumPy code.
