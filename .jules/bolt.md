## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast row-wise squared Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the squared batch norm of a 2D array via `(X ** 2).sum(axis=1)` or `(X * X).sum(axis=1)` allocates an intermediate array of the same shape as X before summing. Using `np.einsum('ij,ij->i', X, X)` avoids this allocation entirely by fusing the multiply and add, yielding a ~3-5x speedup for typical array sizes.
**Action:** Always prefer `np.einsum('ij,ij->i', X, X)` over `(X ** 2).sum(axis=1)` when computing row-wise squared vector norms in NumPy to eliminate memory overhead and improve cache locality. Use `[:, None]` when `keepdims=True` behavior is required.
