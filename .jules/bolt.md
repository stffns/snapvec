## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Replacing sum of squares with einsum in hot loops
**Learning:** In performance-critical NumPy operations, replacing row-wise squared Euclidean norm calculations like `(X ** 2).sum(axis=1)` or `(X * X).sum(axis=1)` with `np.einsum('ij,ij->i', X, X)` prevents large intermediate array allocations, resulting in a ~3-5x execution speedup. Use `[:, None]` to emulate `keepdims=True` behavior. This also applies to 3D arrays using `np.einsum('ijk,ijk->ij', X, X)`.
**Action:** Always prefer `np.einsum` for computing batched squared norms in hot paths over explicit squaring and summation to avoid overhead.
