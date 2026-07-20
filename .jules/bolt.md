## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.
## 2025-02-23 - Optimize squared Euclidean norm calculations with np.einsum
**Learning:** Using `(X ** 2).sum(1)` or `(X * X).sum(1)` in NumPy creates large intermediate array allocations, slowing down performance-critical code paths.
**Action:** Replace row-wise squared Euclidean norm calculations with `np.einsum('ij,ij->i', X, X)` to prevent intermediate array allocations, resulting in a ~3x execution speedup. Append `[:, None]` when `keepdims=True` behavior is required. For 3D arrays, use `np.einsum('ijk,ijk->ij', X, X)`.
