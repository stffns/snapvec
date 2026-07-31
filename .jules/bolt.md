## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.
## 2025-02-27 - Replace squared norm sum with np.einsum
**Learning:** Using `(X ** 2).sum(axis)` creates large intermediate array allocations which bottlenecks performance. Replacing these with `np.einsum('ij,ij->i', X, X)` prevents these allocations, significantly speeding up execution (approx 2-3x faster).
**Action:** Always prefer `np.einsum` for computing squared norms over explicit squaring and summation when operating on numpy arrays in performance-critical areas.
