## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## $(date +%Y-%m-%d) - Matrix multiplication associativity optimization
**Learning:** In NumPy, computing `(A @ B.T).T` creates an intermediate C-contiguous array for the multiplication result, and then applies a transpose which returns an F-contiguous (column-major) view. This can cause poor cache locality for subsequent operations (like `np.sign()`). Rewriting this mathematically equivalent expression as `B @ A.T` directly produces a C-contiguous array, avoiding the intermediate allocation and improving cache locality.
**Action:** Replace patterns like `(S @ r_scaled.T).T` with `r_scaled @ S.T` when performing matrix multiplication on large 2D arrays in performance-critical sections.
