## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.
## 2025-05-18 - Fast vector norm for 1D arrays in NumPy
**Learning:** In performance-critical paths (like single vector queries), computing the L2 norm of a 1D array via `np.linalg.norm(q)` incurs significant Python-level overhead (input validation, kwarg handling, axis checks). Replacing it with `np.sqrt(np.inner(q, q))` yields a ~1.5x speedup for typical high-dimensional embeddings (e.g., D=1536).
**Action:** Prefer `np.sqrt(np.inner(q, q))` (or `np.vdot` for complex numbers) over `np.linalg.norm(q)` for 1D arrays in hot code paths.
