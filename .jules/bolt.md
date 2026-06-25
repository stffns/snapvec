## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2025-02-27 - Fast dictionary updates
**Learning:** Replacing manual dictionary update loops (e.g., `for i, k in enumerate(keys): d[k] = start + i`) with `dict.update(zip(keys, range(start, start + len(keys))))`, and dictionary comprehensions (e.g., `{k: i for i, k in enumerate(keys)}`) with `dict(zip(keys, range(len(keys))))`, provides a significant performance boost (~1.5x for loops, ~15% for comprehensions) by moving iteration to Python's C-level internals.
**Action:** Always prefer `dict.update(zip(...))` and `dict(zip(...))` over `enumerate` loops and dictionary comprehensions for bulk dictionary initialization and updates.
