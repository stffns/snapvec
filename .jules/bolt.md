## 2024-05-18 - Fast row-wise Euclidean norm in pure NumPy
**Learning:** In performance-critical paths, computing the batch norm of a 2D array via `np.linalg.norm(arr, axis=1)` is relatively slow. Using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x speedup on a laptop CPU for typical batch sizes). If `keepdims=True` behavior is needed, appending `[:, np.newaxis]` matches the original shape seamlessly.
**Action:** Always prefer `np.sqrt(np.einsum('ij,ij->i', arr, arr))` over `np.linalg.norm(arr, axis=1)` when computing row-wise vector norms in NumPy to eliminate dispatch overhead and improve execution speed.

## 2024-05-18 - Fast Python dictionary mapping with zip
**Learning:** In code paths that map a list of strings to index positions (like updating an ID-to-row lookup dict), replacing an explicit `for i, v in enumerate(lst): d[v] = start + i` with `d.update(zip(lst, range(start, start + len(lst))))` leverages CPython's highly optimized `zip` and dict C-backend. Benchmarks show this can yield a ~3x speedup. Similar benefits apply to dictionary comprehensions replacing `{v: i for i, v in enumerate(lst)}` with `dict(zip(lst, range(len(lst))))`.
**Action:** Always prefer `d.update(zip(...))` and `dict(zip(...))` over Python-level loops with `enumerate` when building or updating simple 1:1 mapping dictionaries in performance-sensitive contexts.
