## 2024-04-24 - NumPy single vector norm optimization
**Learning:** `np.linalg.norm` is extremely slow for 1D arrays compared to `np.sqrt(np.inner(q, q))` because it contains a lot of Python-level overhead (dimension checks, kwargs, BLAS dispatching) that don't benefit small 1D vectors.
**Action:** Use `np.sqrt(np.inner(x, x))` instead of `np.linalg.norm(x)` for single vectors.
