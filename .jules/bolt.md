## 2025-03-01 - [Fast Row-wise Norms]
**Learning:** For calculating norms across rows of a 2D NumPy array in performance-critical paths, computing the square root of the dot product using `np.sqrt(np.einsum('ij,ij->i', arr, arr))` is significantly faster (~4x) than using `np.linalg.norm(arr, axis=1)`.
**Action:** When computing 2D row norms for batches in NumPy, prefer `np.einsum` over `np.linalg.norm` for performance.
