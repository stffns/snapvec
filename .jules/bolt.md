## 2026-04-18 - Numpy `linalg.norm` and matrix operations bottlenecks

**Learning:** `np.linalg.norm(..., axis=1)` is significantly slower (~4x) than explicitly computing the sum of squares using `np.einsum('ij,ij->i', arr, arr)` and then applying `np.sqrt()`. Using `.astype(np.float32)` when subtracting two `np.float32` arrays creates redundant memory copies. Computing tensor contractions with `r_scaled @ S.T` avoids array transposition in memory vs `(S @ r_scaled.T).T`.
**Action:** When calculating batch norms in performance-critical code paths, use `np.sqrt(np.einsum('ij,ij->i', ...))` instead of `np.linalg.norm`. When operating with arrays that are already of the target dtype, avoid using `.astype()`. Always map tensor operations to avoid `.T` array creation where possible.
