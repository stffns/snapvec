import time
import numpy as np

np.random.seed(0)
N = 10000
D = 384
K = 256

X = np.random.randn(N, D).astype(np.float32)
C = np.random.randn(K, D).astype(np.float32)

t0 = time.time()
for _ in range(100):
    d2_1 = (X ** 2).sum(1, keepdims=True) - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
t1 = time.time()
print(f"Time original assign_l2: {t1 - t0}")

t0 = time.time()
for _ in range(100):
    d2_2 = np.einsum('ij,ij->i', X, X)[:, None] - 2 * X @ C.T + np.einsum('ij,ij->i', C, C)[None, :]
t1 = time.time()
print(f"Time einsum assign_l2: {t1 - t0}")

print(np.allclose(d2_1, d2_2))
