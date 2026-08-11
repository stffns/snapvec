import time
import numpy as np

np.random.seed(0)

# Simulate pq Xj
N = 10000
d_sub = 24
Xj = np.random.randn(N, d_sub).astype(np.float32)

t0 = time.time()
for _ in range(1000):
    norms1 = (Xj ** 2).sum(1, keepdims=True)
t1 = time.time()
print(f"Time original sum(1) [N={N}, d_sub={d_sub}]: {t1 - t0}")

t0 = time.time()
for _ in range(1000):
    norms2 = np.einsum('ij,ij->i', Xj, Xj)[:, None]
t1 = time.time()
print(f"Time einsum sum(1) [N={N}, d_sub={d_sub}]: {t1 - t0}")

print(np.allclose(norms1, norms2))
