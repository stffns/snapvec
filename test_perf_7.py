import time
import numpy as np

np.random.seed(0)

# simulate _kmeans.py line 117
K = 256
D = 384
coarse = np.random.randn(K, D).astype(np.float32)
q = np.random.randn(D).astype(np.float32)

t0 = time.time()
for _ in range(1000):
    val1 = np.float32(2.0) * (coarse @ q) - (coarse ** 2).sum(1)
t1 = time.time()
print(f"Time original sum(1) [K={K}, D={D}]: {t1 - t0}")

t0 = time.time()
for _ in range(1000):
    val2 = np.float32(2.0) * (coarse @ q) - np.einsum('ij,ij->i', coarse, coarse)
t1 = time.time()
print(f"Time einsum sum(1) [K={K}, D={D}]: {t1 - t0}")

print(np.allclose(val1, val2))
