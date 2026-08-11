import time
import numpy as np

np.random.seed(0)

# Simulate ivfpq cnorms
K = 256
D = 384
coarse = np.random.randn(K, D).astype(np.float32)

t0 = time.time()
for _ in range(1000):
    cnorms1 = (coarse * coarse).sum(1)
t1 = time.time()
print(f"Time original sum(1) [K={K}, D={D}]: {t1 - t0}")

t0 = time.time()
for _ in range(1000):
    cnorms2 = np.einsum('ij,ij->i', coarse, coarse)
t1 = time.time()
print(f"Time einsum sum(1) [K={K}, D={D}]: {t1 - t0}")

print(np.allclose(cnorms1, cnorms2))
