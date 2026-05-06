import numpy as np
import timeit

def bench():
    q = np.random.randn(1024).astype(np.float32)

    def using_linalg():
        return np.linalg.norm(q)

    def using_inner():
        return np.sqrt(np.inner(q, q))

    def using_vdot():
        return np.sqrt(np.vdot(q, q))

    t_linalg = timeit.timeit(using_linalg, number=10000)
    t_inner = timeit.timeit(using_inner, number=10000)
    t_vdot = timeit.timeit(using_vdot, number=10000)

    print(f"np.linalg.norm: {t_linalg:.6f}s")
    print(f"np.sqrt(np.inner): {t_inner:.6f}s")
    print(f"np.sqrt(np.vdot): {t_vdot:.6f}s")
    print(f"Speedup inner: {t_linalg/t_inner:.2f}x")
    print(f"Speedup vdot: {t_linalg/t_vdot:.2f}x")

bench()
