import time
import numpy as np
from snapvec import SnapIndex

def bench():
    index = SnapIndex(dim=128, bits=1)
    # Add a lot of vectors
    vectors = np.random.randn(100000, 128).astype(np.float32)
    index.add_batch(vectors, ids=[f"id_{i}" for i in range(100000)])

    start = time.time()
    index.save("test.snpv")
    end = time.time()

    print(f"Time to save: {end - start:.4f}s")

bench()
