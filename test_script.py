import numpy as np
from snapvec import SnapIndex

idx = SnapIndex(dim=384, bits=4, use_prod=True)
vecs = np.random.randn(10, 384).astype(np.float32)
idx.add_batch(list(range(10)), vecs)
results = idx.search(vecs[0], k=3)
print(results)
