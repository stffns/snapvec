"""Filtered search: restrict results to a subset of ids.

Every index type accepts ``filter_ids=<set>`` to limit the candidates
returned.  On IVFPQSnapIndex the filter is cluster-aware: probes skip
clusters that contain no matching id.

Run with: python examples/filter_search.py
"""
from __future__ import annotations

import numpy as np

from snapvec import SnapIndex


def main() -> None:
    rng = np.random.default_rng(0)
    dim, n_corpus = 64, 500

    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)
    ids = [f"doc-{i:04d}" for i in range(n_corpus)]

    idx = SnapIndex(dim=dim, bits=4, seed=0)
    idx.add_batch(ids, corpus)

    query = rng.standard_normal(dim).astype(np.float32)

    unrestricted = idx.search(query, k=5)
    restricted_set = {f"doc-{i:04d}" for i in range(100)}
    restricted = idx.search(query, k=5, filter_ids=restricted_set)

    print("unrestricted top-5:")
    for doc_id, score in unrestricted:
        print(f"  {doc_id}  score={score:+.4f}")

    print("\nrestricted to doc-0000..doc-0099:")
    for doc_id, score in restricted:
        print(f"  {doc_id}  score={score:+.4f}")

    assert all(h[0] in restricted_set for h in restricted)
    print("\nall restricted hits are in the filter set: OK")


if __name__ == "__main__":
    main()
