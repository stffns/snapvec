"""Save and load an index round-trip.

Every index type has ``.save(path)`` / ``.load(path)``.  Writes are
atomic (write to ``<path>.tmp`` then rename) and include a CRC32 trailer
so transport or disk corruption is caught at load time.

Run with: python examples/save_load.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from snapvec import SnapIndex


def main() -> None:
    rng = np.random.default_rng(0)
    dim, n_corpus = 128, 500

    corpus = rng.standard_normal((n_corpus, dim)).astype(np.float32)

    idx = SnapIndex(dim=dim, bits=4, normalized=True, seed=0)
    idx.add_batch(list(range(n_corpus)), corpus)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.snpv"
        idx.save(path)
        size_kb = path.stat().st_size / 1024
        print(f"wrote {path.name}  ({size_kb:.1f} KB)")

        loaded = SnapIndex.load(path)
        print(f"loaded index: n={len(loaded)}, dim={loaded.dim}, bits={loaded.bits}")

        query = corpus[42]
        hit_before = idx.search(query, k=1)[0]
        hit_after = loaded.search(query, k=1)[0]
        assert hit_before == hit_after, "round-trip mismatch"
        print(f"round-trip top-1 identical: id={hit_after[0]} score={hit_after[1]:+.4f}")


if __name__ == "__main__":
    main()
