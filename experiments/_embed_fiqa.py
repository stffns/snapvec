"""One-off: embed FIQA corpus (20 k docs) with BGE-small-en-v1.5.

Result cached to ``experiments/.cache_fiqa_bge_small_20k.npy``.  Only
runs the full embedding pass if the cache is missing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from datasets import load_dataset  # type: ignore[import-untyped]
from fastembed import TextEmbedding  # type: ignore[import-untyped]

CACHE = Path("experiments/.cache_fiqa_bge_small.npy")
N_DOCS = 10_000


def main() -> None:
    if CACHE.exists():
        arr = np.load(CACHE)
        print(f"cache exists: {arr.shape}")
        return
    print(f"loading FIQA corpus, target {N_DOCS} docs…")
    ds = load_dataset("BeIR/fiqa", "corpus", split="corpus")
    n = min(N_DOCS, len(ds))
    texts = []
    for i, row in enumerate(ds.select(range(n))):
        t = (row["title"] + ". " + row["text"]).strip()
        texts.append(t[:2000])
    print(f"  loaded {len(texts)} texts; embedding with BGE-small-en-v1.5 "
          f"(this takes a few minutes)", flush=True)
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    batch = 1000
    chunks = []
    for start in range(0, len(texts), batch):
        sub = texts[start:start + batch]
        chunks.append(np.array(list(model.embed(sub)), dtype=np.float32))
        print(f"    embedded {start + len(sub)}/{len(texts)}", flush=True)
    vecs = np.concatenate(chunks, axis=0)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    CACHE.parent.mkdir(exist_ok=True)
    np.save(CACHE, vecs)
    print(f"  cached {vecs.shape} → {CACHE}")


if __name__ == "__main__":
    main()
