"""Colab Pro / A100 embedding script for snapvec benchmarks.

Run this in a single Colab cell to embed FIQA (~57 k) or HotpotQA
(~100 k) with BGE-small-en-v1.5 in under a minute on an A100, save as
``.npy``, and download to your laptop.  The local benches
(``bench_ivf_pq_1m_baseline.py``, ``_profile_add_batch_1m.py``) pick
this up automatically when placed at
``experiments/.cache_fiqa_bge_small.npy``.

Why this exists: fastembed CPU is too slow / crash-prone for embedding
50 k+ docs on a laptop, and we want a *real* diverse corpus (not the
augmented SciFact one) to validate IVF-PQ recall at large N without
the near-duplicate artefact.

────────────────────────────────────────────────────────────────────────
1. Open https://colab.research.google.com, new notebook.
2. Runtime → Change runtime type → A100 GPU.
3. Paste the entire contents of this file into one cell, run it.
4. When it finishes, it offers a download via the files panel; save it
   as ``experiments/.cache_fiqa_bge_small.npy`` in your snapvec checkout.
────────────────────────────────────────────────────────────────────────
"""
# %% [install]
# Run-once cell.  Re-running is harmless.
import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "fastembed-gpu==0.8.0",   # GPU variant of fastembed (uses CUDA EP)
    "datasets",
    "huggingface_hub",
])


# %% [config]
DATASET = "BeIR/fiqa"            # or "BeIR/hotpotqa", "BeIR/quora"
SPLIT = "corpus"
N_DOCS = 60_000                  # FIQA has ~57 k; will cap automatically
MODEL = "BAAI/bge-small-en-v1.5" # 384-dim
BATCH = 256                      # tune up if GPU memory allows
OUT = f"cache_{DATASET.split('/')[-1]}_bge_small.npy"


# %% [load corpus]
from datasets import load_dataset

print(f"loading {DATASET} / {SPLIT}…", flush=True)
ds = load_dataset(DATASET, SPLIT, split=SPLIT)
n = min(N_DOCS, len(ds))
print(f"  {len(ds)} total docs, taking first {n}", flush=True)

texts = []
for i, row in enumerate(ds.select(range(n))):
    title = row.get("title", "") or ""
    text = row.get("text", "") or ""
    texts.append((title + ". " + text).strip()[:2000])
print(f"  prepared {len(texts)} text strings", flush=True)


# %% [embed on GPU]
import time
import numpy as np
from fastembed import TextEmbedding

print(f"loading {MODEL} on GPU…", flush=True)
model = TextEmbedding(
    model_name=MODEL,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

print(f"embedding {len(texts)} docs in batches of {BATCH}…", flush=True)
t0 = time.perf_counter()
chunks = []
done = 0
for start in range(0, len(texts), BATCH):
    sub = texts[start : start + BATCH]
    chunks.append(np.array(list(model.embed(sub, batch_size=BATCH)),
                           dtype=np.float32))
    done += len(sub)
    if done % (BATCH * 10) == 0:
        rate = done / (time.perf_counter() - t0)
        print(f"    {done}/{len(texts)}  ({rate:.0f} docs/sec)",
              flush=True)
vecs = np.concatenate(chunks, axis=0)
elapsed = time.perf_counter() - t0
print(f"  done: {vecs.shape} in {elapsed:.1f}s "
      f"({len(texts) / elapsed:.0f} docs/sec)", flush=True)


# %% [normalize + save]
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
np.save(OUT, vecs)
size_mb = vecs.nbytes / (1024 * 1024)
print(f"saved {OUT}  ({size_mb:.1f} MB float32)", flush=True)


# %% [download to laptop]
# In Colab, the next two lines pop up the file in the side panel and
# trigger a browser download.  Outside Colab they are no-ops.
try:
    from google.colab import files  # type: ignore[import-untyped]
    print("downloading via browser…", flush=True)
    files.download(OUT)
except ImportError:
    print(f"not in Colab — pick up {OUT} from the working directory",
          flush=True)
