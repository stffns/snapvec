"""Colab Pro / A100 embedding script for snapvec benchmarks.

Run this in a single Colab cell to embed FIQA (~57 k) or HotpotQA
(~100 k) with BGE-small-en-v1.5 in under a minute on an A100, save as
``.npy``, and download to your laptop.  The local benches
(``bench_competitive.py``, ``bench_ivfpq_opq.py``) pick this up
automatically when placed at ``experiments/.cache_fiqa_bge_small.npy``.

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
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "fastembed-gpu==0.8.0",   # GPU variant of fastembed (uses CUDA EP)
    "datasets",
    "huggingface_hub",
])


# %% [config]
DATASET = "BeIR/fiqa"            # or "BeIR/hotpotqa", "BeIR/quora"
# Splits to embed.  ``corpus`` is the searchable docs; ``queries`` is
# the held-out questions split (used by the IR-style recall bench
# ``bench_ivf_pq_fiqa_recall.py``).  Comment out either if you only
# need one.
SPLITS = ["corpus", "queries"]
N_DOCS = 60_000                  # cap; will not exceed actual split size
MODEL = "BAAI/bge-small-en-v1.5" # 384-dim
BATCH = 256                      # tune up if GPU memory allows


def _out_path(split: str) -> str:
    return f"cache_{DATASET.split('/')[-1]}_{split}_bge_small.npy"


# %% [embed each requested split]
import time
import numpy as np
from datasets import load_dataset
from fastembed import TextEmbedding

print(f"loading {MODEL} on GPU…", flush=True)
model = TextEmbedding(
    model_name=MODEL,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

written: list[str] = []
for split in SPLITS:
    print(f"\n── split: {split} ──", flush=True)
    print(f"loading {DATASET} / {split}…", flush=True)
    ds = load_dataset(DATASET, split, split=split)
    n = min(N_DOCS, len(ds))
    print(f"  {len(ds)} total rows, taking first {n}", flush=True)

    texts = []
    for row in ds.select(range(n)):
        # corpus has (_id, title, text); queries have (_id, text) only.
        # Skip the ". " separator when title is empty so the queries
        # split does not embed strings starting with a leading
        # punctuation token that is foreign to the model's input space.
        row_d: dict = dict(row)  # type: ignore[arg-type]
        title = (row_d.get("title", "") or "").strip()
        text = (row_d.get("text", "") or "").strip()
        if title and text:
            joined = f"{title}. {text}"
        else:
            joined = title or text
        texts.append(joined[:2000])
    print(f"  prepared {len(texts)} text strings", flush=True)

    print(f"embedding in batches of {BATCH}…", flush=True)
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
    print(f"  embedded {vecs.shape} in {elapsed:.1f}s "
          f"({len(texts) / elapsed:.0f} docs/sec)", flush=True)

    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    out = _out_path(split)
    np.save(out, vecs)
    size_mb = vecs.nbytes / (1024 * 1024)
    print(f"  saved {out}  ({size_mb:.1f} MB float32)", flush=True)
    written.append(out)


# %% [download to laptop]
# In Colab, this triggers a browser download for each file.  Place the
# corpus file at experiments/.cache_fiqa_bge_small.npy and the queries
# file at experiments/.cache_fiqa_queries_bge_small.npy in your snapvec
# checkout for bench_ivf_pq_fiqa_recall.py to pick them up.
try:
    from google.colab import files  # type: ignore[import-untyped]
    for path in written:
        print(f"downloading {path}…", flush=True)
        files.download(path)
except ImportError:
    print(f"not in Colab — pick up {written} from the working directory",
          flush=True)
