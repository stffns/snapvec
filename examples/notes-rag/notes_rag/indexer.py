"""Walk a directory of ``.md`` files and build a snapvec index.

The embedder is injected as a callable so the indexer itself stays testable
without Ollama running (tests pass a dummy that returns random unit vectors).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .chunker import chunk, parse_markdown
from .store import Store

Embedder = Callable[[list[str]], NDArray[np.float32]]


def build_index(
    root: Path,
    store: Store,
    embed: Embedder,
    batch_size: int = 16,
    progress_cb: Callable[[int, int], None] | None = None,
) -> int:
    """Index every ``*.md`` file under ``root`` into ``store``.

    Returns the total number of chunks indexed.  Embeddings are produced in
    batches to amortise the HTTP cost of calling Ollama (or whatever backend
    the caller wires up).
    """
    root = Path(root).resolve()
    md_files = sorted(p for p in root.rglob("*.md") if p.is_file())
    records: list[dict] = []
    for path in md_files:
        parsed = parse_markdown(path)
        for i, text in enumerate(chunk(parsed["text"])):
            rel_folder = str(path.parent.relative_to(root)) if path.parent != root else ""
            records.append({
                "path": str(path),
                "chunk_idx": i,
                "text": text,
                "tags": parsed["tags"],
                "folder": rel_folder,
            })

    total = len(records)
    if total == 0:
        return 0

    for start in range(0, total, batch_size):
        batch = records[start:start + batch_size]
        vecs = embed([r["text"] for r in batch])
        store.add(vecs, batch)
        if progress_cb is not None:
            progress_cb(min(start + batch_size, total), total)

    return total
