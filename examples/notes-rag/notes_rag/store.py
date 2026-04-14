"""Persistent store: snapvec index + JSON sidecar for chunk metadata.

The snapvec index holds (id → embedding) only.  Everything a user would want
to see or filter on (file path, chunk index, tags, folder, raw text) lives in
a plain JSON sidecar keyed by the same id.  Keeps the snapvec file small and
content-agnostic, which is the right split of responsibilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from snapvec import SnapIndex

DEFAULT_STORE_DIR = Path.home() / ".notes-rag"


class Store:
    """Wrap a ``SnapIndex`` plus a chunk-metadata JSON sidecar."""

    def __init__(
        self,
        root: Path = DEFAULT_STORE_DIR,
        dim: int = 768,
        bits: int = 4,
    ) -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.snpv"
        self.meta_path = self.root / "index.meta.json"
        self.dim = dim
        self.bits = bits
        self.idx: SnapIndex | None = None
        self.meta: dict[int, dict[str, Any]] = {}

    # ---------- lifecycle ---------------------------------------------------

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def create(self) -> None:
        """Initialise an empty index in memory (not yet on disk)."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.idx = SnapIndex(dim=self.dim, bits=self.bits)
        self.meta = {}

    def load(self) -> None:
        self.idx = SnapIndex.load(self.index_path)
        self.meta = {
            int(k): v
            for k, v in json.loads(
                self.meta_path.read_text(encoding="utf-8")
            ).items()
        }
        self.dim = self.idx.dim
        self.bits = self.idx.bits

    def save(self) -> None:
        if self.idx is None:
            raise RuntimeError("Store not initialised; call create() or load() first")
        self.root.mkdir(parents=True, exist_ok=True)
        self.idx.save(self.index_path)
        self.meta_path.write_text(
            json.dumps({str(k): v for k, v in self.meta.items()}),
            encoding="utf-8",
        )

    def clear(self) -> None:
        for p in (self.index_path, self.meta_path):
            if p.exists():
                p.unlink()

    # ---------- data --------------------------------------------------------

    def add(
        self,
        embeddings: NDArray[np.float32],
        records: list[dict[str, Any]],
    ) -> None:
        if self.idx is None:
            raise RuntimeError("Store not initialised; call create() or load() first")
        if len(embeddings) != len(records):
            raise ValueError("embeddings and records must have the same length")
        start = len(self.idx)
        ids = list(range(start, start + len(records)))
        self.idx.add_batch(ids, embeddings)
        for i, rec in zip(ids, records):
            self.meta[i] = rec

    def search(
        self,
        query_vec: NDArray[np.float32],
        k: int = 5,
        tag_filter: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return ``[(meta, score), ...]`` by descending cosine similarity.

        When ``tag_filter`` is supplied we resolve it to the subset of ids
        whose metadata carries ANY of the tags, then let snapvec scan only
        that subset.  This is where ``filter_ids`` earns its keep — for a
        "find in #work notes only" query we skip the other 99% of the index.
        """
        if self.idx is None:
            raise RuntimeError("Store not initialised; call create() or load() first")
        if tag_filter:
            allowed = {
                i for i, m in self.meta.items()
                if tag_filter & set(m.get("tags", []))
            }
            if not allowed:
                return []
            hits = self.idx.search(query_vec, k=k, filter_ids=allowed)
        else:
            hits = self.idx.search(query_vec, k=k)
        return [(self.meta[int(i)], float(s)) for i, s in hits]

    # ---------- diagnostics -------------------------------------------------

    def stats(self) -> dict[str, Any]:
        if self.idx is None:
            raise RuntimeError("Store not initialised; call create() or load() first")
        s = dict(self.idx.stats())
        s["docs"] = len({m["path"] for m in self.meta.values()})
        s["chunks"] = len(self.meta)
        disk_bytes = 0
        for p in (self.index_path, self.meta_path):
            try:
                disk_bytes += p.stat().st_size
            except FileNotFoundError:
                pass
        s["disk_bytes"] = disk_bytes
        return s
