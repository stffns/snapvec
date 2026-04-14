"""End-to-end tests with a deterministic dummy embedder — no Ollama required."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from notes_rag.indexer import build_index
from notes_rag.store import Store


def make_dummy_embedder(dim: int = 64, seed: int = 0):
    """Deterministic embedder keyed by the input text hash — no external deps."""
    def embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(hash(t) & 0xFFFFFFFF ^ seed)
            v = rng.standard_normal(dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out
    return embed


def seed_vault(root: Path) -> None:
    (root / "python.md").write_text(
        "---\ntags: [python, tips]\n---\n"
        "use list comprehensions when the expression fits on one line."
    )
    (root / "deep" / "k8s.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "deep" / "k8s.md").write_text(
        "---\ntags: [kubernetes, infra]\n---\n"
        "kubectl get pods shows running pods across the namespace."
    )
    (root / "notes.md").write_text(
        "no frontmatter here.  "
        "just raw text with an inline #ref tag for good measure."
    )


class TestStoreRoundtrip:
    def test_save_load_preserves_metadata(self, tmp_path: Path):
        store = Store(root=tmp_path / "store", dim=32, bits=4)
        store.create()
        vecs = np.random.default_rng(1).standard_normal((3, 32), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        store.add(vecs, [
            {"path": "a.md", "chunk_idx": 0, "text": "x", "tags": ["t1"], "folder": ""},
            {"path": "a.md", "chunk_idx": 1, "text": "y", "tags": ["t1"], "folder": ""},
            {"path": "b.md", "chunk_idx": 0, "text": "z", "tags": [], "folder": ""},
        ])
        store.save()

        reloaded = Store(root=tmp_path / "store")
        reloaded.load()
        assert len(reloaded.meta) == 3
        assert reloaded.meta[0]["path"] == "a.md"
        assert reloaded.idx is not None
        assert len(reloaded.idx) == 3


class TestBuildAndSearch:
    def test_end_to_end_dummy_embedder(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        seed_vault(vault)
        store = Store(root=tmp_path / "store", dim=64, bits=4)
        store.create()
        n = build_index(vault, store, make_dummy_embedder(dim=64))
        assert n == 3                              # one chunk per file (tiny docs)
        assert store.idx is not None and len(store.idx) == 3

    def test_filter_ids_respected_via_tag(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        seed_vault(vault)
        store = Store(root=tmp_path / "store", dim=64, bits=4)
        store.create()
        embedder = make_dummy_embedder(dim=64)
        build_index(vault, store, embedder)

        # Query with a tag filter — must return only the tagged subset
        q = embedder(["query"])[0]
        hits = store.search(q, k=5, tag_filter={"kubernetes"})
        assert len(hits) == 1
        assert hits[0][0]["path"].endswith("k8s.md")

    def test_save_load_then_search(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        seed_vault(vault)
        store = Store(root=tmp_path / "store", dim=64, bits=4)
        store.create()
        embedder = make_dummy_embedder(dim=64)
        build_index(vault, store, embedder)
        store.save()

        q = embedder(["query"])[0]
        hits_before = store.search(q, k=3)

        fresh = Store(root=tmp_path / "store")
        fresh.load()
        hits_after = fresh.search(q, k=3)
        assert [m["path"] for m, _ in hits_before] == [m["path"] for m, _ in hits_after]

    def test_stats_reports_reasonable_values(self, tmp_path: Path):
        vault = tmp_path / "vault"
        vault.mkdir()
        seed_vault(vault)
        store = Store(root=tmp_path / "store", dim=64, bits=4)
        store.create()
        build_index(vault, store, make_dummy_embedder(dim=64))
        store.save()
        s = store.stats()
        assert s["docs"] == 3
        assert s["chunks"] == 3
        assert s["compression_ratio"] > 1.0
        assert s["disk_bytes"] > 0
