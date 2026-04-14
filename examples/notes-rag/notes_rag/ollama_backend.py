"""Ollama-backed embedder and chat helper.

Imported lazily by the CLI so ``notes-rag --help`` and the unit tests don't
require the ``ollama`` package to be installed.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_embedder(model: str):
    """Return a callable ``embed(texts) -> (n, dim) float32`` backed by Ollama."""
    import ollama  # lazy: only needed when actually indexing / querying

    def embed(texts: list[str]) -> NDArray[np.float32]:
        resp = ollama.embed(model=model, input=texts)
        # ollama-python may return .embeddings (List[List[float]]) or dict-like
        raw = getattr(resp, "embeddings", None)
        if raw is None:  # dict access, older versions
            raw = resp["embeddings"]
        return np.asarray(raw, dtype=np.float32)

    return embed


def chat(model: str, prompt: str) -> str:
    import ollama
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    msg = getattr(resp, "message", None)
    if msg is not None:
        return msg.content
    return resp["message"]["content"]  # type: ignore[index]
