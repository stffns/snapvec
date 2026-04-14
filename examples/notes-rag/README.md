# notes-rag

**Local RAG over a folder of markdown notes.**  Pure-Python stack: `snapvec`
for the vector index, [Ollama](https://ollama.com) for embeddings and LLM.
No cloud, no credentials, no training step.  Reference application showing
how snapvec drops into a realistic retrieval workload.

## Why this exists

- **Zero-dependency feel.** `pip install` and you are done.  Works wherever
  Python + NumPy + Ollama work — Mac M-series, Linux, WSL, even a
  Raspberry Pi with enough RAM to run a small model.
- **Concrete snapvec workload.** Shows `SnapIndex.add_batch` / `search` /
  `save` / `load`, plus `filter_ids` for tag-scoped queries ("only look
  in my `#work` notes").  One file each for indexing, storage, and the
  CLI — nothing clever, readable end to end.
- **Honest footprint.** For 10 k notes × 768-dim `nomic-embed-text`, the
  snapvec index is roughly 5 MB on disk and 5 MB RAM idle.  A single
  laptop happily holds indexes for multiple vaults.

## Install

```bash
# From a clone of the snapvec repo:
cd examples/notes-rag
pip install -e .

# Pull the default models via Ollama (one-off, ~2.5 GB):
ollama pull nomic-embed-text        # 768-dim embeddings
ollama pull llama3.2:3b             # lightweight chat model
```

Ollama must be running (`ollama serve`, usually autostarted on install).

## Usage

```bash
# 1. Index a folder of markdown.
notes-rag init ./example_vault

# 2. Ask a question — top-k chunks become context for the LLM.
notes-rag ask "what is the difference between product quantization and turboquant?"

# 3. Just retrieve, skip the LLM answer:
notes-rag search "kubernetes troubleshooting"

# 4. Restrict to notes carrying a given tag:
notes-rag ask "what should I read this week?" --tag reading

# 5. Inspect the index:
notes-rag status
```

First-run output with the sample vault in this directory looks roughly like:

```
$ notes-rag init ./example_vault
indexing /.../example_vault
  embedding model : nomic-embed-text
  bits            : 4  (snapvec)
  store           : ~/.notes-rag
  embedded        : 3/3 chunks

  docs            : 3
  chunks          : 3
  compression     : 5.91x vs float32
  disk footprint  : 4.3 KB

$ notes-rag ask "when does a flat scan beat a graph index?" --tag vector-search
retrieved 1 chunk(s):
  +0.784  /.../example_vault/vector-search-reading.md [chunk 0]

llama3.2:3b answering...
------------------------------------------------------------
For small N (under about 1 million), a flat compressed scan wins on
cold-start and predictable latency, per vector-search-reading.md.
Graph indices like HNSW take over past N > 10 million.
```

## How it works

```
markdown file ──┐
                │   chunker.py            indexer.py      snapvec
                ├── parse_markdown()  ┬──▶ build_index() ──▶ SnapIndex
                │     (frontmatter   │      (batches of
                │      + hashtags)   │       chunks)
                │                    ▼
                └── chunk() ─────────▶ store.py
                    (paragraph-          Store
                     packing to          .add() / .search()
                     ~1500 chars)        meta → JSON sidecar
```

- **`chunker.py`** splits each file on blank lines and greedily packs
  paragraphs into chunks of ~1500 characters.  Frontmatter tags and
  inline `#hashtag` markers become the `tags` field.
- **`indexer.py`** walks a directory, calls an injectable embedder on
  each batch, and pushes the vectors into the store.  Injecting the
  embedder keeps the tests Ollama-free (they use a seeded dummy).
- **`store.py`** wraps a `SnapIndex` plus a JSON sidecar that maps each
  index id to `{path, chunk_idx, text, tags, folder}`.  `search()`
  resolves a `tag_filter` to a set of ids and passes it straight to
  snapvec's `filter_ids`, so tag-scoped queries scan only the tagged
  subset instead of post-filtering.
- **`cli.py`** wires the above into the five subcommands.

## Defaults worth knowing

| Setting            | Default                  | Change with              |
| ------------------ | ------------------------ | ------------------------ |
| Embedding model    | `nomic-embed-text` (768d) | `--embed-model`         |
| Chat model         | `llama3.2:3b`            | `--chat-model`           |
| snapvec bits       | 4  (≈ 95% recall@10)     | `--bits` (2 / 3 / 4)     |
| Store location     | `~/.notes-rag/`          | `--store`                |
| Chunk target chars | 1500 (about 300 tokens)  | `chunker.chunk(target_chars=...)` if you fork |

Switching to `--bits 2` trades roughly 5–10 points of recall@10 for
a further ~2× reduction in disk / RAM footprint.  Worth it when you
want to index a huge note archive and don't need perfect top-1.

## Run the tests

```bash
pip install -e ".[test]"
pytest -q
```

The test suite uses a deterministic dummy embedder so it runs without
Ollama — the Ollama integration itself is exercised only when you use
the CLI against real notes.

## License

MIT, same as snapvec.
