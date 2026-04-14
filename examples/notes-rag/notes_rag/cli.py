"""notes-rag — CLI entry point.

Commands:
    notes-rag init <path>           build an index from a folder of .md files
    notes-rag ask "question"        retrieve top-k chunks and feed to an LLM
    notes-rag search "query"        just show top-k chunks (no LLM, no Ollama chat)
    notes-rag status                print index size, compression, disk usage
    notes-rag clear                 remove the saved index

The CLI uses ``argparse`` only, so it still runs without Ollama installed
for ``--help``, ``status``, and ``clear``.  Commands that need embeddings
import the Ollama backend lazily.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .indexer import build_index
from .store import DEFAULT_STORE_DIR, Store

DEFAULT_EMBED_MODEL = "nomic-embed-text"   # 768-dim, widely available on Ollama
DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_DIM = 768                          # matches nomic-embed-text output


# ────────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────────

def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    store = Store(root=Path(args.store), dim=args.dim, bits=args.bits)
    if store.exists() and not args.force:
        print(f"index already exists at {store.root}.  use --force to rebuild.",
              file=sys.stderr)
        sys.exit(1)
    store.create()

    from .ollama_backend import make_embedder
    embed = make_embedder(args.embed_model)

    print(f"indexing {root}")
    print(f"  embedding model : {args.embed_model}")
    print(f"  bits            : {args.bits}  (snapvec)")
    print(f"  store           : {store.root}")

    def progress(done: int, total: int) -> None:
        print(f"\r  embedded        : {done}/{total} chunks", end="", flush=True)

    n = build_index(root, store, embed, progress_cb=progress)
    print()  # newline after progress

    if n == 0:
        print(f"no .md files found under {root}", file=sys.stderr)
        sys.exit(1)

    store.save()
    s = store.stats()
    print(f"\n  docs            : {s['docs']}")
    print(f"  chunks          : {s['chunks']}")
    print(f"  compression     : {s['compression_ratio']}x vs float32")
    print(f"  disk footprint  : {s['disk_bytes'] / 1024:.1f} KB")
    print(f"\nindex ready.  try: notes-rag ask \"your question\"")


def cmd_ask(args: argparse.Namespace) -> None:
    store = _load_or_exit(args.store)
    from .ollama_backend import chat, make_embedder
    embed = make_embedder(args.embed_model)

    q_vec = embed([args.query])[0]
    tag_filter = set(args.tag) if args.tag else None
    results = store.search(q_vec, k=args.k, tag_filter=tag_filter)
    if not results:
        print("no matches", file=sys.stderr)
        sys.exit(1)

    print(f"\nretrieved {len(results)} chunk(s):")
    for meta, score in results:
        print(f"  {score:+.3f}  {meta['path']} [chunk {meta['chunk_idx']}]")

    context = "\n\n---\n\n".join(
        f"## {Path(m['path']).name} (chunk {m['chunk_idx']})\n\n{m['text']}"
        for m, _ in results
    )
    prompt = (
        "Answer the question using only the notes below. "
        "Cite filenames in your answer. "
        "If the notes do not contain the answer, say so.\n\n"
        f"NOTES:\n{context}\n\nQUESTION: {args.query}\n\nANSWER:"
    )
    print(f"\n{args.chat_model} answering...\n{'-' * 60}")
    print(chat(args.chat_model, prompt))


def cmd_search(args: argparse.Namespace) -> None:
    store = _load_or_exit(args.store)
    from .ollama_backend import make_embedder
    embed = make_embedder(args.embed_model)

    q_vec = embed([args.query])[0]
    tag_filter = set(args.tag) if args.tag else None
    results = store.search(q_vec, k=args.k, tag_filter=tag_filter)
    for meta, score in results:
        tags = ",".join(meta.get("tags", [])) or "-"
        print(f"\n  score  : {score:+.3f}")
        print(f"  path   : {meta['path']} [chunk {meta['chunk_idx']}]")
        print(f"  tags   : {tags}")
        snippet = " ".join(meta["text"].split())[:220]
        print(f"  text   : {snippet}{'...' if len(snippet) == 220 else ''}")


def cmd_status(args: argparse.Namespace) -> None:
    store = Store(root=Path(args.store))
    if not store.exists():
        print(f"no index at {store.root}")
        return
    store.load()
    s = store.stats()
    print(f"index          : {store.root}")
    print(f"  docs         : {s['docs']}")
    print(f"  chunks       : {s['chunks']}")
    print(f"  dim          : {s['dim']}  (padded {s['padded_dim']})")
    print(f"  bits         : {s['bits']}  (snapvec, tight-packed)")
    print(f"  RAM (idle)   : {s['compressed_bytes'] / 1024:.1f} KB")
    print(f"  disk         : {s['disk_bytes'] / 1024:.1f} KB")
    print(f"  compression  : {s['compression_ratio']}x vs float32")


def cmd_clear(args: argparse.Namespace) -> None:
    store = Store(root=Path(args.store))
    if not store.exists():
        print(f"nothing to clear at {store.root}")
        return
    store.clear()
    print(f"cleared {store.root}")


# ────────────────────────────────────────────────────────────────────────
# Helpers & argparse wiring
# ────────────────────────────────────────────────────────────────────────

def _load_or_exit(store_root: str) -> Store:
    store = Store(root=Path(store_root))
    if not store.exists():
        print(f"no index at {store.root}.  run `notes-rag init <path>` first.",
              file=sys.stderr)
        sys.exit(1)
    store.load()
    return store


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store", default=str(DEFAULT_STORE_DIR),
        help=f"directory holding the snapvec index (default: {DEFAULT_STORE_DIR})",
    )
    parser.add_argument(
        "--embed-model", default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding model (default: {DEFAULT_EMBED_MODEL})",
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="notes-rag",
        description="Local RAG over markdown notes — snapvec + Ollama.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="build an index from a folder of .md files")
    _add_common_args(p_init)
    p_init.add_argument("path", help="folder to index")
    p_init.add_argument("--dim", type=int, default=DEFAULT_DIM,
                        help=f"embedding dimension (default: {DEFAULT_DIM})")
    p_init.add_argument("--bits", type=int, default=4, choices=[2, 3, 4],
                        help="snapvec quantisation bits (default: 4)")
    p_init.add_argument("--force", action="store_true",
                        help="overwrite an existing index")
    p_init.set_defaults(func=cmd_init)

    p_ask = sub.add_parser("ask", help="answer a question via RAG")
    _add_common_args(p_ask)
    p_ask.add_argument("query")
    p_ask.add_argument("-k", type=int, default=5, help="retrieved chunks (default: 5)")
    p_ask.add_argument("--tag", action="append", default=[],
                       help="restrict to notes carrying this tag (repeat for OR)")
    p_ask.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL,
                       help=f"Ollama chat model (default: {DEFAULT_CHAT_MODEL})")
    p_ask.set_defaults(func=cmd_ask)

    p_search = sub.add_parser("search", help="show top-k chunks without LLM")
    _add_common_args(p_search)
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)
    p_search.add_argument("--tag", action="append", default=[])
    p_search.set_defaults(func=cmd_search)

    p_status = sub.add_parser("status", help="show index statistics")
    _add_common_args(p_status)
    p_status.set_defaults(func=cmd_status)

    p_clear = sub.add_parser("clear", help="remove the saved index")
    _add_common_args(p_clear)
    p_clear.set_defaults(func=cmd_clear)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
