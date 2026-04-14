"""Parse markdown files into retrieval chunks.

Chunking strategy: merge consecutive paragraphs until roughly ``target_chars``
are reached.  Keeps paragraph boundaries intact — we never split mid-sentence,
which preserves semantic coherence at query time.

Frontmatter: we extract ``tags`` (YAML list or comma-separated) so callers can
use them with ``filter_ids`` at query time.  Inline ``#hashtag`` markers in the
body are also collected.  This is intentionally a small-surface regex parser —
PyYAML is an unnecessary dependency for the 1% of notes that use frontmatter.
"""
from __future__ import annotations

import re
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HASHTAG_RE = re.compile(r"(?<![\w/])#([a-zA-Z][a-zA-Z0-9_\-/]*)")
_INLINE_LIST_RE = re.compile(r"^tags\s*:\s*\[(.*?)\]\s*$", re.MULTILINE)
_BLOCK_ITEM_RE = re.compile(r"^\s*-\s+(\S.*?)\s*$", re.MULTILINE)


def _extract_tags(frontmatter: str) -> list[str]:
    """Pull ``tags:`` from a frontmatter block, either inline or YAML list."""
    tags: list[str] = []
    m = _INLINE_LIST_RE.search(frontmatter)
    if m:
        tags.extend(
            t.strip(" '\"") for t in m.group(1).split(",") if t.strip()
        )
        return tags
    # Block form:
    #     tags:
    #       - foo
    #       - bar
    lines = frontmatter.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^\s*tags\s*:\s*$", line):
            for follow in lines[i + 1:]:
                m2 = _BLOCK_ITEM_RE.match(follow)
                if not m2:
                    break
                tags.append(m2.group(1).strip(" '\""))
            break
    return tags


def parse_markdown(path: Path) -> dict:
    """Read ``path`` and return ``{"text": body, "tags": [...]}``."""
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    tags: list[str] = []
    body = raw
    m = _FRONTMATTER_RE.match(raw)
    if m:
        tags.extend(_extract_tags(m.group(1)))
        body = raw[m.end():]
    # Inline hashtags in the body too
    tags.extend(_HASHTAG_RE.findall(body))
    # Dedupe, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return {"text": body, "tags": unique}


def chunk(text: str, target_chars: int = 1500, min_chars: int = 200) -> list[str]:
    """Greedy paragraph-packing chunker.

    - Paragraphs are split on blank lines.
    - We merge them until ``target_chars`` is exceeded; emit and start fresh.
    - The last emitted chunk may be shorter than ``target_chars``; we still
      emit it unless it is below ``min_chars`` AND there is a prior chunk to
      merge it into (then we fold it into the prior chunk to avoid orphans).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for p in paragraphs:
        if current_len + len(p) > target_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(p)
        current_len += len(p) + 2
    if current:
        tail = "\n\n".join(current)
        if len(tail) < min_chars and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + tail
        else:
            chunks.append(tail)
    return chunks
