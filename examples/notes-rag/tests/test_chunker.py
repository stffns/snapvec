"""Tests for the markdown chunker."""
from __future__ import annotations

from pathlib import Path

from notes_rag.chunker import chunk, parse_markdown


def _write(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


class TestChunk:
    def test_short_document_becomes_single_chunk(self):
        out = chunk("one paragraph here")
        assert out == ["one paragraph here"]

    def test_paragraphs_merged_until_target(self):
        body = ("a" * 400 + "\n\n" + "b" * 400 + "\n\n" + "c" * 400)
        out = chunk(body, target_chars=900, min_chars=100)
        # 400 + 400 = 800 fits, +400 tips over → split after second
        assert len(out) == 2
        assert "a" * 400 in out[0] and "b" * 400 in out[0]
        assert "c" * 400 in out[1]

    def test_orphan_folded_into_prior_chunk(self):
        body = ("a" * 1400 + "\n\n" + "tiny")
        out = chunk(body, target_chars=1000, min_chars=100)
        # tiny is below min_chars → folded into first chunk
        assert len(out) == 1
        assert "tiny" in out[0]

    def test_empty_text(self):
        assert chunk("") == []


class TestParseMarkdown:
    def test_plain_body(self, tmp_path: Path):
        p = _write(tmp_path / "a.md", "Just some text.\n\nTwo paragraphs.")
        parsed = parse_markdown(p)
        assert parsed["tags"] == []
        assert "Just some text" in parsed["text"]

    def test_inline_frontmatter_tags(self, tmp_path: Path):
        p = _write(tmp_path / "a.md",
                   "---\ntags: [work, deep, python]\n---\nbody here.")
        parsed = parse_markdown(p)
        assert set(parsed["tags"]) == {"work", "deep", "python"}
        assert "body here" in parsed["text"]

    def test_block_frontmatter_tags(self, tmp_path: Path):
        p = _write(tmp_path / "a.md",
                   "---\ntags:\n  - work\n  - deep\n---\nbody here.")
        parsed = parse_markdown(p)
        assert set(parsed["tags"]) == {"work", "deep"}

    def test_inline_hashtags_also_captured(self, tmp_path: Path):
        p = _write(tmp_path / "a.md",
                   "some body with #python and #math tags")
        parsed = parse_markdown(p)
        assert "python" in parsed["tags"]
        assert "math" in parsed["tags"]

    def test_hashtags_do_not_double_count_with_frontmatter(self, tmp_path: Path):
        p = _write(tmp_path / "a.md",
                   "---\ntags: [python]\n---\n#python is nice")
        parsed = parse_markdown(p)
        assert parsed["tags"].count("python") == 1
