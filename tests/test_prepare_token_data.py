"""Tests for prepare_token_data.py corpus preparation script."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prepare_token_data import normalize_whitespace, prepare


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------


def test_normalize_collapses_spaces():
    result = normalize_whitespace("hello   world")
    assert result == "hello world"


def test_normalize_collapses_blank_lines():
    result = normalize_whitespace("line1\n\n\n\nline2")
    assert result == "line1\n\nline2"


def test_normalize_strips_edges():
    result = normalize_whitespace("  hello  ")
    assert result == "hello"


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

LONG_TEXT = "Domain tokenization is important for language models. " * 5


def _make_corpus_dir(tmp_path, files: dict) -> None:
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def test_prepare_combines_files(tmp_path):
    out_file = str(tmp_path / "out" / "corpus.txt")
    _make_corpus_dir(tmp_path / "in", {"a.txt": LONG_TEXT, "b.txt": LONG_TEXT})
    prepare(input_dir=str(tmp_path / "in"), output_file=out_file)
    with open(out_file, encoding="utf-8") as f:
        content = f.read()
    assert "<eos>" in content
    assert content.count("<eos>") == 2


def test_prepare_skips_tiny_files(tmp_path):
    out_file = str(tmp_path / "corpus.txt")
    _make_corpus_dir(
        tmp_path / "in",
        {
            "tiny.txt": "hi",
            "big.txt": LONG_TEXT,
        },
    )
    prepare(input_dir=str(tmp_path / "in"), output_file=out_file)
    with open(out_file, encoding="utf-8") as f:
        content = f.read()
    assert "hi" not in content
    assert "Domain" in content


def test_prepare_raises_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare(input_dir=str(tmp_path / "empty"), output_file=str(tmp_path / "out.txt"))


def test_prepare_raises_all_skipped(tmp_path):
    out_file = str(tmp_path / "corpus.txt")
    _make_corpus_dir(tmp_path / "in", {"tiny.txt": "hi"})
    with pytest.raises(ValueError):
        prepare(input_dir=str(tmp_path / "in"), output_file=out_file)


def test_prepare_recursive(tmp_path):
    out_file = str(tmp_path / "corpus.txt")
    _make_corpus_dir(
        tmp_path / "in",
        {
            "root.txt": LONG_TEXT,
            "sub/nested.txt": LONG_TEXT,
        },
    )
    prepare(input_dir=str(tmp_path / "in"), output_file=out_file)
    with open(out_file, encoding="utf-8") as f:
        content = f.read()
    assert content.count("<eos>") == 2
