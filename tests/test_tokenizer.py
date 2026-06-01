"""Tests for basic tokenizer encode/decode behavior."""

import json
import os
import sys
import tempfile

import pytest

# Allow importing from scripts/ without installing as a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from train_tokenizer import SPECIAL_TOKENS, collect_text_files, train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. " * 40
    + "Domain language models require domain-specific tokenizers. " * 40
)


@pytest.fixture()
def corpus_dir(tmp_path):
    """Create a temporary directory with a couple of .txt training files."""
    (tmp_path / "doc1.txt").write_text(SAMPLE_TEXT, encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "doc2.txt").write_text(
        "Tokenization splits text into subword units. " * 40, encoding="utf-8"
    )
    return tmp_path


@pytest.fixture()
def trained_tokenizer_dir(corpus_dir, tmp_path):
    """Train a small tokenizer and return the output directory."""
    out_dir = tmp_path / "tokenizer"
    train(
        data_dir=str(corpus_dir),
        output_dir=str(out_dir),
        vocab_size=200,
    )
    return out_dir


# ---------------------------------------------------------------------------
# collect_text_files
# ---------------------------------------------------------------------------


def test_collect_text_files_finds_nested(corpus_dir):
    files = collect_text_files(str(corpus_dir))
    assert len(files) == 2
    assert all(f.endswith(".txt") for f in files)


def test_collect_text_files_empty_dir(tmp_path):
    files = collect_text_files(str(tmp_path))
    assert files == []


# ---------------------------------------------------------------------------
# train — output files
# ---------------------------------------------------------------------------


def test_train_creates_tokenizer_json(trained_tokenizer_dir):
    assert (trained_tokenizer_dir / "tokenizer.json").exists()


def test_train_creates_config_json(trained_tokenizer_dir):
    assert (trained_tokenizer_dir / "config.json").exists()


def test_config_json_has_expected_keys(trained_tokenizer_dir):
    with open(trained_tokenizer_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    assert "vocab_size" in config
    assert "special_tokens" in config
    for tok in SPECIAL_TOKENS:
        assert tok in config["special_tokens"]


def test_train_raises_when_no_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        train(data_dir=str(tmp_path), output_dir=str(tmp_path / "out"), vocab_size=200)


# ---------------------------------------------------------------------------
# Encode / decode round-trip
# ---------------------------------------------------------------------------


def test_encode_returns_ids(trained_tokenizer_dir):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained_tokenizer_dir / "tokenizer.json"))
    encoding = tok.encode("The quick brown fox")
    assert len(encoding.ids) > 0
    assert all(isinstance(i, int) for i in encoding.ids)


def test_decode_round_trip(trained_tokenizer_dir):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained_tokenizer_dir / "tokenizer.json"))
    prompt = "the quick brown fox"
    encoding = tok.encode(prompt)
    decoded = tok.decode(encoding.ids)
    # Decoded text should contain the same words (BPE may merge/split differently)
    assert "fox" in decoded


def test_special_tokens_have_ids(trained_tokenizer_dir):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained_tokenizer_dir / "tokenizer.json"))
    for special in SPECIAL_TOKENS:
        token_id = tok.token_to_id(special)
        assert token_id is not None, f"Special token {special!r} has no ID"
        assert isinstance(token_id, int)
