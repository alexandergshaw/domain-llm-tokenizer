"""Train a BPE tokenizer on domain text files."""

import argparse
import glob
import json
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def collect_text_files(data_dir: str) -> list[str]:
    """Recursively collect all .txt files under *data_dir*."""
    pattern = os.path.join(data_dir, "**", "*.txt")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def train(data_dir: str, output_dir: str, vocab_size: int) -> None:
    files = collect_text_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No .txt files found under '{data_dir}'. "
            "Add cleaned text files before training."
        )

    print(f"Found {len(files)} text file(s) for training.")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    tokenizer.train(files, trainer)

    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    special_token_ids = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
    config = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": special_token_ids,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    print("\n--- Training summary ---")
    print(f"  Vocab size : {config['vocab_size']}")
    for name, token_id in special_token_ids.items():
        print(f"  {name:8s} id : {token_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on domain text files."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing .txt training files (default: data/).",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizer",
        help="Directory to save the tokenizer (default: tokenizer/).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size (default: 8000).",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
