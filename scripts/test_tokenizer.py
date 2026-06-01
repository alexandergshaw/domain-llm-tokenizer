"""Load a trained tokenizer and demonstrate encode/decode on a sample prompt."""

import argparse
import os

from tokenizers import Tokenizer


DEFAULT_PROMPT = "The model learns domain-specific language patterns."


def test(tokenizer_path: str, prompt: str) -> None:
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer file not found: '{tokenizer_path}'. "
            "Run scripts/train_tokenizer.py first."
        )

    tokenizer = Tokenizer.from_file(tokenizer_path)

    encoding = tokenizer.encode(prompt)
    decoded = tokenizer.decode(encoding.ids)

    print(f"Prompt  : {prompt}")
    print(f"Tokens  : {encoding.tokens}")
    print(f"Token IDs: {encoding.ids}")
    print(f"Decoded : {decoded}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode and decode a prompt using a trained tokenizer."
    )
    parser.add_argument(
        "--tokenizer",
        default=os.path.join("tokenizer", "tokenizer.json"),
        help="Path to tokenizer.json (default: tokenizer/tokenizer.json).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text to encode/decode.",
    )
    args = parser.parse_args()

    test(tokenizer_path=args.tokenizer, prompt=args.prompt)


if __name__ == "__main__":
    main()
