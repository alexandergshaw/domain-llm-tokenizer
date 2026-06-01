"""Prepare a tokenizer training corpus from a directory of cleaned text files."""

import argparse
import glob
import os
import re


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (preserving single newlines between lines)."""
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs on each line
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def prepare(input_dir: str, output_file: str, min_chars: int = 100) -> None:
    pattern = os.path.join(input_dir, "**", "*.txt")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found under '{input_dir}'."
        )

    processed = 0
    skipped = 0
    total_chars = 0
    segments: list[str] = []

    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()

        cleaned = normalize_whitespace(raw)

        if len(cleaned) < min_chars:
            skipped += 1
            continue

        segments.append(cleaned)
        total_chars += len(cleaned)
        processed += 1

    if not segments:
        raise ValueError(
            f"All files were skipped (each had fewer than {min_chars} characters). "
            "Add more content to your text files."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n<eos>\n".join(segments))
        out.write("\n<eos>\n")

    print(f"Files processed : {processed}")
    print(f"Files skipped   : {skipped} (fewer than {min_chars} characters)")
    print(f"Total characters: {total_chars}")
    print(f"Corpus written  : {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine cleaned .txt files into a single tokenizer training corpus."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing cleaned .txt files.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path for the combined corpus output file.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="Minimum character count to include a file (default: 100).",
    )
    args = parser.parse_args()

    prepare(
        input_dir=args.input_dir,
        output_file=args.output_file,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    main()
