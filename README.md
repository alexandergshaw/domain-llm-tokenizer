# domain-llm-tokenizer

Train a custom **Byte-Pair Encoding (BPE) tokenizer** from cleaned domain text,
ready to use with a small GPT-style model.

---

## Directory layout

```
domain-llm-tokenizer/
├── data/               ← place your cleaned .txt files here
├── tokenizer/          ← trained tokenizer.json + config.json land here
├── scripts/
│   ├── prepare_token_data.py   ← combine raw docs into one corpus
│   ├── train_tokenizer.py      ← train the BPE tokenizer
│   └── test_tokenizer.py       ← quick encode/decode sanity check
├── tests/              ← pytest test suite
├── requirements.txt
└── .gitignore
```

---

## Quick-start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Add cleaned text

Drop one or more `.txt` files into `data/` (sub-directories are fine).
Each file should contain plain UTF-8 text — no HTML, no markdown artefacts.

### 3 — (Optional) Prepare a corpus

If you have many raw documents in a separate folder, merge them into a single
training corpus first:

```bash
python scripts/prepare_token_data.py \
    --input-dir  /path/to/raw/docs \
    --output-file data/corpus.txt
```

The script will:
- recursively find every `.txt` file under `--input-dir`
- skip files shorter than 100 characters
- strip excessive whitespace
- join documents with `<eos>` separators
- report how many files were processed and the total character count

### 4 — Train the tokenizer

```bash
python scripts/train_tokenizer.py
```

By default this reads `data/` and writes `tokenizer/tokenizer.json` plus
`tokenizer/config.json`.

**Options**

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Source directory for `.txt` files |
| `--output-dir` | `tokenizer` | Destination directory |
| `--vocab-size` | `8000` | Vocabulary size |

Example with a custom vocab size:

```bash
python scripts/train_tokenizer.py --vocab-size 4000
```

### 5 — Test the tokenizer

```bash
python scripts/test_tokenizer.py
```

This encodes a sample sentence and prints the tokens, token IDs, and decoded
text. Pass `--prompt "your text here"` to test a custom string.

---

## Recommended vocab sizes for tiny models

| Model size | Recommended vocab size | Notes |
|---|---|---|
| Toy / demo (< 1 M params) | 1 000 – 2 000 | Fits in RAM; fast iteration |
| Tiny (1 – 10 M params) | 4 000 – 8 000 | Good coverage of domain terms |
| Small (10 – 100 M params) | 8 000 – 16 000 | Closer to GPT-2 range |
| Medium (> 100 M params) | 16 000 – 32 000 | General-purpose quality |

The default of **8 000** is a sensible starting point for a tiny domain model
with a few MB of training text.

---

## Special tokens

| Token | Role |
|-------|------|
| `<pad>` | Padding (id 0) |
| `<unk>` | Unknown subword |
| `<bos>` | Beginning of sequence |
| `<eos>` | End of sequence / document separator |

---

## Running the tests

```bash
pip install pytest
pytest tests/
```