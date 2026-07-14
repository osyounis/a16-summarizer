"""
Stage 1 — Data prep.

Downloads DialogSum (MIT) and formats each example into a (prompt, target) pair
using Qwen2.5's chat template. Writes train/val/test to ./data (git-ignored).

Run on: RTX 3080 box.

Format contract (MUST stay identical across Stage 1 / Stage 3 eval / Stage 5 app):
  - SYSTEM + INSTRUCTION below define the exact wording the app must replicate.
  - `prompt` is the chat template rendered with add_generation_prompt=True, i.e. it
    ends with the `<|im_start|>assistant\n` header. Re-tokenizing this string (as
    eval_rouge.py does) reproduces the identical token ids because Qwen's
    <|im_start|>/<|im_end|> are real vocab special tokens.
  - `completion` is the plain summary text (no template, no EOS). TRL appends EOS
    during Stage 2 training; the app stops generation on <|im_end|>.

Dataset notes (verified against knkarthick/dialogsum):
  - Splits: train / validation / test. Columns: id, dialogue, summary, topic.
  - The TEST split is multi-reference: each dialogue appears as 3 rows with ids
    test_<n>_1/_2/_3 (same dialogue, 3 different human summaries). We group these
    into ONE row per dialogue with a `references` list so Stage 3 can score proper
    multi-reference ROUGE. Train/validation are single-reference (one row/dialogue).
"""

from pathlib import Path
from collections import OrderedDict
import json
import re
from datasets import load_dataset
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "knkarthick/dialogsum"
OUT_DIR = Path("data")

SYSTEM = "You are a helpful assistant that writes a concise, third-person summary of a conversation."
INSTRUCTION = "Summarize the following conversation:\n\n{dialogue}"

# test ids look like "test_0_1", "test_0_2", "test_0_3" -> stem "test_0".
_REF_SUFFIX = re.compile(r"_\d+$")


def render_prompt(tokenizer, dialogue: str) -> str:
    """Render the Qwen2.5 chat prompt (system + user), ready for generation."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTION.format(dialogue=dialogue)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_example(tokenizer, dialogue: str, summary: str) -> dict:
    """Build a Qwen2.5 chat-format prompt with the summary as the target."""
    return {
        "prompt": render_prompt(tokenizer, dialogue),
        "completion": summary.strip(),
    }


def write_jsonl(rows, path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_single_ref(tokenizer, split_ds) -> list:
    """train / validation: one (prompt, completion) per row."""
    return [
        format_example(tokenizer, ex["dialogue"], ex["summary"])
        for ex in split_ds
    ]


def build_test(tokenizer, split_ds):
    """test: group the 3 human references per dialogue into one row.

    Returns (rows, refs_histogram) where each row is
    {"prompt", "completion" (= references[0]), "references": [...]}.
    """
    groups = OrderedDict()  # stem -> {"dialogue": str, "summaries": [str, ...]}
    for ex in split_ds:
        stem = _REF_SUFFIX.sub("", ex["id"])  # "test_0_1" -> "test_0"
        g = groups.get(stem)
        if g is None:
            g = {"dialogue": ex["dialogue"], "summaries": []}
            groups[stem] = g
        s = ex["summary"].strip()
        if s not in g["summaries"]:  # dedupe, preserve order
            g["summaries"].append(s)

    rows = []
    histogram = {}
    for g in groups.values():
        refs = g["summaries"]
        histogram[len(refs)] = histogram.get(len(refs), 0) + 1
        rows.append({
            "prompt": render_prompt(tokenizer, g["dialogue"]),
            "completion": refs[0],
            "references": refs,
        })
    return rows, histogram


def token_len_stats(tokenizer, rows) -> dict:
    """Length (in tokens) of prompt+completion, to inform Stage 2 max_seq_length."""
    lengths = sorted(
        len(tokenizer(r["prompt"] + r["completion"]).input_ids) for r in rows
    )
    n = len(lengths)

    def pct(p):
        return lengths[min(n - 1, int(p * n))]

    return {"n": n, "max": lengths[-1], "p95": pct(0.95), "p99": pct(0.99)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = load_dataset(DATASET_ID)

    outputs = {}  # split -> rows

    for split in ["train", "validation"]:
        rows = build_single_ref(tokenizer, ds[split])
        write_jsonl(rows, OUT_DIR / f"{split}.jsonl")
        outputs[split] = rows
        print(f"{split}: {len(rows)} rows -> {OUT_DIR / f'{split}.jsonl'}")

    test_rows, hist = build_test(tokenizer, ds["test"])
    write_jsonl(test_rows, OUT_DIR / "test.jsonl")
    outputs["test"] = test_rows
    print(
        f"test: {len(ds['test'])} raw rows -> {len(test_rows)} unique dialogues "
        f"-> {OUT_DIR / 'test.jsonl'}"
    )
    print(f"  refs-per-dialogue histogram: {dict(sorted(hist.items()))}")

    # Token-length stats (prompt+completion) to pick Stage 2 max_seq_length.
    print("\n--- token length (prompt+completion) ---")
    for split, rows in outputs.items():
        print(f"{split}: {token_len_stats(tokenizer, rows)}")

    # Sanity check: print a few formatted examples.
    print("\n--- sample train pair ---")
    print(json.dumps(outputs["train"][0], indent=2)[:1400])
    print("\n--- sample test row (multi-reference) ---")
    print(json.dumps(outputs["test"][0], indent=2)[:1600])


if __name__ == "__main__":
    main()
