"""
Stage 1 — Data prep.

Downloads DialogSum (MIT) and formats each example into a (prompt, target) pair
using Qwen2.5's chat template. Writes train/val/test to ./data (git-ignored).

Run on: RTX 3080 box.

TODO / verify:
  - Confirm the current DialogSum HF dataset id and split names.
  - Confirm Qwen2.5-1.5B-Instruct chat-template application (apply_chat_template).
  - Decide on a max token length / truncation policy for long dialogues.
"""

from pathlib import Path
import json
from datasets import load_dataset
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_ID = "knkarthick/dialogsum"  # TODO: verify id
OUT_DIR = Path("data")

SYSTEM = "You are a helpful assistant that writes a concise, third-person summary of a conversation."
INSTRUCTION = "Summarize the following conversation:\n\n{dialogue}"


def format_example(tokenizer, dialogue: str, summary: str) -> dict:
    """Build a Qwen2.5 chat-format prompt with the summary as the target."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": INSTRUCTION.format(dialogue=dialogue)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # For SFT, target text is the summary. TRL's SFTTrainer can take a single
    # "text" field (prompt+target) or a prompt/completion pair — pick one and be consistent.
    return {"prompt": prompt, "completion": summary.strip()}


def main():
    OUT_DIR.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    ds = load_dataset(DATASET_ID)

    # TODO: verify column names ("dialogue", "summary") for the chosen DialogSum mirror.
    for split in ["train", "validation", "test"]:
        rows = [
            format_example(tokenizer, ex["dialogue"], ex["summary"])
            for ex in ds[split]
        ]
        out = OUT_DIR / f"{split}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(rows)} -> {out}")

    # Sanity check: print one formatted example.
    print("\n--- sample ---")
    print(json.dumps(rows[0], indent=2)[:1200])


if __name__ == "__main__":
    main()
