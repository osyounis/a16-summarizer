"""
Stage 3b — Evaluate base vs. fine-tuned with ROUGE.

Generates summaries from BOTH stock Qwen2.5-1.5B and the merged fine-tuned model over
the DialogSum test split, scores ROUGE-1/2/L, writes results/rouge_comparison.md.

This is the deliverable that proves the fine-tune worked. Run on: RTX 3080 box.

TODO / verify:
  - Batch generation for speed; cap max_new_tokens (summaries are short).
  - Use the same prompt template as prepare_data.py for a fair comparison.
  - Consider a small subset (e.g. 200 test examples) for a fast first pass, then full.
"""

from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MERGED_MODEL = "out/merged"
TEST = "data/test.jsonl"
OUT = Path("results/rouge_comparison.md")


def load_test():
    with open(TEST) as f:
        return [json.loads(l) for l in f]


def generate(model, tokenizer, prompt, max_new_tokens=96):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def score(model_path, rows, rouge, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    preds = [generate(model, tokenizer, r["prompt"]) for r in rows]
    refs = [r["completion"] for r in rows]
    return rouge.compute(predictions=preds, references=refs)


def main():
    rouge = evaluate.load("rouge")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    rows = load_test()  # TODO: subset for a fast first pass

    base = score(BASE_MODEL, rows, rouge, tokenizer)
    tuned = score(MERGED_MODEL, rows, rouge, tokenizer)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        f.write("# ROUGE — base vs. fine-tuned (DialogSum test)\n\n")
        f.write("| Metric | Base | Fine-tuned | Δ |\n|---|---:|---:|---:|\n")
        for k in ["rouge1", "rouge2", "rougeL"]:
            b, t = base[k], tuned[k]
            f.write(f"| {k} | {b:.4f} | {t:.4f} | {t - b:+.4f} |\n")
    print(f"wrote {OUT}")
    print("base:", base)
    print("tuned:", tuned)


if __name__ == "__main__":
    main()
