"""
Stage 4 — Score the 4-bit MLX model with the SAME multi-reference ROUGE as Stage 3.

Generation now goes through mlx_lm (not transformers), but scoring reuses the exact Stage 3
path (train/rouge_common.py) so the 4-bit numbers are directly comparable to the committed
fp16 numbers (README / results/rouge_comparison.md). Run on: M2 Mac (MLX is Apple-silicon).

    python convert/eval_mlx_rouge.py --limit 3 --show 3   # Gate A: eyeball a few outputs
    python convert/eval_mlx_rouge.py                       # Gate B: full 500-dialogue re-score

Decoding parity with the fp16 run (train/eval_rouge.py build_gen_config), verified:
  - greedy: make_sampler(temp=0.0), no top_p/top_k  ==  do_sample=False, num_beams=1
  - NO repetition penalty  ==  the fp16 run set repetition_penalty=1.0 (neutralised Qwen's 1.1);
    mlx_lm applies none unless asked, so this matches.
  - max_tokens=96  ==  max_new_tokens=96
  - stop on {151645, 151643}  ==  EOS_IDS. mlx_model's tokenizer already reports exactly this
    eos_token_ids set (asserted below), so <|im_end|> and <|endoftext|> both stop generation.
  - the prompt in test.jsonl already carries the full Qwen chat template and ends with the
    assistant header; it is fed VERBATIM (no re-templating) to reproduce identical inputs.

Scoring is identical: references as list-of-lists -> multi-reference MAX f-measure per rouge
type, use_stemmer=True, own mean over per-example scores (rouge_common.score_rouge / means).
"""

from pathlib import Path
import argparse
import json
import sys
import time

import evaluate
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# Reuse the exact Stage 3 scoring + diagnostics (torch-free module).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from rouge_common import (  # noqa: E402
    ROUGE_TYPES,
    diagnostics,
    load_test,
    means,
    score_precision_recall,
    score_rouge,
)

EOS_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|> — must match train/eval_rouge.py EOS_IDS
RESULTS = Path("results")


def parse_args():
    p = argparse.ArgumentParser(description="Score the 4-bit MLX model with multi-reference ROUGE.")
    p.add_argument("--mlx-path", default="mlx_model")
    p.add_argument("--test", default="data/test.jsonl")
    p.add_argument("--limit", type=int, default=None, help="Stride-sample N dialogues (fast path).")
    p.add_argument("--max-tokens", type=int, default=96, help="Matches fp16 max_new_tokens.")
    p.add_argument("--no-stemmer", action="store_true", help="Score without Porter stemming.")
    p.add_argument("--show", type=int, default=0, help="Print this many dialogue/ref/pred examples.")
    p.add_argument("--out", default=None, help="Default: results/rouge_mlx_4bit[_limit{N}].md")
    return p.parse_args()


def model_dir_bytes(path: str) -> int:
    return sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))


def generate_all(model, tokenizer, prompts, sampler, max_tokens):
    """Greedy-generate one prediction per prompt (fed verbatim). Returns (preds, meta) in order.

    meta rows carry n_new_tokens and hit_cap so rouge_common.diagnostics works unchanged:
      - n_new_tokens: mlx generation_tokens (generated tokens incl. the stop token)
      - hit_cap: finish_reason == 'length' (ran out the max_tokens budget without hitting EOS)
    """
    preds, meta = [], []
    n = len(prompts)
    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt)  # verbatim chat prompt -> ids; no chat-template re-apply
        text, last = "", None
        for resp in stream_generate(model, tokenizer, ids, max_tokens=max_tokens, sampler=sampler):
            text += resp.text
            last = resp
        preds.append(text.strip())
        meta.append({
            "n_new_tokens": int(last.generation_tokens) if last else 0,
            "hit_cap": bool(last and last.finish_reason == "length"),
        })
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n}", flush=True)
    return preds, meta


def write_report(path, rows, preds, raw, diag, pr, ref_tokens, args, size_bytes):
    n = len(rows)
    refs_hist = {}
    for r in rows:
        refs_hist[len(r["references"])] = refs_hist.get(len(r["references"]), 0) + 1

    with path.open("w") as f:
        f.write("# ROUGE — 4-bit MLX (DialogSum test)\n\n")
        f.write(
            f"Qwen2.5-1.5B-Instruct QLoRA fine-tune (r=16, a=32), merged to fp16, then quantized to "
            f"**4-bit MLX** (q-bits 4, group size 64).\n**{n} dialogues**, multi-reference ROUGE "
            f"(max f-measure over {'/'.join(str(k) for k in sorted(refs_hist))} human refs per "
            f"dialogue).\n\n"
            "Same decoding and same scoring as the fp16 run (`results/rouge_comparison.md`), so these "
            "numbers are directly comparable. Quantization delta vs fp16: "
            "`results/quantization_delta.md`.\n\n"
        )

        f.write("## Headline — raw model output\n\n")
        f.write("| Metric | 4-bit MLX |\n|---|---:|\n")
        for k in ROUGE_TYPES:
            f.write(f"| {k} | {raw[k]:.4f} |\n")
        f.write("\n")

        f.write("## Diagnostics\n\n")
        f.write("| | 4-bit MLX |\n|---|---:|\n")
        for key, label in [
            ("preamble_rate", "preamble rate"),
            ("mean_output_tokens", f"mean output tokens (refs: {ref_tokens:.1f})"),
            ("median_output_tokens", "median output tokens"),
            ("truncation_rate", f"truncation rate (hit {args.max_tokens} cap)"),
            ("empty_rate", "empty output rate"),
        ]:
            fmt = ".1f" if "tokens" in key else ".3f"
            f.write(f"| {label} | {diag[key]:{fmt}} |\n")
        f.write("\n")

        f.write("## Precision / recall decomposition\n\n")
        f.write("| Metric | Precision | Recall | F |\n|---|---:|---:|---:|\n")
        for k in ROUGE_TYPES:
            s = pr[k]
            f.write(f"| {k} | {s['precision']:.4f} | {s['recall']:.4f} | {s['fmeasure']:.4f} |\n")
        f.write("\n")

        f.write("## Run config\n\n")
        f.write(f"- examples: **{n}**" + (" (stride-sampled from 500)\n" if args.limit else "\n"))
        f.write(f"- refs per dialogue: {dict(sorted(refs_hist.items()))}\n")
        f.write(
            "- decoding (matches fp16): greedy (`temp=0.0`, no top_p/top_k), "
            f"`max_tokens={args.max_tokens}`, no repetition penalty, stop on "
            "`{151645, 151643}`\n"
        )
        f.write(f"- prompt: Stage 1 chat template, fed verbatim (ends `<|im_start|>assistant\\n`)\n")
        f.write(
            f"- stemmer: `use_stemmer={not args.no_stemmer}` | aggregator: own mean over "
            "per-example scores\n"
        )
        f.write(f"- model: `{args.mlx_path}` (4-bit MLX, {size_bytes / 2**20:.0f} MB safetensors)\n")
        f.write("- generation: mlx_lm `stream_generate`; scoring reuses `train/rouge_common.py`\n")


def main():
    args = parse_args()
    use_stemmer = not args.no_stemmer
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS / (
        "rouge_mlx_4bit.md" if args.limit is None else f"rouge_mlx_4bit_limit{args.limit}.md"
    )

    rows = load_test(args.test, args.limit)
    prompts = [r["prompt"] for r in rows]

    print(f"loading {args.mlx_path} ...", flush=True)
    model, tokenizer = load(args.mlx_path)
    # Parity guard: the fp16 run stopped on exactly these ids. Fail loudly if the converted
    # tokenizer disagrees, rather than silently decoding to a different stop set.
    if set(tokenizer.eos_token_ids) != EOS_IDS:
        raise SystemExit(
            f"tokenizer.eos_token_ids={set(tokenizer.eos_token_ids)} != fp16 EOS_IDS={EOS_IDS} — "
            "decoding would not match the fp16 run."
        )
    sampler = make_sampler(temp=0.0)  # greedy

    print(f"{len(rows)} dialogues | greedy, max_tokens={args.max_tokens}", flush=True)
    t0 = time.time()
    preds, meta = generate_all(model, tokenizer, prompts, sampler, args.max_tokens)
    secs = time.time() - t0
    print(f"generated in {secs:.0f}s ({len(rows) / secs:.1f} dialogues/s)")

    rouge = evaluate.load("rouge")

    # Same self-check as eval_rouge.py: a reference scored against itself must be rouge1=1.0.
    selfcheck = score_rouge(rouge, [r["references"][0] for r in rows], rows, use_stemmer)
    if not np.allclose(selfcheck["rouge1"], 1.0):
        raise SystemExit("self-scoring a reference didn't yield rouge1=1.0 — scoring path is wrong")

    raw = means(score_rouge(rouge, preds, rows, use_stemmer))
    diag = diagnostics(preds, meta)
    pr = score_precision_recall(preds, rows, use_stemmer)
    # evaluate (headline f) and rouge_scorer (P/R path) must agree, exactly as eval_rouge.py asserts.
    for rt in ROUGE_TYPES:
        if not np.isclose(pr[rt]["fmeasure"], raw[rt], atol=1e-6):
            raise SystemExit(f"rouge_scorer and evaluate disagree on {rt}: {pr[rt]['fmeasure']} vs {raw[rt]}")

    ref_tokens = float(np.mean([len(tokenizer.encode(ref)) for r in rows for ref in r["references"]]))
    size_bytes = model_dir_bytes(args.mlx_path)

    write_report(out, rows, preds, raw, diag, pr, ref_tokens, args, size_bytes)

    print(f"\nwrote {out}\n")
    print(f"{'metric':<8} {'4-bit MLX':>10}")
    for k in ROUGE_TYPES:
        print(f"{k:<8} {raw[k]:>10.4f}")
    print(
        f"\nmean output tokens {diag['mean_output_tokens']:.1f} (refs {ref_tokens:.1f}) | "
        f"truncation {diag['truncation_rate']:.3f} | empty {diag['empty_rate']:.3f}"
    )

    for j in range(min(args.show, len(rows))):
        r = rows[j]
        print(f"\n{'=' * 70}\n[example {j}] refs={len(r['references'])}")
        dlg = r["prompt"].split("Summarize the following conversation:\n\n", 1)[-1].split("<|im_end|>", 1)[0]
        print(f"--- dialogue ---\n{dlg.strip()[:900]}")
        print(f"--- reference[0] ---\n{r['references'][0]}")
        print(f"--- 4-bit MLX summary ---\n{preds[j] or '(empty)'}")


if __name__ == "__main__":
    main()
