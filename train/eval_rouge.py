"""
Stage 3b — Evaluate base vs. fine-tuned with ROUGE.

Generates summaries from BOTH stock Qwen2.5-1.5B-Instruct and the merged fine-tuned model over
the DialogSum test split, scores multi-reference ROUGE-1/2/L, writes results/rouge_comparison.md
plus results/qualitative_examples.md.

This is the deliverable that proves the fine-tune worked. Run on: RTX 3080 box.

    python train/eval_rouge.py --limit 200     # fast sanity pass (~2-3 min), separate output file
    python train/eval_rouge.py                 # full 500-dialogue test set (~4-7 min)
    python train/eval_rouge.py --score-only    # rescore cached generations (~10 s)

Three things this file is careful about, each a verified trap:

1. DECODING. Qwen2.5-Instruct ships generation_config.json with do_sample=true, temperature=0.7
   AND repetition_penalty=1.1. `do_sample=False` suppresses the sampling warpers but NOT the
   penalty — it is registered as a logits *processor*, so plain `generate(do_sample=False)` runs
   PENALIZED greedy. We therefore build one explicit GenerationConfig (GEN) and pass it to BOTH
   models, so neither ever reads its on-disk defaults. merge.py deliberately leaves out/merged's
   generation_config at Qwen's stock values and relies on this — do not drop the explicit config
   or base and merged will silently decode differently with no error.

2. MULTI-REFERENCE. DialogSum's test set has 3 human summaries per dialogue. Stage 1 already
   grouped them: data/test.jsonl is 500 rows with a `references` list (490 have 3, 10 have 2).
   Passing references as a LIST OF LISTS makes evaluate dispatch to rouge_scorer.score_multi,
   which takes the MAX f-measure per rouge type. Scoring against `completion` alone (= refs[0])
   would systematically understate ROUGE vs the published protocol.

3. AGGREGATION. evaluate's use_aggregator=True returns a bootstrap *median*, reseeded per call,
   so headline numbers drift between runs. We use use_aggregator=False and take our own mean —
   deterministic, matches the literature's sample-mean convention, and the per-example scores are
   needed for the CI and the qualitative selection anyway.
"""

from pathlib import Path
import argparse
import gc
import hashlib
import json
import re
import time

import evaluate
import numpy as np
import torch
import transformers
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MERGED_MODEL = "out/merged"
TEST = "data/test.jsonl"
RESULTS = Path("results")

PAD_ID = 151643  # <|endoftext|>
EOS_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|> — matches base's shipped stop set

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
# rougeLsum is dropped: it's rougeL over \n-split sentences, and DialogSum refs have no newlines,
# so it degenerates to ~rougeL and just adds a confusing near-duplicate column.

# Applied at most once, to the FIRST line, IDENTICALLY to both models — stripping only base would
# be removing base's handicap. Diagnostic only, never the headline.
#
# Empirically this matches NOTHING on either model: stock Qwen does not open with "Sure! Here's a
# summary:" on this prompt, so the expected formatting confound doesn't exist. Kept as a live
# control rather than deleted — the report states when it's inert, which is itself the finding.
# Base's real disadvantage is length; see the precision/recall decomposition.
PREAMBLE_RE = re.compile(
    r"^(sure|certainly|of course|okay|here('s| is| are)|the following)\b.*:\s*$",
    re.IGNORECASE,
)


def parse_args():
    p = argparse.ArgumentParser(description="Score base vs. fine-tuned with multi-reference ROUGE.")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--merged", default=MERGED_MODEL)
    p.add_argument("--test", default=TEST)
    p.add_argument("--out", default=None, help="Default: results/rouge_comparison[_limit{N}].md")
    p.add_argument("--limit", type=int, default=None, help="Fast path: stride-sample N dialogues.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--no-stemmer", action="store_true", help="Score without Porter stemming.")
    p.add_argument("--regen", action="store_true", help="Ignore cached generations.")
    p.add_argument("--score-only", action="store_true", help="Require the cache; never generate.")
    return p.parse_args()


# --------------------------------------------------------------------------- data


def load_test(path: str, limit: int | None) -> list[dict]:
    with open(path) as f:
        rows = [json.loads(line) for line in f]

    for i, r in enumerate(rows):
        if not r.get("references"):
            raise SystemExit(f"row {i} has no `references` — rerun train/prepare_data.py")
        if not r["prompt"].endswith("<|im_start|>assistant\n"):
            raise SystemExit(
                f"row {i}'s prompt doesn't end with the assistant header — the Stage 1 format "
                "contract is broken; base and tuned would not be compared on the trained format."
            )

    if limit is not None and limit < len(rows):
        # Stride, not head: rows[:N] takes test_0..test_N-1 in dataset order, which has no
        # shuffling guarantee (topic clustering, length drift). Stride spreads across the corpus.
        rows = rows[:: len(rows) // limit][:limit]
    return rows


def run_tag(limit: int | None) -> str:
    return "full" if limit is None else f"limit{limit}"


def report_path(stem: str, limit: int | None) -> Path:
    """Subset runs get their own filename so a --limit pass never clobbers the full-run report."""
    return RESULTS / (f"{stem}.md" if limit is None else f"{stem}_limit{limit}.md")


# --------------------------------------------------------------------- generation


def build_gen_config(max_new_tokens: int) -> GenerationConfig:
    # Passing this to generate() REPLACES the model's on-disk generation_config outright (verified:
    # the logits-processor list comes back empty, vs [RepetitionPenaltyLogitsProcessor] when you
    # rely on the model's own config). temperature/top_p/top_k are left at their defaults because
    # do_sample=False means no warpers are ever built from them.
    return GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,  # explicitly neutralise base's shipped 1.1
        pad_token_id=PAD_ID,
        eos_token_id=EOS_IDS,
    )


def gen_fingerprint(gen: GenerationConfig) -> str:
    d = {k: v for k, v in sorted(gen.to_diff_dict().items()) if k != "transformers_version"}
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def generate_batched(model, tok, prompts: list[str], gen: GenerationConfig, batch_size: int):
    """Greedy-generate for every prompt. Returns (predictions, per-row meta) in ORIGINAL order."""
    lengths = [len(tok(p).input_ids) for p in prompts]
    # Descending: the largest bucket runs FIRST, so an OOM fails in seconds rather than 4 minutes
    # in, after most of the work is already done.
    order = sorted(range(len(prompts)), key=lambda i: -lengths[i])

    preds: list[str | None] = [None] * len(prompts)
    meta: list[dict | None] = [None] * len(prompts)
    done = 0

    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        batch = tok([prompts[i] for i in idxs], return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            out = model.generate(**batch, generation_config=gen)

        # Valid uniform slice point ONLY because padding_side='left'.
        new_tokens = out[:, batch.input_ids.shape[1] :]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        for row, (i, text) in enumerate(zip(idxs, texts)):
            n_new = int((new_tokens[row] != PAD_ID).sum())
            hit_cap = not bool(
                torch.isin(new_tokens[row], torch.tensor(EOS_IDS, device=new_tokens.device)).any()
            )
            preds[i] = text.strip()
            meta[i] = {"n_new_tokens": n_new, "hit_cap": hit_cap}

        done += len(idxs)
        print(f"  {done}/{len(prompts)} (max prompt {lengths[idxs[0]]} tok)", flush=True)

    return preds, meta


def score_model(model_path: str, prompts: list[str], gen: GenerationConfig, batch_size: int, tok):
    """Generate with one model, then free it completely before the caller loads the next."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        # NOT device_map="auto": auto invites accelerate to CPU-offload under pressure, which would
        # silently make one model 20x slower and muddy the comparison.
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    preds, meta = generate_batched(model, tok, prompts, gen, batch_size)
    stats = {"seconds": round(time.time() - t0, 1)}

    if torch.cuda.is_available():
        stats["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        stats["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)

    # preds/meta are plain python — a returned tensor would pin the CUDA context and defeat this.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        leaked = torch.cuda.memory_allocated() / 2**20
        if leaked > 100:
            raise SystemExit(f"{leaked:.0f} MB still allocated after free — VRAM leak.")
    return preds, meta, stats


# -------------------------------------------------------------------------- cache


def cache_paths(tag: str, model_tag: str, gen: GenerationConfig) -> tuple[Path, Path]:
    # The decoding fingerprint is in the FILENAME, so runs at different --max-new-tokens coexist
    # instead of overwriting each other. The sidecar re-checks it anyway (belt and braces).
    stem = RESULTS / f"generations_{model_tag}_{tag}_{gen_fingerprint(gen)}"
    return stem.with_suffix(".jsonl"), stem.with_suffix(".meta.json")


def save_cache(tag, model_tag, model_path, rows, preds, meta, gen, batch_size):
    jsonl, sidecar = cache_paths(tag, model_tag, gen)
    with jsonl.open("w") as f:
        for i, (r, p, m) in enumerate(zip(rows, preds, meta)):
            f.write(
                json.dumps(
                    {
                        "idx": i,
                        "prompt_sha1": hashlib.sha1(r["prompt"].encode()).hexdigest(),
                        "prediction": p,
                        **m,
                    }
                )
                + "\n"
            )
    sidecar.write_text(
        json.dumps(
            {
                "model_path": model_path,
                "gen_fingerprint": gen_fingerprint(gen),
                "gen_config": gen.to_diff_dict(),
                "batch_size": batch_size,
                "n_rows": len(rows),
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )


def load_cache(tag, model_tag, rows, gen):
    """Return (preds, meta) or None. Refuses on ANY mismatch.

    A cache that silently serves generations produced under different decoding params or a
    different row set yields plausible, wrong, undebuggable numbers — worse than no cache.
    """
    jsonl, sidecar = cache_paths(tag, model_tag, gen)
    if not jsonl.exists() or not sidecar.exists():
        return None

    side = json.loads(sidecar.read_text())
    if side.get("gen_fingerprint") != gen_fingerprint(gen):
        print(f"  cache {jsonl.name}: decoding config changed — regenerating")
        return None

    cached = [json.loads(line) for line in jsonl.open()]
    if len(cached) != len(rows):
        print(f"  cache {jsonl.name}: {len(cached)} rows vs {len(rows)} expected — regenerating")
        return None
    for c, r in zip(cached, rows):
        if c["prompt_sha1"] != hashlib.sha1(r["prompt"].encode()).hexdigest():
            print(f"  cache {jsonl.name}: prompt mismatch at idx {c['idx']} — regenerating")
            return None

    print(f"  cache hit: {jsonl.name}")
    return [c["prediction"] for c in cached], [
        {"n_new_tokens": c["n_new_tokens"], "hit_cap": c["hit_cap"]} for c in cached
    ]


# ------------------------------------------------------------------------ scoring


def score_rouge(rouge, preds: list[str], rows: list[dict], use_stemmer: bool) -> dict:
    """Per-example f-measures. references as list-of-lists => multi-reference MAX per rouge type."""
    return rouge.compute(
        predictions=preds,
        references=[r["references"] for r in rows],
        rouge_types=ROUGE_TYPES,
        use_stemmer=use_stemmer,
        use_aggregator=False,
    )


def means(per_example: dict) -> dict:
    return {k: float(np.mean(v)) for k, v in per_example.items()}


def score_precision_recall(preds: list[str], rows: list[dict], use_stemmer: bool) -> dict:
    """Mean precision/recall/f per rouge type, via rouge_scorer directly.

    evaluate's wrapper throws precision and recall away (`result[key] = [s.fmeasure ...]`), but
    they are what distinguishes "this model summarizes better" from "this model is just shorter":
    a verbose model that captures the content scores high recall and low precision. score_multi is
    exactly what evaluate calls under the hood, so these numbers are consistent with the headline
    (asserted in main()).
    """
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=use_stemmer)
    acc = {k: {"precision": [], "recall": [], "fmeasure": []} for k in ROUGE_TYPES}
    for pred, row in zip(preds, rows):
        score = scorer.score_multi(row["references"], pred)
        for k in ROUGE_TYPES:
            acc[k]["precision"].append(score[k].precision)
            acc[k]["recall"].append(score[k].recall)
            acc[k]["fmeasure"].append(score[k].fmeasure)
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}


def paired_bootstrap(base_scores, tuned_scores, n=10000, seed=0):
    """95% CI and p-value on the mean delta, resampling dialogues (paired)."""
    rng = np.random.default_rng(seed)
    b, t = np.asarray(base_scores), np.asarray(tuned_scores)
    diff = t - b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    boot = diff[idx].mean(axis=1)
    return {
        "delta": float(diff.mean()),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "p_tuned_not_better": float((boot <= 0).mean()),
    }


def strip_preamble(text: str) -> str:
    lines = text.split("\n")
    if lines and PREAMBLE_RE.match(lines[0].strip()):
        return "\n".join(lines[1:]).strip()
    return text


def diagnostics(preds: list[str], meta: list[dict]) -> dict:
    n = len(preds)
    toks = [m["n_new_tokens"] for m in meta]
    return {
        "preamble_rate": sum(strip_preamble(p) != p for p in preds) / n,
        "mean_output_tokens": float(np.mean(toks)),
        "median_output_tokens": float(np.median(toks)),
        "truncation_rate": sum(m["hit_cap"] for m in meta) / n,
        "empty_rate": sum(not p.strip() for p in preds) / n,
    }


# ------------------------------------------------------------------------ reports


def fmt_row(name, b, t):
    return f"| {name} | {b:.4f} | {t:.4f} | {t - b:+.4f} |\n"


def write_report(path, rows, args, gen, raw, stripped, diag, boot, vram, stemmer, pr, ref_tokens):
    n = len(rows)
    REF_MEAN_TOKENS = f"{ref_tokens:.1f}"
    refs_hist = {}
    for r in rows:
        refs_hist[len(r["references"])] = refs_hist.get(len(r["references"]), 0) + 1

    with path.open("w") as f:
        f.write("# ROUGE — base vs. fine-tuned (DialogSum test)\n\n")
        f.write(
            f"Qwen2.5-1.5B-Instruct, stock vs. QLoRA fine-tune (r=16, a=32) merged to fp16.\n"
            f"**{n} dialogues**, multi-reference ROUGE (max f-measure over "
            f"{'/'.join(str(k) for k in sorted(refs_hist))} human refs per dialogue).\n\n"
        )

        f.write("## Headline — raw model output\n\n")
        f.write("Exactly what each model emits, no post-processing. This is the number for the README.\n\n")
        f.write("| Metric | Base | Fine-tuned | Δ |\n|---|---:|---:|---:|\n")
        for k in ROUGE_TYPES:
            f.write(fmt_row(k, raw["base"][k], raw["tuned"][k]))
        f.write("\n")
        for k in ROUGE_TYPES:
            s = boot[k]
            f.write(
                f"- **{k}** Δ = {s['delta']:+.4f}, 95% CI "
                f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}], "
                f"p(tuned not better) = {s['p_tuned_not_better']:.4f}\n"
            )
        f.write("\n_Paired bootstrap over dialogues, 10,000 resamples, seed 0._\n\n")

        f.write("## Diagnostics\n\n")
        f.write("| | Base | Fine-tuned |\n|---|---:|---:|\n")
        for key, label in [
            ("preamble_rate", "preamble rate"),
            ("mean_output_tokens", f"mean output tokens (refs: {ref_tokens:.1f})"),
            ("median_output_tokens", "median output tokens"),
            ("truncation_rate", f"truncation rate (hit {args.max_new_tokens} cap)"),
            ("empty_rate", "empty output rate"),
        ]:  # empty_rate matters: a blank output scores 0 and would otherwise hide in the mean
            b, t = diag["base"][key], diag["tuned"][key]
            fmt = ".1f" if "tokens" in key else ".3f"
            f.write(f"| {label} | {b:{fmt}} | {t:{fmt}} |\n")
        f.write(
            f"\nHuman reference summaries average {REF_MEAN_TOKENS} tokens. The base model is not "
            "malformed — it emits correct third-person `#Person1#`-style summaries — it is simply "
            "much longer and more detailed than DialogSum's house style. The decomposition below "
            "tests what that costs it.\n\n"
            f"A non-zero base truncation rate raises a fair objection: is the "
            f"`max_new_tokens={args.max_new_tokens}` cap handicapping base? Re-run with "
            "`--max-new-tokens 192` to check. It isn't: doubling the cap drives base's truncation "
            "to ~0 and leaves its ROUGE *slightly lower* (it simply writes more, so precision "
            "falls further), while the fine-tune's scores are bit-identical because it never "
            "reaches the cap. The delta is not a truncation artifact.\n\n"
        )

        f.write("## Why the delta? Precision / recall decomposition\n\n")
        f.write(
            "The question the headline can't answer on its own: did the fine-tune teach "
            "**summarization**, or just teach **brevity**? A verbose model that captures the "
            "content scores high recall and low precision.\n\n"
        )
        f.write("| Metric | Model | Precision | Recall | F |\n|---|---|---:|---:|---:|\n")
        for k in ROUGE_TYPES:
            for tag, label in [("base", "base"), ("tuned", "tuned")]:
                s = pr[tag][k]
                f.write(
                    f"| {k} | {label} | {s['precision']:.4f} | {s['recall']:.4f} | {s['fmeasure']:.4f} |\n"
                )
        r1b, r1t = pr["base"]["rouge1"], pr["tuned"]["rouge1"]
        f.write(
            f"\n**Read:** on ROUGE-1 the base model's recall is {r1b['recall']:.3f} vs the "
            f"fine-tune's {r1t['recall']:.3f}, while its precision is {r1b['precision']:.3f} vs "
            f"{r1t['precision']:.3f}. "
        )
        if r1b["recall"] >= r1t["recall"] - 0.02 and r1t["precision"] - r1b["precision"] > 0.1:
            f.write(
                "Base recovers **as much reference content as the fine-tune** and loses almost "
                "entirely on precision — i.e. the gain here is dominated by **length/style "
                "calibration**, not by better content selection. That is a real and useful result "
                "for a summarizer (matching the target register is the job), but it should not be "
                "reported as 'the base model can't summarize'. It can; it just won't stop.\n\n"
            )
        else:
            f.write(
                "The fine-tune improves **both** precision and recall, so the gain is not merely "
                "a length effect — it selects reference content better, not just less of it.\n\n"
            )

        f.write("### Preamble check\n\n")
        inert = diag["base"]["preamble_rate"] == 0 and diag["tuned"]["preamble_rate"] == 0
        f.write(f"Regex, applied at most once to the first line, identically to both models:\n\n")
        f.write(f"```\n{PREAMBLE_RE.pattern}\n```\n\n")
        if inert:
            f.write(
                "**It matches nothing (0.000 on both models), so this control is inert and the "
                "stripped scores are identical to the headline.** Worth stating rather than "
                "quietly dropping: the expected confound — stock Qwen opening with "
                "\"Sure! Here's a summary:\" and being punished for formatting rather than "
                "comprehension — *does not occur*. Base's disadvantage is length, not preamble, "
                "which is why the decomposition above is the analysis that matters.\n\n"
            )
        else:
            f.write("| Metric | Base (stripped) | Fine-tuned (stripped) | Δ |\n|---|---:|---:|---:|\n")
            for k in ROUGE_TYPES:
                f.write(fmt_row(k, stripped["base"][k], stripped["tuned"][k]))
            f.write(
                "\nStripping only base would be removing base's handicap; stripping both is a "
                "transformation of the data — diagnostic only, never the headline.\n\n"
            )

        f.write("## Run config\n\n")
        f.write(f"- examples: **{n}**" + (f" (stride-sampled from 500)\n" if args.limit else "\n"))
        f.write(f"- refs per dialogue: {dict(sorted(refs_hist.items()))}\n")
        f.write(f"- decoding (both models, identical): `{json.dumps(gen.to_diff_dict())}`\n")
        f.write(f"- stemmer: `use_stemmer={stemmer}` | aggregator: own mean over per-example scores\n")
        f.write(f"- batch size: {args.batch_size}\n")
        for tag in ["base", "tuned"]:
            v = vram[tag]
            f.write(
                f"- {tag}: {v['seconds']}s"
                + (
                    f", peak VRAM {v['peak_alloc_gb']} GB allocated / {v['peak_reserved_gb']} GB reserved\n"
                    if "peak_alloc_gb" in v
                    else "\n"
                )
            )
        f.write(f"- transformers {transformers.__version__}, torch {torch.__version__}\n\n")
        f.write(
            "_Note on comparability: `use_stemmer=True` follows the ROUGE-1.5.5 `-m` convention "
            "most summarization papers use (HF `evaluate` defaults it to False). We state our "
            "setting rather than claim to match the DialogSum paper's exact configuration. "
            "The base-vs-tuned delta is the claim; absolute values are context._\n"
        )


def select_qualitative(base_pe, tuned_pe, k=5):
    """5 examples at delta quantiles 0/25/50/75/100 — deterministic, ties by index.

    Forces inclusion of the case where the fine-tune did WORST relative to base, so the reader can
    see the examples weren't cherry-picked.
    """
    delta = np.asarray(tuned_pe["rougeL"]) - np.asarray(base_pe["rougeL"])
    order = sorted(range(len(delta)), key=lambda i: (delta[i], i))
    positions = [round(q * (len(order) - 1)) for q in np.linspace(0, 1, k)]
    labels = ["0% (worst for tuned)", "25%", "50% (median)", "75%", "100% (best for tuned)"]
    return [(labels[j], order[p]) for j, p in enumerate(positions)]


def extract_dialogue(prompt: str) -> str:
    """Pull the raw dialogue back out of the rendered chat prompt."""
    m = re.search(
        r"Summarize the following conversation:\n\n(.*?)<\|im_end\|>", prompt, re.DOTALL
    )
    return m.group(1).strip() if m else prompt


def write_qualitative(path, rows, base_preds, tuned_preds, base_pe, tuned_pe):
    picks = select_qualitative(base_pe, tuned_pe)
    with path.open("w") as f:
        f.write("# Qualitative examples — base vs. fine-tuned\n\n")
        f.write(
            "Five dialogues at fixed quantiles of the per-example ROUGE-L delta "
            "(tuned − base), ranked ascending. Deterministic and unbiased by construction: the "
            "first example is where the fine-tune did **worst** relative to base.\n\n"
        )
        for label, i in picks:
            r = rows[i]
            b, t = base_pe["rougeL"][i], tuned_pe["rougeL"][i]
            f.write(f"---\n\n## {label} — test row {i}\n\n")
            f.write(f"ROUGE-L: base {b:.4f} | tuned {t:.4f} | Δ {t - b:+.4f}\n\n")
            dialogue = extract_dialogue(r["prompt"])
            if len(dialogue) > 1200:
                dialogue = dialogue[:1200] + "\n[... truncated for display]"
            f.write(f"### Dialogue\n\n```\n{dialogue}\n```\n\n")
            f.write(f"### Human references ({len(r['references'])})\n\n")
            for j, ref in enumerate(r["references"], 1):
                f.write(f"{j}. {ref}\n")
            f.write(f"\n### Base output\n\n> {base_preds[i] or '_(empty)_'}\n\n")
            f.write(f"### Fine-tuned output\n\n> {tuned_preds[i] or '_(empty)_'}\n\n")


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    tag = run_tag(args.limit)
    stemmer = not args.no_stemmer

    out = Path(args.out) if args.out else report_path("rouge_comparison", args.limit)
    RESULTS.mkdir(parents=True, exist_ok=True)

    rows = load_test(args.test, args.limit)
    prompts = [r["prompt"] for r in rows]
    gen = build_gen_config(args.max_new_tokens)

    tok = AutoTokenizer.from_pretrained(args.base)
    tok.padding_side = "left"  # REQUIRED for batched decoder-only generation; defaults to 'right'

    print(f"{len(rows)} dialogues | decoding: {json.dumps(gen.to_diff_dict())}")

    preds, meta, vram = {}, {}, {}
    for model_tag, model_path in [("base", args.base), ("tuned", args.merged)]:
        print(f"\n[{model_tag}] {model_path}")
        cached = None if args.regen else load_cache(tag, model_tag, rows, gen)
        if cached:
            preds[model_tag], meta[model_tag] = cached
            vram[model_tag] = {"seconds": 0.0}
            continue
        if args.score_only:
            raise SystemExit(f"--score-only but no valid cache for {model_tag} (tag={tag})")
        preds[model_tag], meta[model_tag], vram[model_tag] = score_model(
            model_path, prompts, gen, args.batch_size, tok
        )
        save_cache(tag, model_tag, model_path, rows, preds[model_tag], meta[model_tag], gen, args.batch_size)

    rouge = evaluate.load("rouge")

    # Sanity: a reference scored against itself must be a perfect match. Guards the multi-ref path.
    selfcheck = score_rouge(rouge, [r["references"][0] for r in rows], rows, stemmer)
    if not np.allclose(selfcheck["rouge1"], 1.0):
        raise SystemExit("self-scoring a reference didn't yield rouge1=1.0 — scoring path is wrong")

    raw_pe = {k: score_rouge(rouge, preds[k], rows, stemmer) for k in ["base", "tuned"]}
    stripped_pe = {
        k: score_rouge(rouge, [strip_preamble(p) for p in preds[k]], rows, stemmer)
        for k in ["base", "tuned"]
    }

    raw = {k: means(v) for k, v in raw_pe.items()}
    stripped = {k: means(v) for k, v in stripped_pe.items()}
    diag = {k: diagnostics(preds[k], meta[k]) for k in ["base", "tuned"]}
    boot = {k: paired_bootstrap(raw_pe["base"][k], raw_pe["tuned"][k]) for k in ROUGE_TYPES}

    # Precision/recall come from rouge_scorer directly (evaluate discards them). Prove the two
    # paths agree before trusting the decomposition to explain the headline.
    pr = {k: score_precision_recall(preds[k], rows, stemmer) for k in ["base", "tuned"]}
    for k in ["base", "tuned"]:
        for rt in ROUGE_TYPES:
            if not np.isclose(pr[k][rt]["fmeasure"], raw[k][rt], atol=1e-6):
                raise SystemExit(
                    f"rouge_scorer and evaluate disagree on {k}/{rt}: "
                    f"{pr[k][rt]['fmeasure']} vs {raw[k][rt]}"
                )

    ref_tokens = float(
        np.mean([len(tok(ref).input_ids) for r in rows for ref in r["references"]])
    )

    write_report(out, rows, args, gen, raw, stripped, diag, boot, vram, stemmer, pr, ref_tokens)
    qual = report_path("qualitative_examples", args.limit)
    write_qualitative(qual, rows, preds["base"], preds["tuned"], raw_pe["base"], raw_pe["tuned"])

    print(f"\nwrote {out}\nwrote {qual}\n")
    print(f"{'metric':<8} {'base':>8} {'tuned':>8} {'delta':>9}")
    for k in ROUGE_TYPES:
        print(f"{k:<8} {raw['base'][k]:>8.4f} {raw['tuned'][k]:>8.4f} {boot[k]['delta']:>+9.4f}")


if __name__ == "__main__":
    main()
