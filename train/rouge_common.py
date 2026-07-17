"""
Shared, model-agnostic ROUGE scoring + diagnostics helpers.

Extracted verbatim from eval_rouge.py so the Stage 4 MLX evaluation
(convert/eval_mlx_rouge.py) can reuse the EXACT scoring path rather than
duplicating it. Nothing here imports torch or transformers: everything operates
on `preds: list[str]`, `rows` (with a `references` list), and `meta` dicts —
so both the transformers (Stage 3) and MLX (Stage 4) generation paths feed the
same scorers and the numbers stay directly comparable.

The three verified traps this file preserves (see eval_rouge.py's docstring for
the full account):

2. MULTI-REFERENCE. references passed as a LIST OF LISTS => evaluate dispatches
   to rouge_scorer.score_multi, taking the MAX f-measure per rouge type.
3. AGGREGATION. use_aggregator=False + our own mean — deterministic, matches the
   literature's sample-mean convention, and exposes per-example scores.
"""

import json
import re

import numpy as np
from rouge_score import rouge_scorer

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
    (asserted by the caller).
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
