# Quantization cost — fp16 merged → 4-bit MLX

What the 4-bit MLX conversion costs, measured directly. The fine-tuned model was scored twice
on the **same 500-dialogue DialogSum test split**, with the **same decoding** (greedy,
`max_new_tokens/max_tokens=96`, no repetition penalty, stop on `<|im_end|>`/`<|endoftext|>`)
and the **same multi-reference max-ROUGE scoring** (`use_stemmer=True`, own mean over
per-example scores — `train/rouge_common.py`, shared by both evaluators). The only difference
is the model: fp16 merged (transformers, `train/eval_rouge.py`) vs 4-bit MLX (mlx_lm,
`convert/eval_mlx_rouge.py`). So the delta below is quantization + runtime, isolated from any
data/decoding/scoring change.

## ROUGE (headline f-measure)

| Metric | fp16 merged | 4-bit MLX | Δ (abs) | Δ (rel) |
|--------|------------:|----------:|--------:|--------:|
| ROUGE-1 | 0.5581 | 0.5433 | −0.0148 | −2.7% |
| ROUGE-2 | 0.3101 | 0.2905 | −0.0196 | −6.3% |
| ROUGE-L | 0.4808 | 0.4622 | −0.0186 | −3.9% |

## Where the loss comes from — precision / recall

| Metric | | fp16 P | fp16 R | 4-bit P | 4-bit R |
|--------|---|------:|------:|-------:|-------:|
| ROUGE-1 | | 0.5402 | 0.6042 | 0.5171 | 0.6008 |
| ROUGE-2 | | 0.3022 | 0.3365 | 0.2751 | 0.3256 |
| ROUGE-L | | 0.4643 | 0.5221 | 0.4370 | 0.5150 |

**Recall barely moves** (ROUGE-1 recall 0.604 → 0.601); the drop is almost entirely
**precision**. The 4-bit model recovers the same reference content, it just writes slightly
longer/looser summaries — mean output grows 33.6 → 35.7 tokens (refs average 27.8). This is a
register drift, not a comprehension failure.

## Diagnostics

| | fp16 merged | 4-bit MLX |
|---|------------:|----------:|
| mean output tokens | 33.6 | 35.7 |
| median output tokens | 30.5 | 33.0 |
| truncation rate (hit 96 cap) | 0.002 | 0.004 |
| empty output rate | 0.000 | 0.000 |
| preamble rate | 0.000 | 0.000 |

No degeneration: no repetition loops, no truncation spike, no empty or malformed outputs.

## Size

| | fp16 merged | 4-bit MLX |
|---|------------:|----------:|
| on disk | ~3.1 GB | **847 MB** |
| bits / weight | 16 | 4.501 |

`--q-bits 4 --q-group-size 64`; the effective 4.5 bits/weight reflects embeddings and norms
kept at higher precision by the group-quantization default. ~3.7× smaller.

## Verdict

**Accepted at 4-bit.** The cost is a mild, expected ~1.5–2 ROUGE-point drop (largest on
ROUGE-2, −6.3% relative) with clean diagnostics and coherent, terse output — no quality cliff.
At ~847 MB it fits the iPhone 14 Pro (A16, 6 GB) with headroom, which is the entire point of
the project: a task-scoped summarizer running below Apple's A17 Pro / 8 GB on-device line.
8-bit (~1.6 GB, near-lossless) and mixed 4/6-bit recipes were considered and declined in favor
of the smaller footprint.

_4-bit numbers: `results/rouge_mlx_4bit.md`. fp16 base-vs-tuned report:
`results/rouge_comparison.md`._
