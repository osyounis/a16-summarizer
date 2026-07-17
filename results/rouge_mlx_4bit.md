# ROUGE — 4-bit MLX (DialogSum test)

Qwen2.5-1.5B-Instruct QLoRA fine-tune (r=16, a=32), merged to fp16, then quantized to **4-bit MLX** (q-bits 4, group size 64).
**500 dialogues**, multi-reference ROUGE (max f-measure over 2/3 human refs per dialogue).

Same decoding and same scoring as the fp16 run (`results/rouge_comparison.md`), so these numbers are directly comparable. Quantization delta vs fp16: `results/quantization_delta.md`.

## Headline — raw model output

| Metric | 4-bit MLX |
|---|---:|
| rouge1 | 0.5433 |
| rouge2 | 0.2905 |
| rougeL | 0.4622 |

## Diagnostics

| | 4-bit MLX |
|---|---:|
| preamble rate | 0.000 |
| mean output tokens (refs: 27.8) | 35.7 |
| median output tokens | 33.0 |
| truncation rate (hit 96 cap) | 0.004 |
| empty output rate | 0.000 |

## Precision / recall decomposition

| Metric | Precision | Recall | F |
|---|---:|---:|---:|
| rouge1 | 0.5171 | 0.6008 | 0.5433 |
| rouge2 | 0.2751 | 0.3256 | 0.2905 |
| rougeL | 0.4370 | 0.5150 | 0.4622 |

## Run config

- examples: **500**
- refs per dialogue: {2: 10, 3: 490}
- decoding (matches fp16): greedy (`temp=0.0`, no top_p/top_k), `max_tokens=96`, no repetition penalty, stop on `{151645, 151643}`
- prompt: Stage 1 chat template, fed verbatim (ends `<|im_start|>assistant\n`)
- stemmer: `use_stemmer=True` | aggregator: own mean over per-example scores
- model: `mlx_model` (4-bit MLX, 828 MB safetensors)
- generation: mlx_lm `stream_generate`; scoring reuses `train/rouge_common.py`
