# On-Device Dialogue Summarizer

A LoRA-fine-tuned small language model that runs **entirely on an iPhone 14 Pro (A16)** —
a device Apple deemed too constrained for Apple Intelligence. This project reproduces
Apple's own on-device recipe (small base model + LoRA task adapter + aggressive
quantization) with a bring-your-own model, and ships it running natively on the phone.

> **Thesis:** Apple draws the on-device-LLM line at the A17 Pro / 8 GB RAM. This shows a
> scoped, task-specialized summarizer running below that line on an A16 / 6 GB device —
> and documents *why* the tradeoffs (model size, bit-width, memory) make that possible.

## What this demonstrates

- **PEFT / LoRA fine-tuning** (QLoRA on a single RTX 3080)
- **Rigorous evaluation** — ROUGE, base vs. fine-tuned, on a held-out test split
- **Quantization + on-device conversion** (4-bit, MLX)
- **On-device deployment** — SwiftUI + MLX Swift, running on the Neural Engine / GPU
- **On-device profiling** — tokens/sec, peak memory, model size on real hardware

## Stack (chosen for clean licensing + on-device fit)

| Piece | Choice | License | Why |
|-------|--------|---------|-----|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` | Apache 2.0 | Clean license (the 3B is **not** Apache); ~1 GB at 4-bit fits 6 GB RAM |
| Dataset | DialogSum | MIT | Dialogue summarization; repo-safe (unlike SAMSum's CC BY-NC-ND) |
| Training | PEFT + TRL + bitsandbytes | — | QLoRA on the RTX 3080 (CUDA) |
| Runtime | MLX Swift | — | Apple-native, runs on A16, HF-native model loading |
| Stretch | Core ML | — | Higher Apple credibility; attempt only after MLX works |

## Repo ↔ Hub layout

Code lives here on GitHub. **Weights and data never touch this repo.** The trained
artifacts live on the Hugging Face Hub and are referenced by ID:

- **Code (this repo):** `osyounis/a16-summarizer`
- **LoRA adapter:** `https://huggingface.co/osyounis/a16-summarizer-lora`
- **Quantized MLX model (loaded by the app):** `https://huggingface.co/osyounis/a16-summarizer-mlx-4bit`

## Quickstart

```bash
# On the RTX 3080 box (CUDA):
pip install -r requirements-train.txt
python train/prepare_data.py
python train/train_lora.py
python train/merge.py
python train/eval_rouge.py        # writes results/rouge_comparison.md

# On the M2 Mac (MLX is Apple-silicon only):
pip install -r requirements-convert.txt
python convert/to_mlx.py --upload-repo osyounis/a16-summarizer-mlx-4bit

# App: see app/README.md
```

---

## Model Card (this is the portfolio artifact)

**Model:** `osyounis/a16-summarizer-mlx-4bit`
**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0)
**Fine-tuned on:** DialogSum (MIT), dialogue → abstractive summary
**Method:** QLoRA, rank 16, alpha 32, 2 epochs
**Quantization:** 4-bit (MLX), group size 64 (4.501 effective bits/weight)

### Results

DialogSum test split, **500 dialogues**, multi-reference ROUGE (max f-measure over the 3 human
reference summaries per dialogue; 10 dialogues have 2 after identical annotations are deduped).
All models decoded identically: greedy, `max_new_tokens=96`.

| Metric | Base (Qwen2.5-1.5B) | Fine-tuned (fp16 merged) | 4-bit MLX (on-device) | Δ tuned vs base |
|--------|--------------------:|-------------------------:|----------------------:|----------------:|
| ROUGE-1 | 0.3706 | **0.5581** | 0.5433 | **+0.1875** |
| ROUGE-2 | 0.1498 | **0.3101** | 0.2905 | **+0.1602** |
| ROUGE-L | 0.2889 | **0.4808** | 0.4622 | **+0.1919** |

All three tuned-vs-base deltas have a 95% CI excluding zero (paired bootstrap over dialogues,
10k resamples). The **4-bit MLX** column is the model the app actually ships: quantization costs
~1.5–2 ROUGE points (largest on ROUGE-2, −6.3% relative), with no degeneration — recall is
essentially unchanged, the small loss is precision from slightly looser output. The 4-bit model
is scored with identical decoding and scoring, so the columns are directly comparable. Reports:
[`results/rouge_comparison.md`](results/rouge_comparison.md) (base vs fp16),
[`results/rouge_mlx_4bit.md`](results/rouge_mlx_4bit.md) and
[`results/quantization_delta.md`](results/quantization_delta.md) (fp16 vs 4-bit).

**What the delta actually is:** register calibration, not comprehension. The base model's ROUGE-1
*recall* (0.616) slightly **exceeds** the fine-tune's (0.604) — it recovers the reference content
fine, and emits well-formed third-person summaries unprompted. But it writes ~68 tokens against
27.8-token references, so its precision collapses (0.277 vs 0.540). The fine-tune's gain is
learning DialogSum's length and house style. That is the job for a task-scoped summarizer, but
it is not "the base model can't summarize" — it can; it won't stop. Side-by-side outputs,
including the case where the fine-tune does *worst*, are in
[`results/qualitative_examples.md`](results/qualitative_examples.md).

### On-device (iPhone 14 Pro, A16, 6 GB)

| Measure | Value |
|---------|------:|
| Model size on disk | 847 MB (4-bit MLX) |
| Prefill / decode tokens/sec | `TODO` |
| Peak memory | `TODO` |

### Intended use & limits

Task-specific dialogue summarization for demonstration. Not a general chatbot; inherits
Qwen2.5 limitations plus quantization quality loss. English only (DialogSum).

## License

Code: MIT (see `LICENSE`). Model derivative: Apache 2.0 (inherits Qwen2.5). See `NOTICE`.
