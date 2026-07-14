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

- **Code (this repo):** `<github-user>/a16-summarizer`
- **LoRA adapter:** `https://huggingface.co/<hf-user>/a16-summarizer-lora`
- **Quantized MLX model (loaded by the app):** `https://huggingface.co/<hf-user>/a16-summarizer-mlx-4bit`

> Replace `<github-user>` / `<hf-user>` before first push.

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
python convert/to_mlx.py --upload-repo <hf-user>/a16-summarizer-mlx-4bit

# App: see app/README.md
```

---

## Model Card (fill in after eval — this is the portfolio artifact)

**Model:** `<hf-user>/a16-summarizer-mlx-4bit`
**Base model:** `Qwen/Qwen2.5-1.5B-Instruct` (Apache 2.0)
**Fine-tuned on:** DialogSum (MIT), dialogue → abstractive summary
**Method:** QLoRA, rank `<r>`, alpha `<a>`, `<n>` epochs
**Quantization:** 4-bit (MLX), group size `<g>`

### Results

| Metric | Base (Qwen2.5-1.5B) | Fine-tuned | Δ |
|--------|--------------------:|-----------:|---:|
| ROUGE-1 | `TODO` | `TODO` | `TODO` |
| ROUGE-2 | `TODO` | `TODO` | `TODO` |
| ROUGE-L | `TODO` | `TODO` | `TODO` |

### On-device (iPhone 14 Pro, A16, 6 GB)

| Measure | Value |
|---------|------:|
| Model size on disk | `TODO` |
| Prefill / decode tokens/sec | `TODO` |
| Peak memory | `TODO` |

### Intended use & limits

Task-specific dialogue summarization for demonstration. Not a general chatbot; inherits
Qwen2.5 limitations plus quantization quality loss. English only (DialogSum).

## License

Code: MIT (see `LICENSE`). Model derivative: Apache 2.0 (inherits Qwen2.5). See `NOTICE`.
