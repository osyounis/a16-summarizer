"""
Stage 2 — QLoRA fine-tune.

4-bit base (bitsandbytes nf4) + LoRA adapter via PEFT/TRL. Output is a small adapter
(a few MB), not a full model. Run on: RTX 3080 10 GB box (WSL).

Verified against the installed stack (transformers 5.13.1, trl 1.8.0, peft 0.19.1,
bitsandbytes 0.49.2, torch 2.13.0):
  - Data is (prompt, completion) JSONL from Stage 1. `prompt` ends with the
    `<|im_start|>assistant\n` header; `completion` is the bare summary (no EOS).
  - TRL auto-detects prompt+completion and enables completion-only loss (prompt
    tokens masked from the loss). We do NOT set `completion_only_loss`.
  - TRL auto-appends the tokenizer EOS to `completion`. Qwen's eos is `<|im_end|>`
    (id 151645) — exactly the app's stop token, so training matches inference.
  - Passing an already-4-bit model + `peft_config` lets TRL wrap it (kbit prep,
    `enable_input_require_grads` for grad checkpointing, adapter cast to bf16). We
    do NOT call `prepare_model_for_kbit_training` / `get_peft_model` ourselves.
  - API: `max_length` (not `max_seq_length`), `eval_strategy` (not `evaluation_strategy`).

VRAM (~9 GB usable): bs=2 x grad_accum=8 (eff batch 16), max_length=1024, gradient
checkpointing on -> expected peak ~6-8 GB. If OOM: max_length=768, then bs=1/accum=16.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_OUT = "out/adapter"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[  # Qwen2 arch: attention + MLP projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    ds = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "validation": "data/validation.jsonl",
    })

    cfg = SFTConfig(
        output_dir="out/train",
        num_train_epochs=2,
        per_device_train_batch_size=2,      # eff batch = 2 * 8 = 16
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        max_length=1024,                    # covers ~99.5% of prompt+completion+eos
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,                    # ~3% of ~1558 total steps
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",           # paged optimizer smooths OOM spikes
        logging_steps=20,                   # train loss
        eval_strategy="steps",
        eval_steps=200,                     # validation loss (~8 points over the run)
        save_strategy="epoch",
        report_to="none",
        # completion_only_loss: left unset -> TRL auto-enables it for prompt/completion.
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(ADAPTER_OUT)
    print(f"adapter saved -> {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
