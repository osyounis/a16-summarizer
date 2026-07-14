"""
Stage 2 — QLoRA fine-tune.

4-bit base (bitsandbytes) + LoRA adapter via PEFT/TRL. Output is a small adapter,
not a full model. Run on: RTX 3080 box.

TODO / verify:
  - Confirm current TRL SFTTrainer / SFTConfig API (it changes between versions).
  - Confirm BitsAndBytesConfig 4-bit (nf4) setup for the current transformers version.
  - Tune: start r=16, alpha=32, 1-3 epochs. Adjust batch size to fit 10 GB VRAM.
  - Pick target_modules for Qwen2.5 (q/k/v/o + gate/up/down proj).
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
    target_modules=[  # TODO: verify names for Qwen2.5
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto"
    )

    ds = load_dataset("json", data_files={
        "train": "data/train.jsonl",
        "validation": "data/validation.jsonl",
    })

    cfg = SFTConfig(
        output_dir="out/train",
        num_train_epochs=2,
        per_device_train_batch_size=4,     # TODO: tune for 10 GB VRAM
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=20,
        eval_strategy="epoch",
        bf16=True,
        # TODO: confirm how this TRL version consumes prompt/completion fields.
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
