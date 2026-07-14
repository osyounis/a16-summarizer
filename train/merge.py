"""
Stage 3a — Merge the LoRA adapter into the base model.

Produces a standalone (full-precision) merged model that the MLX converter consumes.
Run on: RTX 3080 box. Merged model is git-ignored — host on HF Hub if you want it shared.

TODO / verify:
  - Merge in fp16/bf16 on CPU or GPU (NOT under 4-bit quantization — load base in fp16 here).
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "out/adapter"
MERGED_OUT = "out/merged"


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_OUT)
    tokenizer.save_pretrained(MERGED_OUT)
    print(f"merged model -> {MERGED_OUT}")


if __name__ == "__main__":
    main()
