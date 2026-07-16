"""
Stage 3a — Merge the LoRA adapter into the base model.

Produces a standalone fp16 merged model that eval_rouge.py scores and the MLX converter consumes.
Run on: RTX 3080 box. Merged model is git-ignored — host on HF Hub if you want it shared.

This is a PURE merge: it folds the adapter weights in and saves, and deliberately does NOT touch
the generation config. `out/merged/generation_config.json` therefore inherits Qwen2.5-Instruct's
shipped sampling defaults (do_sample=true, temperature=0.7, repetition_penalty=1.1).

  >>> That is only safe because eval_rouge.py builds an explicit GenerationConfig and passes it to
  >>> BOTH models, so neither ever reads its on-disk defaults. If you ever "simplify" eval to drop
  >>> that, base and merged will silently decode differently and the comparison becomes invalid
  >>> with no error. The two files must be read together.

fp16 is a deliberate downcast — the base config is bf16. Every bf16 value here is inside fp16's
range, Stage 4 requantizes to 4-bit anyway, and fp16 is the lower-risk dtype for the Apple-silicon
target (bf16 on A16-class Metal is emulated at best).
"""

from pathlib import Path
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "out/adapter"
MERGED_OUT = "out/merged"

# The weight we fingerprint to prove the merge actually changed something.
PROBE_LAYER = 0


def parse_args():
    p = argparse.ArgumentParser(description="Merge the Stage 2 LoRA adapter into the base model.")
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--adapter", default=ADAPTER)
    p.add_argument("--out", default=MERGED_OUT)
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Merge on GPU (~10s) or CPU (~1-3min; fp16 matmul is slow there, no accuracy benefit).",
    )
    return p.parse_args()


def probe_weight(model) -> torch.Tensor:
    """The q_proj weight we compare before/after merging, detached to CPU fp32."""
    w = model.model.layers[PROBE_LAYER].self_attn.q_proj.weight
    return w.detach().to("cpu", torch.float32).clone()


def check_adapter(adapter: str, base: str) -> None:
    cfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
    trained_on = cfg.get("base_model_name_or_path")
    if trained_on != base:
        raise SystemExit(
            f"adapter was trained on {trained_on!r} but you're merging into {base!r} — refusing."
        )
    print(f"adapter: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} on {trained_on}")


def verify(out: Path, before: torch.Tensor, after: torch.Tensor) -> None:
    """Fail loudly on the silent failures that would masquerade as 'the fine-tune did nothing'."""
    for name in ["config.json", "generation_config.json"]:
        if not (out / name).exists():
            raise SystemExit(f"{out / name} missing — Stage 4's mlx_lm.convert needs it.")
    shards = list(out.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors written to {out}")

    # A silently unapplied adapter presents downstream as a flat ROUGE delta — a Stage-3-shaped
    # symptom with a Stage-3a cause. Catch it here instead.
    if torch.allclose(before, after):
        raise SystemExit("merged q_proj is identical to base q_proj — the adapter was NOT applied.")

    delta = (after - before).abs()
    size_gb = sum(f.stat().st_size for f in out.glob("*")) / 2**30
    print(f"merge verified: layer{PROBE_LAYER}.q_proj changed, max|dw|={delta.max():.5f}")
    print(f"wrote {len(shards)} shard(s), {size_gb:.2f} GB total -> {out}")


def main():
    args = parse_args()
    out = Path(args.out)
    device_map = {"": 0} if args.device == "cuda" else "cpu"

    check_adapter(args.adapter, args.base)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.float16, device_map=device_map
    )
    before = probe_weight(base)

    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    after = probe_weight(model)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    # Tokenizer from BASE, not out/adapter: both tokenize all 500 test prompts identically, but
    # base is the canonical source and out/adapter's copy is just a training by-product.
    tokenizer.save_pretrained(out)

    verify(out, before, after)
    if args.device == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
