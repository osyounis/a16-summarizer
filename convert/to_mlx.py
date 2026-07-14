"""
Stage 4 — Quantize + convert the merged model to 4-bit MLX, optionally push to HF Hub.

Run on: M2 Mac ONLY (MLX is Apple-silicon). Point --merged at a local copy of out/merged
(scp'd from the 3080 box) or an HF repo id if you pushed the merged model there.

The convert step and the publish step are one command: mlx_lm.convert can upload directly.

TODO / verify:
  - Confirm current mlx_lm.convert flags: `python -m mlx_lm convert --help`
    (expected: --hf-path, --mlx-path, -q, --q-bits, --q-group-size, --upload-repo).
  - After converting, sanity-check with: python -m mlx_lm generate --model <mlx-path> --prompt "..."
"""

import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="out/merged",
                    help="local path or HF id of the merged fp16 model")
    ap.add_argument("--mlx-path", default="mlx_model")
    ap.add_argument("--q-bits", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=64)
    ap.add_argument("--upload-repo", default=None,
                    help="e.g. <hf-user>/a16-summarizer-mlx-4bit")
    args = ap.parse_args()

    cmd = [
        "python", "-m", "mlx_lm", "convert",
        "--hf-path", args.merged,
        "--mlx-path", args.mlx_path,
        "-q",
        "--q-bits", str(args.q_bits),
        "--q-group-size", str(args.q_group_size),
    ]
    if args.upload_repo:
        cmd += ["--upload-repo", args.upload_repo]

    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nMLX model at {args.mlx_path}")
    print("Sanity check:")
    print(f'  python -m mlx_lm generate --model {args.mlx_path} '
          f'--prompt "Summarize the following conversation:\\n\\nA: ..."')


if __name__ == "__main__":
    main()
