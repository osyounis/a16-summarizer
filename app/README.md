# app — a16 Summarizer (SwiftUI + MLX Swift, on-device)

A minimal, standalone SwiftUI app that runs the
[`osyounis/a16-summarizer-mlx-4bit`](https://huggingface.co/osyounis/a16-summarizer-mlx-4bit)
model **entirely on-device** (iPhone, Apple GPU via MLX Swift). Paste a conversation, tap
**Summarize**, get a concise third-person summary — no server, no network after the
first-launch model download.

## What's here

```
app/
├── project.yml                 # XcodeGen spec (source of truth for the project)
├── A16Summarizer.xcodeproj     # generated, committed so it's buildable without XcodeGen
└── A16Summarizer/
    ├── A16SummarizerApp.swift  # @main entry; sets MLX cache limit
    ├── Summarizer.swift        # model load + generate glue, exposes tokens/sec
    ├── ContentView.swift       # paste → Summarize → summary + stats UI
    ├── A16Summarizer.entitlements
    └── Assets.xcassets/
```

Three small Swift files — everything model-specific lives in `Summarizer.swift`.

## Train / inference match (why summaries match the reported ROUGE)

`Summarizer.swift` reproduces the exact setup from training/eval:

- **Prompt** (from `train/prepare_data.py`): system message
  *"You are a helpful assistant that writes a concise, third-person summary of a conversation."*
  \+ user *"Summarize the following conversation:\n\n{dialogue}"*, wrapped by the Qwen chat
  template (`UserInput` → `prepare(input:)`).
- **Decoding** (from `convert/eval_mlx_rouge.py`): greedy, `temperature: 0` (→ `ArgMaxSampler`),
  `maxTokens: 96`, stop on `<|im_end|>` (`extraEOSTokens`).

## Build & run

Requirements: Xcode 16+ (built with Xcode 26), a **physical iPhone**. MLX also needs the
Metal Toolchain once per machine: `xcodebuild -downloadComponent MetalToolchain`.

1. Open `A16Summarizer.xcodeproj` in Xcode.
2. Target **A16Summarizer** → **Signing & Capabilities** → set your **Team** (Automatic signing).
   The bundle id is `com.osyounis.a16summarizer` — change the reverse-DNS prefix to your own.
3. Select your iPhone as the run destination → **⌘R**.
   - On first build, Xcode prompts to **Trust & Enable** the `mlx-swift-lm` Swift macros — accept (one-time).
   - On first launch, the app downloads the model (~847 MB) from the Hub over Wi-Fi.

Regenerate the project after editing `project.yml`: `brew install xcodegen && cd app && xcodegen generate`.

> **The iOS Simulator will not run this.** MLX needs a real Metal GPU; on the simulator
> `mlx::core::metal::Device` fails to initialize and the app aborts at launch. Build and run
> on the physical device — which is the point of the exercise anyway.

## Entitlements

Minimal, device-run only (no App Store / distribution scaffolding):

- `com.apple.developer.kernel.increased-memory-limit` — raises the process memory ceiling
  (~5.25 GB on a 6 GB device) so the 4-bit model has comfortable headroom.
- `com.apple.security.network.client` — first-launch Hub download (a macOS-sandbox key; iOS ignores it).

## Measured on iPhone 14 Pro (A16, 6 GB)

| Measure | Value |
|---|---|
| Model size on disk | 847 MB |
| Prefill / decode tokens/sec | ~137 / 44.4 |
| Peak memory | 1.05 GB (831 MB active) |

See the repo [`README.md`](../README.md) model card and `results/hero_screen.PNG`.

## Attribution

Structure adapted from Apple's `ml-explore/mlx-swift-examples` (`LLMBasic`), MIT-licensed —
see the repo `NOTICE`. The model-specific code, prompt formatting, and UI are original.

## Stretch: Core ML (Stage 6)

Only after MLX works. Convert with `coremltools`, handle the stateful KV-cache path, run on
the Neural Engine. If it gets hard, a short "what I tried" writeup is itself worth shipping.
