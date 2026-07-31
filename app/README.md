# A16 Summarizer app (SwiftUI + MLX Swift, on-device)

A small standalone SwiftUI app that runs the
[`osyounis/a16-summarizer-mlx-4bit`](https://huggingface.co/osyounis/a16-summarizer-mlx-4bit)
model entirely on-device (iPhone, Apple GPU via MLX Swift). Paste a conversation, tap
Summarize, and you get a short third-person summary. No server, and no network at all
after the first-launch model download.

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

Three small Swift files. Everything model-specific lives in `Summarizer.swift`.

## Matching training at inference time

The reported ROUGE numbers only mean anything if the app runs the model the same way the
eval did, so `Summarizer.swift` reproduces the exact setup from training/eval:

- Prompt (from `train/prepare_data.py`): system message
  *"You are a helpful assistant that writes a concise, third-person summary of a conversation."*
  plus user *"Summarize the following conversation:\n\n{dialogue}"*, wrapped in the Qwen
  chat template (`UserInput` via `prepare(input:)`).
- Decoding (from `convert/eval_mlx_rouge.py`): greedy, `temperature: 0` (`ArgMaxSampler`),
  `maxTokens: 96`, stop on `<|im_end|>` (`extraEOSTokens`).

## Build and run

You need Xcode 16 or newer (this was built with Xcode 26) and a physical iPhone. MLX also
needs the Metal Toolchain, once per machine: `xcodebuild -downloadComponent MetalToolchain`.

1. Open `A16Summarizer.xcodeproj` in Xcode.
2. In the A16Summarizer target, go to Signing & Capabilities and set your Team
   (Automatic signing). The bundle id is `com.osyounis.a16summarizer`; change the
   reverse-DNS prefix to your own.
3. Select your iPhone as the run destination and hit ⌘R.
   - On the first build, Xcode asks to Trust & Enable the `mlx-swift-lm` Swift macros.
     Accept it, it's a one-time thing.
   - On first launch the app downloads the model (~847 MB) from the Hub, so be on Wi-Fi.

If you edit `project.yml`, regenerate the project with
`brew install xcodegen && cd app && xcodegen generate`.

Note that the iOS Simulator will not run this. MLX needs a real Metal GPU; on the
simulator `mlx::core::metal::Device` fails to initialize and the app aborts at launch.
Build to the physical device, which is sort of the point of the project anyway.

## Entitlements

Kept minimal since this is device-run only, with no App Store or distribution scaffolding:

- `com.apple.developer.kernel.increased-memory-limit` raises the process memory ceiling
  (about 5.25 GB on a 6 GB device) so the 4-bit model has comfortable headroom.
- `com.apple.security.network.client` covers the first-launch Hub download (it's a
  macOS-sandbox key; iOS ignores it).

## Measured on iPhone 14 Pro (A16, 6 GB)

| Measure | Value |
|---|---|
| Model size on disk | 847 MB |
| Prefill / decode tokens/sec | ~137 / 44.4 |
| Peak memory | 1.05 GB (831 MB active) |

See the model card in the repo [`README.md`](../README.md) and `results/hero_screen.PNG`.

## Attribution

The structure is adapted from Apple's `ml-explore/mlx-swift-examples` (`LLMBasic`), which
is MIT-licensed; see the repo `NOTICE`. The model-specific code, prompt formatting, and UI
are original.

## Stretch: Core ML (Stage 6)

Only after MLX works. Convert with `coremltools`, deal with the stateful KV-cache path,
and run on the Neural Engine. If it turns into a slog, a short "what I tried" writeup is
still worth shipping.
