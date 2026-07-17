# App — SwiftUI + MLX Swift (on-device)

The guaranteed deliverable: your quantized model running on the iPhone 14 Pro.

## Approach

Start from Apple's example rather than from scratch:

1. Clone [`ml-explore/mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples).
2. Use the **`LLMEval`** target as the base (it's a working on-device chat/generation app).
3. Change the model source to your Hub repo id:
   `osyounis/a16-summarizer-mlx-4bit` — mlx-swift can load a model from the Hub.
   (Or bundle a local copy for a fully offline first launch.)
4. Swap the prompt UI for a "paste dialogue → summarize" flow using the **same** system +
   instruction template as `train/prepare_data.py`, so on-device behavior matches eval.
5. Build to the **physical** device via Xcode (free personal provisioning is fine; not App Store).

## Capture these numbers (for the README model card)

- Model size on disk
- Prefill + decode tokens/sec
- Peak memory (Xcode → Instruments → Allocations / os_signpost)
- A screenshot of it running on-device

## Notes / gotchas

- **Simulator won't reflect real performance** — measure on the 14 Pro.
- First launch may download the model from the Hub; handle the loading state in the UI.
- Don't commit the `.xcodeproj` user data or any signing assets (already in `.gitignore`).

## Stretch: Core ML (Stage 6)

Only after MLX works. Convert with `coremltools`, handle the stateful KV-cache path, run on
the Neural Engine. If it gets hard, a short "what I tried" writeup is itself worth shipping.
