import MLX
import SwiftUI

/// On-device dialogue summarizer built on the a16-summarizer 4-bit MLX model
/// (Qwen2.5-1.5B, QLoRA-tuned on DialogSum). Runs entirely on the Apple GPU via MLX Swift
/// (Metal) — no server, no network after the first-launch model download.
@main
struct A16SummarizerApp: App {
    init() {
        // Keep MLX's Metal buffer cache small; the weights dominate the footprint.
        Memory.cacheLimit = 20 * 1024 * 1024
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
