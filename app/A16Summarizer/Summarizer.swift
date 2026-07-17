import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import SwiftUI
import Tokenizers

/// Loads the a16-summarizer model and produces summaries, exposing generation stats.
///
/// The prompt formatting and decoding are kept identical to how the model was trained and
/// evaluated (see `train/prepare_data.py` and `convert/eval_mlx_rouge.py` in this repo), so
/// on-device behavior matches the reported ROUGE:
///   • system + "Summarize the following conversation:\n\n{dialogue}" via the Qwen chat template
///   • greedy decoding (temperature 0 → ArgMaxSampler), 96-token cap, stop on `<|im_end|>`
@MainActor
@Observable
final class Summarizer {

    /// The published 4-bit MLX model, downloaded from the Hugging Face Hub on first launch.
    /// `<|im_end|>` is added as an extra EOS token to match the eval stop set.
    static let modelConfiguration = ModelConfiguration(
        id: "osyounis/a16-summarizer-mlx-4bit",
        extraEOSTokens: ["<|im_end|>"]
    )

    /// Exact system message from `train/prepare_data.py`.
    static let systemPrompt =
        "You are a helpful assistant that writes a concise, third-person summary of a conversation."

    /// Matches `max_tokens=96` used when ROUGE was measured.
    static let maxTokens = 96

    enum Phase: Equatable {
        case idle
        case downloading(Double)   // 0...1
        case loading
        case ready
        case summarizing
        case failed(String)
    }

    private(set) var phase: Phase = .idle
    var summary: String = ""

    // Generation stats (from the model's own timing).
    private(set) var decodeTokensPerSecond: Double = 0
    private(set) var prefillTokensPerSecond: Double = 0
    private(set) var promptTokenCount: Int = 0
    private(set) var generationTokenCount: Int = 0

    private var container: ModelContainer?

    var isBusy: Bool {
        switch phase {
        case .downloading, .loading, .summarizing: return true
        case .idle, .ready, .failed: return false
        }
    }

    /// Download (first launch) and load the model. Idempotent — safe to call repeatedly.
    func loadIfNeeded() async {
        guard container == nil else { return }
        phase = .downloading(0)
        do {
            let loaded = try await #huggingFaceLoadModelContainer(
                configuration: Self.modelConfiguration
            ) { progress in
                Task { @MainActor in
                    self.phase = .downloading(progress.fractionCompleted)
                }
            }
            phase = .loading
            container = loaded
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    /// Summarize a pasted dialogue. Each call is an independent, single-shot request
    /// (no conversation history), matching the evaluation setup.
    func summarize(_ dialogue: String) async {
        let trimmed = dialogue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        await loadIfNeeded()
        guard let container else { return }

        phase = .summarizing
        summary = ""
        decodeTokensPerSecond = 0
        prefillTokensPerSecond = 0
        promptTokenCount = 0
        generationTokenCount = 0

        let input = UserInput(chat: [
            .system(Self.systemPrompt),
            .user("Summarize the following conversation:\n\n" + trimmed),
        ])
        let parameters = GenerateParameters(maxTokens: Self.maxTokens, temperature: 0)

        do {
            let prepared = try await container.prepare(input: input)
            let stream = try await container.generate(input: prepared, parameters: parameters)
            for await event in stream {
                switch event {
                case .chunk(let text):
                    summary += text
                case .info(let info):
                    prefillTokensPerSecond = info.promptTokensPerSecond
                    decodeTokensPerSecond = info.tokensPerSecond
                    promptTokenCount = info.promptTokenCount
                    generationTokenCount = info.generationTokenCount
                default:
                    break
                }
            }
            summary = summary.trimmingCharacters(in: .whitespacesAndNewlines)
            phase = .ready
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }
}
