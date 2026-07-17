import SwiftUI

struct ContentView: View {
    @State private var model = Summarizer()
    @State private var dialogue = ContentView.sampleDialogue
    @FocusState private var editorFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    dialogueEditor
                    summarizeButton
                    if !model.summary.isEmpty {
                        summaryCard
                    }
                    if model.generationTokenCount > 0 {
                        statsGrid
                    }
                    if case .failed(let message) = model.phase {
                        errorBanner(message)
                    }
                }
                .padding()
            }
            .navigationTitle("a16 Summarizer")
            .navigationBarTitleDisplayMode(.inline)
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItem(placement: .keyboard) {
                    Button("Done") { editorFocused = false }
                }
            }
            .overlay(alignment: .top) { loadingBanner }
        }
        .task { await model.loadIfNeeded() }
    }

    // MARK: - Sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Paste a conversation and get a concise, third-person summary — generated fully on-device.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
    }

    private var dialogueEditor: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("DIALOGUE")
                .font(.caption).bold()
                .foregroundStyle(.secondary)
            TextEditor(text: $dialogue)
                .font(.system(.body, design: .monospaced))
                .frame(minHeight: 220)
                .padding(8)
                .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 12))
                .focused($editorFocused)
                .overlay(alignment: .topTrailing) {
                    if !dialogue.isEmpty {
                        Button {
                            dialogue = ""
                            editorFocused = true
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .padding(10)
                    }
                }
        }
    }

    private var summarizeButton: some View {
        Button {
            editorFocused = false
            Task { await model.summarize(dialogue) }
        } label: {
            HStack {
                if case .summarizing = model.phase {
                    ProgressView().tint(.white)
                    Text("Summarizing…")
                } else {
                    Image(systemName: "text.append")
                    Text("Summarize")
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .controlSize(.large)
        .disabled(dialogue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || model.isBusy)
    }

    private var summaryCard: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("SUMMARY")
                .font(.caption).bold()
                .foregroundStyle(.secondary)
            Text(model.summary)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(.tint.opacity(0.10), in: .rect(cornerRadius: 12))
        }
    }

    private var statsGrid: some View {
        let columns = [GridItem(.flexible()), GridItem(.flexible())]
        return LazyVGrid(columns: columns, spacing: 10) {
            stat("Decode", String(format: "%.1f tok/s", model.decodeTokensPerSecond), "speedometer")
            stat("Prefill", String(format: "%.1f tok/s", model.prefillTokensPerSecond), "gauge.with.dots.needle.67percent")
            stat("Prompt tokens", "\(model.promptTokenCount)", "text.alignleft")
            stat("Generated tokens", "\(model.generationTokenCount)", "number")
        }
    }

    private func stat(_ title: String, _ value: String, _ icon: String) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundStyle(.tint)
                .frame(width: 24)
            VStack(alignment: .leading, spacing: 2) {
                Text(value).font(.headline).monospacedDigit()
                Text(title).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(12)
        .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 12))
    }

    private func errorBanner(_ message: String) -> some View {
        Text(message)
            .font(.footnote)
            .foregroundStyle(.red)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(.red.opacity(0.10), in: .rect(cornerRadius: 12))
    }

    @ViewBuilder private var loadingBanner: some View {
        switch model.phase {
        case .downloading(let fraction):
            banner {
                VStack(spacing: 6) {
                    Text("Downloading model… \(Int(fraction * 100))%")
                        .font(.footnote)
                    ProgressView(value: fraction).frame(maxWidth: 240)
                }
            }
        case .loading:
            banner { Label("Loading model…", systemImage: "cpu").font(.footnote) }
        default:
            EmptyView()
        }
    }

    private func banner<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        content()
            .padding(12)
            .background(.regularMaterial, in: .rect(cornerRadius: 12))
            .shadow(radius: 4, y: 2)
            .padding(.top, 8)
    }
}

extension ContentView {
    /// A sample DialogSum-style conversation so the app is usable on first launch.
    static let sampleDialogue = """
        #Person1#: Hi Mr. Smith. I'm Doctor Hawkins. Why are you here today?
        #Person2#: I found it would be a good idea to get a check-up.
        #Person1#: Yes, well, you haven't had one for five years. You should have one every year.
        #Person2#: I know. I figure as long as there is nothing wrong, why go see the doctor?
        #Person1#: Well, the best way to avoid serious illnesses is to find out about them early. So try to come at least once a year for your own good.
        #Person2#: Ok.
        #Person1#: Let me see here. Your eyes and ears look fine. Take a deep breath, please. Do you smoke, Mr. Smith?
        #Person2#: Yes.
        #Person1#: Smoking is the leading cause of lung cancer and heart disease, you know. You really should quit.
        #Person2#: I've tried hundreds of times, but I just can't seem to kick the habit.
        #Person1#: Well, we have classes and some medications that might help. I'll give you more information before you leave.
        #Person2#: Ok, thanks doctor.
        """
}

#Preview {
    ContentView()
}
