import SwiftUI
import SwiftData
import FlashMoEBridge

struct ChatView: View {
    let conversationId: UUID
    @Environment(PrivateAgentEngine.self) private var engine
    @Environment(\.modelContext) private var modelContext
    @State private var viewModel: ChatViewModel?
    @State private var showSystemPrompt = false

    var body: some View {
        Group {
            if let viewModel {
                chatContent(viewModel: viewModel)
            } else {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            if viewModel == nil {
                viewModel = ChatViewModel(
                    conversationId: conversationId,
                    modelContext: modelContext,
                    engine: engine
                )
            }
        }
    }

    @ViewBuilder
    private func chatContent(viewModel: ChatViewModel) -> some View {
        VStack(spacing: 0) {
            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        let messages = viewModel.sortedMessages
                        ForEach(messages) { message in
                            let isLastAssistant = viewModel.isGenerating
                                && message.role == .assistant
                                && message.id == messages.last(where: { $0.role == .assistant })?.id

                            MessageBubble(
                                message: message,
                                overrideContent: isLastAssistant && !viewModel.streamingText.isEmpty
                                    ? viewModel.streamingText
                                    : nil
                            )
                            .padding(.horizontal, 16)
                            .id(message.id)
                        }

                        // Streaming indicator when assistant message is empty
                        if viewModel.isGenerating && viewModel.streamingText.isEmpty {
                            HStack(spacing: 6) {
                                ProgressView()
                                    .scaleEffect(0.8)
                                Text("Generating…")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 16)
                            .id("generating-indicator")
                        }

                        // Spacer anchor for auto-scroll
                        Color.clear
                            .frame(height: 1)
                            .id("bottom-anchor")
                    }
                    .padding(.top, 12)
                    .padding(.bottom, 4)
                }
                .onChange(of: viewModel.streamingText) { _, _ in
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo("bottom-anchor", anchor: .bottom)
                    }
                }
                .onChange(of: viewModel.sortedMessages.count) { _, _ in
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo("bottom-anchor", anchor: .bottom)
                    }
                }
            }

            // Stats bar
            if viewModel.isGenerating || !viewModel.currentStats.isEmpty {
                HStack {
                    Spacer()
                    Text(viewModel.isGenerating && viewModel.currentStats.isEmpty
                         ? "Generating…"
                         : viewModel.currentStats)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 4)
                }
                .background(.bar)
            }

            Divider()

            // Input bar
            InputBar(viewModel: viewModel)
        }
        .navigationTitle(viewModel.conversation?.title ?? "Chat")
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .toolbar {
            #if os(iOS)
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showSystemPrompt = true
                } label: {
                    Image(systemName: "gear")
                }
            }
            #else
            ToolbarItem(placement: .automatic) {
                Button {
                    showSystemPrompt = true
                } label: {
                    Image(systemName: "gear")
                }
            }
            #endif
        }
        .sheet(isPresented: $showSystemPrompt) {
            SystemPromptSheet(conversation: viewModel.conversation) {
                viewModel.invalidateCache()
            }
        }
    }
}

// MARK: - System Prompt Sheet

private struct SystemPromptSheet: View {
    let conversation: Conversation?
    var onSave: (() -> Void)?
    @Environment(\.dismiss) private var dismiss
    @State private var draft: String = ""

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                Text("System Prompt")
                    .font(.headline)
                    .padding(.horizontal)

                TextEditor(text: $draft)
                    .font(.body)
                    .padding(8)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                    .padding(.horizontal)

                Spacer()
            }
            .padding(.top)
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        conversation?.systemPrompt = draft
                        onSave?()
                        dismiss()
                    }
                }
            }
            .onAppear {
                draft = conversation?.systemPrompt ?? ""
            }
        }
    }
}
