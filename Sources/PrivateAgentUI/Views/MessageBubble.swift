import SwiftUI

struct MessageBubble: View {
    let message: Message
    /// Override displayed text (used for streaming assistant message)
    var overrideContent: String? = nil

    private var displayContent: String {
        overrideContent ?? message.content
    }

    private var isUser: Bool {
        message.role == .user
    }

    var body: some View {
        VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
            if let thinking = message.thinkingContent, !thinking.isEmpty {
                DisclosureGroup("Thinking") {
                    Text(thinking)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 4)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
            }

            HStack {
                if isUser { Spacer(minLength: 48) }

                bubbleText
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(isUser ? Color.accentColor : Color.secondary.opacity(0.15))
                    .foregroundStyle(isUser ? .white : .primary)
                    .clipShape(RoundedRectangle(cornerRadius: 18))

                if !isUser { Spacer(minLength: 48) }
            }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
    }

    @ViewBuilder
    private var bubbleText: some View {
        let content = displayContent
        if let attributed = try? AttributedString(markdown: content, options: .init(interpretedSyntax: .full)) {
            Text(attributed)
                .font(.body)
                .textSelection(.enabled)
        } else {
            Text(content)
                .font(.body)
                .textSelection(.enabled)
        }
    }
}
