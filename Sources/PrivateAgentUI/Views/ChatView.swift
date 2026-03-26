import SwiftUI
import FlashMoEBridge

struct ChatView: View {
    let conversationId: UUID
    @Environment(PrivateAgentEngine.self) private var engine

    var body: some View {
        Text("Chat: \(conversationId.uuidString.prefix(8))")
            .navigationTitle("Chat")
    }
}
