import SwiftUI
import SwiftData
import FlashMoEBridge

public struct ContentView: View {
    @State private var engine = PrivateAgentEngine()
    @State private var path = NavigationPath()

    public init() {}

    public var body: some View {
        NavigationStack(path: $path) {
            ConversationListView(path: $path)
                .navigationDestination(for: UUID.self) { conversationId in
                    ChatView(conversationId: conversationId)
                }
        }
        .environment(engine)
        .modelContainer(for: [Conversation.self, Message.self])
    }
}
