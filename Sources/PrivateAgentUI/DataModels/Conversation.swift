import Foundation
import SwiftData

@Model
class Conversation {
    @Attribute(.unique) var id: UUID
    var title: String
    var systemPrompt: String
    var createdAt: Date
    var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \Message.conversation)
    var messages: [Message]
    var modelId: String

    init(
        title: String = "New Chat",
        systemPrompt: String = "You are a helpful assistant. /no_think",
        modelId: String = "default"
    ) {
        self.id = UUID()
        self.title = title
        self.systemPrompt = systemPrompt
        self.createdAt = Date()
        self.updatedAt = Date()
        self.messages = []
        self.modelId = modelId
    }
}
