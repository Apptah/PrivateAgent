import Foundation
import SwiftData

enum MessageRole: String, Codable, Sendable {
    case user
    case assistant
}

@Model
class Message {
    @Attribute(.unique) var id: UUID
    var conversation: Conversation?
    var role: MessageRole
    var content: String
    var thinkingContent: String?
    var ordinal: Int
    var createdAt: Date
    var tokenCount: Int?

    init(
        role: MessageRole,
        content: String,
        ordinal: Int,
        thinkingContent: String? = nil
    ) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.thinkingContent = thinkingContent
        self.ordinal = ordinal
        self.createdAt = Date()
    }
}
