# PrivateAgent Plan 6: SwiftUI Chat UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete SwiftUI chat interface — conversation list, chat view with streaming, model manager, settings — connected to the existing PrivateAgentEngine (mock generation for now).

**Architecture:** Native iOS style with NavigationStack. SwiftData for persistence. All views bind to PrivateAgentEngine (@MainActor @Observable). Chat streaming uses AsyncThrowingStream from Plan 3.

**Tech Stack:** SwiftUI, SwiftData, Swift 6.0, iOS 18.0+

**Spec:** `docs/superpowers/specs/2026-03-26-privateagent-design.md` — Section 4

**Depends on:** Plans 1-3 (engine scaffold + generation API)

**Produces:** Fully functional chat app UI that works with mock engine. Ready for real engine drop-in.

---

### Task 1: SwiftData models — Conversation + Message

**Files:**
- Create: `Sources/PrivateAgentUI/DataModels/Conversation.swift`
- Create: `Sources/PrivateAgentUI/DataModels/Message.swift`

- [ ] **Step 1: Create Conversation.swift**

```swift
import Foundation
import SwiftData

@Model
public class Conversation {
    @Attribute(.unique) public var id: UUID
    public var title: String
    public var systemPrompt: String
    public var createdAt: Date
    public var updatedAt: Date
    @Relationship(deleteRule: .cascade, inverse: \Message.conversation)
    public var messages: [Message]
    public var modelId: String

    public init(
        title: String = "New Chat",
        systemPrompt: String = "You are a helpful assistant.",
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
```

- [ ] **Step 2: Create Message.swift**

```swift
import Foundation
import SwiftData

public enum MessageRole: String, Codable, Sendable {
    case user
    case assistant
}

@Model
public class Message {
    @Attribute(.unique) public var id: UUID
    public var conversation: Conversation?
    public var role: MessageRole
    public var content: String
    public var thinkingContent: String?
    public var ordinal: Int
    public var createdAt: Date
    public var tokenCount: Int?

    public init(
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
```

- [ ] **Step 3: Build, commit**

```bash
swift build 2>&1 | tail -5
git add Sources/PrivateAgentUI/DataModels/
git commit -m "feat: add SwiftData models — Conversation + Message"
```

---

### Task 2: ConversationListView — main screen

**Files:**
- Modify: `Sources/PrivateAgentUI/ContentView.swift` → rewrite as app root
- Create: `Sources/PrivateAgentUI/Views/ConversationListView.swift`

- [ ] **Step 1: Create ConversationListView.swift**

```swift
import SwiftUI
import SwiftData

public struct ConversationListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]
    @State private var searchText = ""

    public init() {}

    public var body: some View {
        List {
            ForEach(filteredConversations) { conversation in
                NavigationLink(value: conversation.id) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(conversation.title)
                            .font(.headline)
                            .lineLimit(1)
                        Text(conversation.updatedAt, style: .relative)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 2)
                }
            }
            .onDelete(perform: deleteConversations)
        }
        .searchable(text: $searchText, prompt: "Search conversations")
        .navigationTitle("PrivateAgent")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button(action: createConversation) {
                    Image(systemName: "plus")
                }
            }
        }
    }

    private var filteredConversations: [Conversation] {
        if searchText.isEmpty { return conversations }
        return conversations.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
        }
    }

    private func createConversation() {
        let conversation = Conversation()
        modelContext.insert(conversation)
    }

    private func deleteConversations(at offsets: IndexSet) {
        for index in offsets {
            modelContext.delete(filteredConversations[index])
        }
    }
}
```

- [ ] **Step 2: Rewrite ContentView.swift as app root**

```swift
import SwiftUI
import SwiftData
import FlashMoEBridge

public struct ContentView: View {
    @State private var engine = PrivateAgentEngine()
    @State private var selectedConversationId: UUID?

    public init() {}

    public var body: some View {
        NavigationSplitView {
            ConversationListView()
                .navigationDestination(for: UUID.self) { id in
                    ChatView(conversationId: id)
                        .environment(engine)
                }
        } detail: {
            Text("Select a conversation")
                .foregroundStyle(.secondary)
        }
    }
}
```

NOTE: ChatView doesn't exist yet — this will cause a build error until Task 3. The implementer should create a placeholder ChatView to keep the build green.

- [ ] **Step 3: Build, commit**

```bash
swift build 2>&1 | tail -5
git add Sources/PrivateAgentUI/
git commit -m "feat: add ConversationListView with search and CRUD"
```

---

### Task 3: ChatView — streaming chat interface

**Files:**
- Create: `Sources/PrivateAgentUI/Views/ChatView.swift`
- Create: `Sources/PrivateAgentUI/Views/MessageBubble.swift`
- Create: `Sources/PrivateAgentUI/Views/InputBar.swift`
- Create: `Sources/PrivateAgentUI/ViewModels/ChatViewModel.swift`

- [ ] **Step 1: Create ChatViewModel.swift**

A @MainActor @Observable class that:
- Holds reference to PrivateAgentEngine and SwiftData modelContext
- Loads conversation by ID
- Sends messages: creates user Message, compiles prompt via PromptCompiler, calls engine.generate, streams tokens into assistant Message
- Handles streaming state (isGenerating, currentStreamText)
- Saves to SwiftData at checkpoints (not per-token)
- Supports cancel

- [ ] **Step 2: Create MessageBubble.swift**

A view that renders a single message:
- User messages: right-aligned, blue tint
- Assistant messages: left-aligned, plain
- Basic markdown rendering via Text(AttributedString)
- Thinking disclosure group for thinkingContent

- [ ] **Step 3: Create InputBar.swift**

- TextField(axis: .vertical) with 1-5 line limit
- Send button (arrow.up.circle.fill) or Stop button when generating
- Disabled when engine not ready

- [ ] **Step 4: Create ChatView.swift**

Combines:
- ScrollView + LazyVStack of MessageBubble
- Auto-scroll to bottom on new messages
- StatsBar showing tok/s during generation
- InputBar at bottom
- Toolbar with system prompt edit

- [ ] **Step 5: Build, commit**

```bash
swift build 2>&1 | tail -5
git add Sources/PrivateAgentUI/
git commit -m "feat: add ChatView with streaming messages, input bar, and stats"
```

---

### Task 4: ModelManagerView + SettingsView

**Files:**
- Create: `Sources/PrivateAgentUI/Views/ModelManagerView.swift`
- Create: `Sources/PrivateAgentUI/Views/SettingsView.swift`

- [ ] **Step 1: Create ModelManagerView.swift**

Shows:
- List of catalog entries with download size
- Download progress for active downloads
- Downloaded models with delete option
- Available disk space

- [ ] **Step 2: Create SettingsView.swift**

Sections:
- Generation: max tokens slider, temperature slider
- System prompt text editor
- About section with GitHub link

- [ ] **Step 3: Wire into ContentView navigation**

Add toolbar buttons for Settings (gear icon) and Models.

- [ ] **Step 4: Build, commit**

```bash
swift build 2>&1 | tail -5
git add Sources/PrivateAgentUI/
git commit -m "feat: add ModelManagerView and SettingsView"
```

---

### Task 5: App entry point + SwiftData container

**Files:**
- Create: `Apps/PrivateAgentiOS/PrivateAgentApp.swift`
- Modify: `Package.swift` (if needed for app target)

- [ ] **Step 1: Create PrivateAgentApp.swift**

```swift
import SwiftUI
import SwiftData
import PrivateAgentUI

@main
struct PrivateAgentApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(for: [Conversation.self, Message.self])
    }
}
```

NOTE: This is an iOS app entry point. It may need a separate Xcode project to compile for iOS (SPM can't produce .app bundles). For now, create the file and verify it compiles as part of PrivateAgentUI target or as a standalone.

- [ ] **Step 2: Build, commit**

```bash
swift build 2>&1 | tail -5
git add Apps/ Sources/
git commit -m "feat: add iOS app entry point with SwiftData container"
```

---

## Summary

After Plan 6:

- Full SwiftUI chat interface: conversation list, chat with streaming, model manager, settings
- SwiftData persistence: Conversation + Message with cascade delete, ordinal sorting
- ChatViewModel: manages generation lifecycle, prompt compilation, checkpoint saves
- Works with mock engine out of the box — drop in real engine later
- Native iOS look and feel (NavigationSplitView, .searchable, system colors)
