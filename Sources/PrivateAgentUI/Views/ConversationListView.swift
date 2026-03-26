import SwiftUI
import SwiftData

struct ConversationListView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \Conversation.updatedAt, order: .reverse) private var conversations: [Conversation]

    @State private var searchText = ""
    @State private var showSettings = false
    @State private var showModels = false
    @Binding var path: NavigationPath

    var filteredConversations: [Conversation] {
        if searchText.isEmpty {
            return conversations
        }
        return conversations.filter { $0.title.localizedCaseInsensitiveContains(searchText) }
    }

    var body: some View {
        List {
            ForEach(filteredConversations) { conversation in
                NavigationLink(value: conversation.id) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(conversation.title)
                            .font(.headline)
                        Text(conversation.updatedAt.formatted(.relative(presentation: .named)))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 2)
                }
            }
            .onDelete(perform: deleteConversations)
        }
        .searchable(text: $searchText, prompt: "Search conversations")
        .navigationTitle("Chats")
        .toolbar {
            #if os(iOS)
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    newConversation()
                } label: {
                    Image(systemName: "square.and.pencil")
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showModels = true
                } label: {
                    Image(systemName: "square.grid.2x2")
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "gear")
                }
            }
            ToolbarItem(placement: .topBarLeading) {
                EditButton()
            }
            #else
            ToolbarItem(placement: .automatic) {
                Button {
                    newConversation()
                } label: {
                    Image(systemName: "square.and.pencil")
                }
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    showModels = true
                } label: {
                    Image(systemName: "square.grid.2x2")
                }
            }
            ToolbarItem(placement: .automatic) {
                Button {
                    showSettings = true
                } label: {
                    Image(systemName: "gear")
                }
            }
            #endif
        }
        .sheet(isPresented: $showSettings) {
            NavigationStack {
                SettingsView()
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showSettings = false }
                        }
                    }
            }
        }
        .sheet(isPresented: $showModels) {
            NavigationStack {
                ModelManagerView()
                    .toolbar {
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Done") { showModels = false }
                        }
                    }
            }
        }
    }

    private func newConversation() {
        let convo = Conversation()
        modelContext.insert(convo)
        try? modelContext.save()
        path.append(convo.id)
    }

    private func deleteConversations(at offsets: IndexSet) {
        for index in offsets {
            modelContext.delete(filteredConversations[index])
        }
        try? modelContext.save()
    }
}
