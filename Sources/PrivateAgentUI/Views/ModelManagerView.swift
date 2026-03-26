import SwiftUI
import ModelHub

struct ModelManagerView: View {
    @State private var catalogEntries: [CatalogEntry] = []
    @State private var downloadManager = DownloadManager()
    @State private var availableSpace: String = "—"
    @State private var errorMessage: String?
    @State private var downloadedModelIds: Set<String> = []
    @State private var confirmingDeleteId: String?
    @AppStorage("selectedModelId") private var selectedModelId: String = ""

    var body: some View {
        List {
            if let errorMessage {
                Section {
                    Label(errorMessage, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }

            Section("Available Models") {
                ForEach(catalogEntries) { entry in
                    ModelRow(
                        entry: entry,
                        progress: downloadManager.activeDownloads[entry.id],
                        isDownloaded: downloadedModelIds.contains(entry.id),
                        isSelected: selectedModelId == entry.id,
                        onDownload: { startDownload(entry) },
                        onCancel: { cancelDownload(entry.id) },
                        onDelete: { confirmingDeleteId = entry.id },
                        onSelect: { selectedModelId = entry.id }
                    )
                }
            }

            Section("Storage") {
                LabeledContent("Available Space", value: availableSpace)
            }
        }
        .navigationTitle("Models")
        .task {
            catalogEntries = await ModelCatalog.shared.entries
            await refreshState()
        }
        .alert("Delete Model", isPresented: .init(
            get: { confirmingDeleteId != nil },
            set: { if !$0 { confirmingDeleteId = nil } }
        )) {
            Button("Delete", role: .destructive) {
                if let id = confirmingDeleteId {
                    deleteModel(id)
                }
            }
            Button("Cancel", role: .cancel) {
                confirmingDeleteId = nil
            }
        } message: {
            if let id = confirmingDeleteId,
               let entry = catalogEntries.first(where: { $0.id == id }) {
                Text("Delete \(entry.displayName) and free up \(String(format: "%.1f GB", entry.totalSizeGB))? You can re-download it later.")
            } else {
                Text("Are you sure you want to delete this model?")
            }
        }
    }

    private func refreshState() async {
        let storage = ModelStorage()
        if let bytes = try? await storage.availableSpaceBytes() {
            availableSpace = String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
        }
        if let models = try? await storage.listModels() {
            downloadedModelIds = Set(models.map { $0.lastPathComponent })
            // Auto-select if no selection or current selection was deleted
            if !downloadedModelIds.contains(selectedModelId) {
                selectedModelId = downloadedModelIds.first ?? ""
            }
        }
    }

    private func startDownload(_ entry: CatalogEntry) {
        errorMessage = nil
        Task {
            do {
                try await downloadManager.download(entry: entry)
                await refreshState()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func cancelDownload(_ catalogId: String) {
        downloadManager.cancel(catalogId: catalogId)
    }

    private func deleteModel(_ catalogId: String) {
        Task {
            do {
                try await ModelStorage().deleteModel(catalogId: catalogId)
                downloadManager.cancel(catalogId: catalogId)
                await refreshState()
            } catch {
                errorMessage = "Failed to delete: \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Model Row

private struct ModelRow: View {
    let entry: CatalogEntry
    let progress: DownloadManager.DownloadProgress?
    let isDownloaded: Bool
    let isSelected: Bool
    let onDownload: () -> Void
    let onCancel: () -> Void
    let onDelete: () -> Void
    let onSelect: () -> Void

    private var isReady: Bool {
        isDownloaded || progress?.status == .complete
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(entry.displayName).font(.headline)
                        if isSelected && isReady {
                            Text("Active")
                                .font(.caption2)
                                .fontWeight(.semibold)
                                .foregroundStyle(.white)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.blue, in: Capsule())
                        }
                    }
                    Text(entry.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()

                if let progress {
                    switch progress.status {
                    case .downloading:
                        Button(action: onCancel) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    case .complete:
                        HStack(spacing: 8) {
                            if !isSelected {
                                Button("Use", action: onSelect)
                                    .buttonStyle(.bordered)
                                    .controlSize(.small)
                            }
                            Button(action: onDelete) {
                                Image(systemName: "trash")
                                    .foregroundStyle(.red)
                            }
                            .buttonStyle(.plain)
                        }
                    case .failed:
                        Image(systemName: "exclamationmark.circle.fill")
                            .foregroundStyle(.red)
                    default:
                        ProgressView()
                    }
                } else if isDownloaded {
                    HStack(spacing: 8) {
                        if !isSelected {
                            Button("Use", action: onSelect)
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                        }
                        Button(action: onDelete) {
                            Image(systemName: "trash")
                                .foregroundStyle(.red)
                        }
                        .buttonStyle(.plain)
                    }
                } else {
                    Button("Download", action: onDownload)
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }
            }

            if let progress, progress.status == .downloading {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: progress.progress)
                        .progressViewStyle(.linear)
                    HStack {
                        Text("\(progress.filesCompleted)/\(progress.filesTotal) files")
                        Spacer()
                        Text(formatBytes(progress.bytesDownloaded) + " / " + formatBytes(progress.bytesTotal))
                    }
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    Text(progress.currentFile)
                        .font(.caption2)
                        .foregroundStyle(.quaternary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func formatBytes(_ bytes: UInt64) -> String {
        if bytes >= 1_073_741_824 {
            return String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
        } else if bytes >= 1_048_576 {
            return String(format: "%.0f MB", Double(bytes) / 1_048_576)
        } else {
            return String(format: "%.0f KB", Double(bytes) / 1024)
        }
    }
}
