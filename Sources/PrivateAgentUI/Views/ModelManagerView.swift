import SwiftUI
import ModelHub

struct ModelManagerView: View {
    @State private var catalogEntries: [CatalogEntry] = []
    @State private var downloadManager = DownloadManager()
    @State private var availableSpace: String = "—"
    @State private var errorMessage: String?

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
                        onDownload: { startDownload(entry) },
                        onCancel: { cancelDownload(entry.id) }
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
            if let bytes = try? await ModelStorage().availableSpaceBytes() {
                availableSpace = String(format: "%.1f GB", Double(bytes) / 1_073_741_824)
            }
        }
    }

    private func startDownload(_ entry: CatalogEntry) {
        errorMessage = nil
        Task {
            do {
                try await downloadManager.download(entry: entry)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func cancelDownload(_ catalogId: String) {
        downloadManager.cancel(catalogId: catalogId)
    }
}

// MARK: - Model Row

private struct ModelRow: View {
    let entry: CatalogEntry
    let progress: DownloadManager.DownloadProgress?
    let onDownload: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.displayName).font(.headline)
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
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                    case .failed:
                        Image(systemName: "exclamationmark.circle.fill")
                            .foregroundStyle(.red)
                    default:
                        ProgressView()
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
