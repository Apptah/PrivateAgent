import SwiftUI
import ModelHub

struct ModelManagerView: View {
    @State private var catalogEntries: [CatalogEntry] = []
    @State private var downloadManager = DownloadManager()
    @State private var availableSpace: String = "—"
    @State private var downloadingId: String?
    @State private var errorMessage: String?

    var body: some View {
        List {
            Section("Available Models") {
                ForEach(catalogEntries) { entry in
                    ModelRow(
                        entry: entry,
                        progress: downloadManager.activeDownloads[entry.id],
                        isDownloading: downloadingId == entry.id,
                        onDownload: { startDownload(entry) },
                        onCancel: { cancelDownload(entry.id) }
                    )
                }
            }

            if let errorMessage {
                Section {
                    Label(errorMessage, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
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
        downloadingId = entry.id

        Task {
            do {
                let hasSpace = try await downloadManager.checkFreeSpace(for: entry)
                guard hasSpace else {
                    errorMessage = "Not enough disk space for \(entry.displayName) (\(String(format: "%.1f", entry.totalSizeGB)) GB)"
                    downloadingId = nil
                    return
                }
                try await downloadManager.download(entry: entry)
            } catch {
                errorMessage = "Download failed: \(error.localizedDescription)"
                downloadingId = nil
            }
        }
    }

    private func cancelDownload(_ catalogId: String) {
        downloadManager.cancel(catalogId: catalogId)
        downloadingId = nil
    }
}

// MARK: - Model Row

private struct ModelRow: View {
    let entry: CatalogEntry
    let progress: DownloadManager.DownloadProgress?
    let isDownloading: Bool
    let onDownload: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.displayName).font(.headline)
                    Text("\(entry.quantization) • \(String(format: "%.1f", entry.totalSizeGB)) GB")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()

                if let progress {
                    // Downloading
                    VStack(alignment: .trailing, spacing: 2) {
                        Button(action: onCancel) {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                        Text("\(progress.filesCompleted)/\(progress.filesTotal) files")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                } else if isDownloading {
                    ProgressView()
                } else {
                    Button("Download", action: onDownload)
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }
            }

            if let progress {
                ProgressView(value: progress.progress)
                    .progressViewStyle(.linear)
                Text(progress.currentFile)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 4)
    }
}
