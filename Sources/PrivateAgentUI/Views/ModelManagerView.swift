import SwiftUI
import ModelHub

struct ModelManagerView: View {
    @State private var catalogEntries: [CatalogEntry] = []

    var body: some View {
        List {
            Section("Available Models") {
                ForEach(catalogEntries) { entry in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(entry.displayName).font(.headline)
                            Text("\(entry.quantization) • \(String(format: "%.1f", entry.totalSizeGB)) GB")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Download") { }
                            .buttonStyle(.bordered)
                    }
                }
            }
            Section("Storage") {
                LabeledContent("Available", value: "— GB")
            }
        }
        .navigationTitle("Models")
        .task {
            catalogEntries = await ModelCatalog.shared.entries
        }
    }
}
