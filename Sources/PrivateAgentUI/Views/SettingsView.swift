import SwiftUI

struct SettingsView: View {
    @AppStorage("maxTokens") private var maxTokens: Double = 2048
    @AppStorage("temperature") private var temperature: Double = 0.7
    @AppStorage("defaultSystemPrompt") private var systemPrompt: String = "You are a helpful assistant."

    var body: some View {
        Form {
            Section("Generation") {
                VStack(alignment: .leading) {
                    Text("Max Tokens: \(Int(maxTokens))")
                    Slider(value: $maxTokens, in: 128...4096, step: 128)
                }
                VStack(alignment: .leading) {
                    Text("Temperature: \(String(format: "%.1f", temperature))")
                    Slider(value: $temperature, in: 0...2, step: 0.1)
                }
            }
            Section("System Prompt") {
                TextEditor(text: $systemPrompt)
                    .frame(minHeight: 80)
            }
            Section("About") {
                LabeledContent("Version", value: "0.1.0")
                Link("GitHub", destination: URL(string: "https://github.com")!)
            }
        }
        .navigationTitle("Settings")
    }
}
