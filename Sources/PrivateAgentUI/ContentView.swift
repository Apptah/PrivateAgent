import SwiftUI
import FlashMoEBridge

public struct ContentView: View {
    @State private var engine = PrivateAgentEngine()

    public init() {}

    public var body: some View {
        NavigationStack {
            VStack {
                Text("PrivateAgent")
                    .font(.largeTitle)
                Text("Engine: \(engine.state == .idle ? "Ready" : "...")")
                    .foregroundStyle(.secondary)
            }
            .navigationTitle("PrivateAgent")
        }
    }
}
