import SwiftUI
import PrivateAgentUI

@main
struct PrivateAgentApp: App {

    init() {
        // Suppress UIKit keyboard auto-layout constraint warnings (iOS system issue)
        UserDefaults.standard.set(false, forKey: "_UIConstraintBasedLayoutLogUnsatisfiable")
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
