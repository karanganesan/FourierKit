import FourierKit
import FourierKitUI
import SwiftUI

@main
struct FourierPlaygroundApp: App {
    @State private var app = AppModel()

    init() {
        #if DEBUG
        MainThreadWatchdog.shared.start()
        CodeSnippets.selfCheck()
        #endif
        #if os(macOS)
        DispatchQueue.main.async { MenuLowercaser.shared.start() }
        #endif
    }

    var body: some Scene {
        #if os(macOS)
        Window("fourier playground", id: "main") {
            ContentView()
                .environment(app)
                .task { await app.launch() }
        }
        .defaultSize(width: 1120, height: 720)
        // always open with the window, even if it was closed when the app last quit
        .defaultLaunchBehavior(.presented)
        .windowResizability(.contentMinSize)
        .commands {
            PlaygroundCommands(app: app)
        }

        Settings {
            SettingsView()
                .environment(app)
        }
        #else
        // one scene: the app's state, the engine and the microphone are one per process
        WindowGroup {
            ContentView()
                .environment(app)
                .task { await app.launch() }
        }
        #endif
    }
}
