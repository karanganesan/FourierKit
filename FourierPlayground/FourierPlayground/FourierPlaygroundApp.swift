import FourierKit
import FourierKitUI
import SwiftUI

@main
struct FourierPlaygroundApp: App {
    @State private var app = AppModel()

    init() {
        #if DEBUG
        MainThreadWatchdog.shared.start()
        #endif
        DispatchQueue.main.async { MenuLowercaser.shared.start() }
    }

    var body: some Scene {
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
    }
}
