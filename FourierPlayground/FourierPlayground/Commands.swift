import AppKit
import FourierKit
import SwiftUI

/// the menu bar, all lowercase, with the standard mac shortcuts. the system's own groups are replaced by the same
/// actions (sent through the responder chain) so their titles follow the app's lowercase rule.
struct PlaygroundCommands: Commands {
    let app: AppModel

    var body: some Commands {
        // app menu
        CommandGroup(replacing: .appInfo) {
            Button("about fourier playground") { NSApp.orderFrontStandardAboutPanel(nil) }
        }
        CommandGroup(replacing: .appVisibility) {
            Button("hide fourier playground") { NSApp.hide(nil) }
                .keyboardShortcut("h")
            Button("hide others") { NSApp.hideOtherApplications(nil) }
                .keyboardShortcut("h", modifiers: [.command, .option])
            Button("show all") { NSApp.unhideAllApplications(nil) }
        }
        CommandGroup(replacing: .appTermination) {
            Button("quit fourier playground") { NSApp.terminate(nil) }
                .keyboardShortcut("q")
        }

        // file
        CommandGroup(replacing: .newItem) {
            Button("open audio file…") {
                app.selection = .mode(.transcribe)
                app.transcribe.choosing = true
            }
            .keyboardShortcut("o")
        }
        CommandGroup(replacing: .saveItem) {
            Button("close") { NSApp.keyWindow?.performClose(nil) }
                .keyboardShortcut("w")
        }

        // edit
        CommandGroup(replacing: .undoRedo) {
            Button("undo") { Self.send("undo:") }
                .keyboardShortcut("z")
            Button("redo") { Self.send("redo:") }
                .keyboardShortcut("z", modifiers: [.command, .shift])
        }
        CommandGroup(replacing: .pasteboard) {
            Button("cut") { Self.send("cut:") }
                .keyboardShortcut("x")
            Button("copy") { Self.send("copy:") }
                .keyboardShortcut("c")
            Button("paste") { Self.send("paste:") }
                .keyboardShortcut("v")
            Button("delete") { Self.send("delete:") }
            Button("select all") { Self.send("selectAll:") }
                .keyboardShortcut("a")
            Divider()
            Button("copy transcript") {
                if let text = app.currentText { Pasteboard.copy(text) }
            }
            .keyboardShortcut("c", modifiers: [.command, .shift])
            .disabled(app.currentText == nil)
        }

        // view
        CommandGroup(replacing: .sidebar) {
            Button("toggle sidebar") { Self.send("toggleSidebar:") }
                .keyboardShortcut("s", modifiers: [.command, .control])
            Button(app.showStats ? "hide live stats" : "show live stats") { app.showStats.toggle() }
                .keyboardShortcut("i", modifiers: [.command, .control])
        }

        // speech
        CommandMenu("speech") {
            Button(app.toggleTitle) { app.toggleCurrent() }
                .keyboardShortcut(.return, modifiers: .command)
                .disabled(!app.canToggleCurrent)
            Divider()
            ForEach(Array(Mode.allCases.enumerated()), id: \.element) { index, mode in
                Button(mode.title) { app.selection = .mode(mode) }
                    .keyboardShortcut(KeyEquivalent(Character("\(index + 1)")), modifiers: .command)
            }
        }

        // window
        CommandGroup(replacing: .windowSize) {
            Button("minimize") { NSApp.keyWindow?.performMiniaturize(nil) }
                .keyboardShortcut("m")
            Button("zoom") { NSApp.keyWindow?.performZoom(nil) }
        }
        // help: no help book, so no dead-end item
        CommandGroup(replacing: .help) {}
    }

    /// sends a standard action to the first responder, as the system's own menu items do
    @MainActor
    private static func send(_ action: String) {
        NSApp.sendAction(Selector(action), to: nil, from: nil)
    }
}
