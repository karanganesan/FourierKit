import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// copies the model's exact text, never the lowercased display
enum Pasteboard {
    @MainActor
    static func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// a plain text file for export, with the model's exact text
struct TextFile: FileDocument {
    static let readableContentTypes: [UTType] = [.plainText]
    var text: String

    init(text: String) {
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        text = configuration.file.regularFileContents.flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

/// lowercases every menu item the app shows, including the ones the system adds, so menus follow the
/// all-lowercase rule. titles only: shortcuts and actions are untouched.
@MainActor
final class MenuLowercaser {
    static let shared = MenuLowercaser()
    private var observers: [NSObjectProtocol] = []

    func start() {
        guard observers.isEmpty else { return }
        let center = NotificationCenter.default
        for name in [NSMenu.didAddItemNotification, NSMenu.didChangeItemNotification, NSMenu.didBeginTrackingNotification] {
            observers.append(center.addObserver(forName: name, object: nil, queue: .main) { note in
                nonisolated(unsafe) let menu = note.object as? NSMenu
                MainActor.assumeIsolated {
                    // only menus under the app's main menu: the apple menu belongs to the system
                    if let menu, Self.isInMainMenu(menu) { Self.lowercase(menu) }
                    MenuLowercaser.shared.scheduleFullPass()
                }
            })
        }
        if let main = NSApp.mainMenu { Self.lowercase(main) }
    }

    private var passScheduled = false

    /// swiftui and appkit fill some items in just after a change; one coalesced pass on the next turn catches them
    func scheduleFullPass() {
        guard !passScheduled else { return }
        passScheduled = true
        DispatchQueue.main.async {
            MainActor.assumeIsolated {
                MenuLowercaser.shared.passScheduled = false
                NSApp.mainMenu.map(Self.lowercase)
            }
        }
    }

    private static func isInMainMenu(_ menu: NSMenu) -> Bool {
        var current: NSMenu? = menu
        while let menu = current {
            if menu === NSApp.mainMenu { return true }
            current = menu.supermenu
        }
        return false
    }

    private static func lowercase(_ menu: NSMenu) {
        let lower = menu.title.lowercased()
        if menu.title != lower { menu.title = lower }
        for item in menu.items {
            let title = item.title.lowercased()
            if item.title != title { item.title = title }
            if let attributed = item.attributedTitle, attributed.string != attributed.string.lowercased() {
                let lower = NSMutableAttributedString(attributedString: attributed)
                lower.mutableString.setString(attributed.string.lowercased())
                item.attributedTitle = lower
            }
            if let submenu = item.submenu { lowercase(submenu) }
        }
    }
}
