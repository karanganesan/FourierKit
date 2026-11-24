#if os(macOS)
import AppKit
#else
import UIKit
#endif
import SwiftUI
import UniformTypeIdentifiers

/// the words text uses for the device the app runs on
enum DeviceText {
    #if os(macOS)
    static let this = "this mac"
    #else
    static let this = "this device"
    #endif
}

/// copies the model's exact text, never the lowercased display
enum Pasteboard {
    @MainActor
    static func copy(_ text: String) {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        #else
        UIPasteboard.general.string = text
        #endif
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

/// a recording as a wav file: 16 khz, one channel, 32-bit float, exactly the samples the model hears
struct WAVFile: FileDocument {
    static let readableContentTypes: [UTType] = [.wav]
    var samples: [Float]
    var sampleRate = 16_000

    init(samples: [Float]) {
        self.samples = samples
    }

    init(configuration: ReadConfiguration) throws {
        throw CocoaError(.fileReadUnsupportedScheme)
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data())
    }

    func data() -> Data {
        func le<T: FixedWidthInteger>(_ value: T) -> [UInt8] {
            withUnsafeBytes(of: value.littleEndian) { Array($0) }
        }
        let bytes = samples.count * 4
        var out = Data(capacity: 58 + bytes)
        out += Array("RIFF".utf8) + le(UInt32(50 + bytes)) + Array("WAVE".utf8)
        // format 3 is ieee float; a non-pcm format carries the extension size (0) and a fact chunk
        out += Array("fmt ".utf8) + le(UInt32(18)) + le(UInt16(3)) + le(UInt16(1)) + le(UInt32(sampleRate))
            + le(UInt32(sampleRate * 4)) + le(UInt16(4)) + le(UInt16(32)) + le(UInt16(0))
        out += Array("fact".utf8) + le(UInt32(4)) + le(UInt32(samples.count))
        out += Array("data".utf8) + le(UInt32(bytes))
        samples.withUnsafeBytes { out.append(contentsOf: $0) }
        return out
    }
}

#if os(macOS)
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

/// the system share menu for a piece of text, opened when asked for. a `ShareLink` in the toolbar
/// needs the text while drawing, which would rebuild the toolbar every time the text changes.
enum SharePicker {
    @MainActor
    static func show(_ text: String) {
        guard let view = NSApp.keyWindow?.contentView else { return }
        let anchor = CGRect(x: view.bounds.maxX - 160, y: view.bounds.maxY - 48, width: 1, height: 1)
        NSSharingServicePicker(items: [text]).show(relativeTo: anchor, of: view, preferredEdge: .minY)
    }
}
#else
/// the system share sheet for a piece of text, opened when asked for, from whatever is on top
enum SharePicker {
    @MainActor
    static func show(_ text: String) {
        let scenes = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }
        let scene = scenes.first { $0.activationState == .foregroundActive } ?? scenes.first
        guard var top = scene?.keyWindow?.rootViewController else { return }
        while let presented = top.presentedViewController { top = presented }
        let sheet = UIActivityViewController(activityItems: [text], applicationActivities: nil)
        if let popover = sheet.popoverPresentationController {
            // ipad shows it as a popover; it points at the top trailing corner, where the share button is
            let bounds = top.view.bounds
            popover.sourceView = top.view
            popover.sourceRect = CGRect(x: bounds.maxX - 120, y: top.view.safeAreaInsets.top, width: 1, height: 1)
            popover.permittedArrowDirections = .up
        }
        top.present(sheet, animated: true)
    }
}
#endif
