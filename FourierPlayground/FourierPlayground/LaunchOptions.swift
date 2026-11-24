import Foundation

/// hidden launch arguments for automation: screenshots, the app-hosted golden run and debugging.
/// they are compiled into release builds too, because the screenshots of a release build use them.
/// none of them change what a person sees when they open the app normally.
///
/// any argument that starts with `-fourier` marks the launch as automated (see `isAutomated`), so an
/// automated launch never reads or writes the transcripts in the person's own list.
///
///     -fourierMode listen|record|transcribe   opens that mode
///     -fourierOpen <file>                     transcribes the file at launch (transcribe mode)
///     -fourierListenFile <file>               listens to the file at real speed through FileAudioSource (listen mode)
///     -fourierRecordFile <file>               records the file at real speed through FileAudioSource (record mode)
///     -fourierShowStats                       opens the stats inspector
///     -fourierShowCode                        opens the code pane
///     -fourierFullScreen                      enters full screen (mac)
///     -fourierWindowSize <width>x<height>     sets the window's content size, e.g. 760x520
///     -fourierRecent <n>                      selects the n-th recent transcript (0 is the newest)
///     -fourierSettingsTab <tab>               opens settings on that tab (compute, decoding, audio, listen, lifecycle, model)
///     -fourierRecentList                      opens the recent list (on a phone, where it is a tab of its own)
///     -fourierSkipPrepare                     display only, for layout screenshots on a machine that has no model.
///                                             the prepare panel is skipped and a "display only" note stays on screen.
///                                             no model is prepared, the network is off and the model folder points at
///                                             an empty one, so nothing can listen, record or transcribe.
///
/// paths may be absolute or relative to the current directory. on ios, where the current directory is
/// the root of the file system, they are relative to the app's documents folder instead.
struct LaunchOptions {
    var mode: Mode?
    var openFile: URL?
    var listenFile: URL?
    var recordFile: URL?
    var showStats = false
    var showCode = false
    var fullScreen = false
    var windowSize: CGSize?
    var recent: Int?
    var settingsTab: String?
    var skipPrepare = false
    var recentList = false

    /// true when any `-fourier…` argument is present. read from the arguments themselves, not from the
    /// options parsed out of them, so a new argument counts without anyone remembering to add it here.
    private(set) var isAutomated = false

    static let current = LaunchOptions(arguments: ProcessInfo.processInfo.arguments)

    init(arguments: [String]) {
        isAutomated = arguments.dropFirst().contains { $0.hasPrefix("-fourier") }
        var index = 1
        func value() -> String? {
            guard index + 1 < arguments.count else { return nil }
            index += 1
            return arguments[index]
        }
        func file(_ path: String?) -> URL? {
            guard let path else { return nil }
            let expanded = (path as NSString).expandingTildeInPath
            #if os(macOS)
            let base = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            #else
            let base = URL.documentsDirectory
            #endif
            return URL(fileURLWithPath: expanded, relativeTo: base).standardizedFileURL
        }
        while index < arguments.count {
            switch arguments[index] {
            case "-fourierMode": mode = value().flatMap(Mode.init(rawValue:))
            case "-fourierOpen": openFile = file(value()); mode = mode ?? .transcribe
            case "-fourierListenFile": listenFile = file(value()); mode = mode ?? .listen
            case "-fourierRecordFile": recordFile = file(value()); mode = mode ?? .record
            case "-fourierShowStats": showStats = true
            case "-fourierShowCode": showCode = true
            case "-fourierFullScreen": fullScreen = true
            case "-fourierWindowSize":
                let parts = (value() ?? "").split(separator: "x").compactMap { Double($0) }
                if parts.count == 2 { windowSize = CGSize(width: parts[0], height: parts[1]) }
            case "-fourierRecent": recent = value().flatMap { Int($0) }
            case "-fourierSettingsTab": settingsTab = value()
            case "-fourierSkipPrepare": skipPrepare = true
            case "-fourierRecentList": recentList = true
            default: break
            }
            index += 1
        }
    }
}
