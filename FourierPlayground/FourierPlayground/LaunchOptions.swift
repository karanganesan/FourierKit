import Foundation

/// hidden launch arguments for automation: screenshots, the app-hosted golden run and debugging.
/// none of them change what a person sees when they open the app normally.
///
///     -fourierMode transcribe|record|listen   opens that mode
///     -fourierOpen <file>                     transcribes the file at launch (transcribe mode)
///     -fourierListenFile <file>               listens to the file at real speed through FileAudioSource (listen mode)
///     -fourierRecordFile <file>               records the file at real speed through FileAudioSource (record mode)
///     -fourierShowStats                       opens the stats inspector
///     -fourierWindowSize <width>x<height>     sets the window's content size, e.g. 760x520
///     -fourierRecent <n>                      selects the n-th recent transcript (0 is the newest)
///     -fourierSettingsTab <tab>               opens settings on that tab (compute, decoding, audio, listen, lifecycle, model)
///
/// paths may be absolute or relative to the current directory.
struct LaunchOptions {
    var mode: Mode?
    var openFile: URL?
    var listenFile: URL?
    var recordFile: URL?
    var showStats = false
    var windowSize: CGSize?
    var recent: Int?
    var settingsTab: String?

    var isAutomated: Bool {
        mode != nil || openFile != nil || listenFile != nil || recordFile != nil || showStats || windowSize != nil
            || recent != nil || settingsTab != nil
    }

    static let current = LaunchOptions(arguments: ProcessInfo.processInfo.arguments)

    init(arguments: [String]) {
        var index = 1
        func value() -> String? {
            guard index + 1 < arguments.count else { return nil }
            index += 1
            return arguments[index]
        }
        func file(_ path: String?) -> URL? {
            guard let path else { return nil }
            let expanded = (path as NSString).expandingTildeInPath
            return URL(fileURLWithPath: expanded, relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
                .standardizedFileURL
        }
        while index < arguments.count {
            switch arguments[index] {
            case "-fourierMode": mode = value().flatMap(Mode.init(rawValue:))
            case "-fourierOpen": openFile = file(value()); mode = mode ?? .transcribe
            case "-fourierListenFile": listenFile = file(value()); mode = mode ?? .listen
            case "-fourierRecordFile": recordFile = file(value()); mode = mode ?? .record
            case "-fourierShowStats": showStats = true
            case "-fourierWindowSize":
                let parts = (value() ?? "").split(separator: "x").compactMap { Double($0) }
                if parts.count == 2 { windowSize = CGSize(width: parts[0], height: parts[1]) }
            case "-fourierRecent": recent = value().flatMap { Int($0) }
            case "-fourierSettingsTab": settingsTab = value()
            default: break
            }
            index += 1
        }
    }
}
