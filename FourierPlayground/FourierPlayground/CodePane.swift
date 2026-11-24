import FourierKit
import FourierKitUI
import SwiftUI

/// the swift you would paste into your own app to do what the playground is doing right now:
/// how to add the package, the options that differ from the defaults, and the calls for this mode.
enum CodeSnippets {
    static let packageURL = "https://github.com/karanganesan/FourierKit.git"

    static var package: String {
        """
        // Package.swift
        dependencies: [
            .package(url: "\(packageURL)", from: "\(Fourier.version)"),
        ],
        targets: [
            .target(name: "YourApp", dependencies: [
                .product(name: "FourierKit", package: "FourierKit"),
                // optional default views
                .product(name: "FourierKitUI", package: "FourierKit"),
            ]),
        ]
        """
    }

    static let plist = """
        <!-- Info.plist: needed for record and listen -->
        <key>NSMicrophoneUsageDescription</key>
        <string>to turn what you say into text on this mac.</string>

        <!-- entitlements, only if the app is sandboxed -->
        <key>com.apple.security.device.audio-input</key>
        <true/>
        <key>com.apple.security.network.client</key>
        <true/>
        """

    /// one line per setting that is not at its default. empty when everything is default.
    static func optionLines(_ settings: PlaygroundSettings) -> [String] {
        let base = PlaygroundSettings.defaults
        var lines: [String] = []
        func quoted(_ text: String) -> String {
            "\"" + text.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
        }
        func number(_ value: Double) -> String {
            value == value.rounded() ? String(Int(value)) : String(value)
        }
        func search(_ search: PlaygroundSettings.Search, _ width: Int) -> String {
            switch search {
            case .automatic: ".automatic"
            case .greedy: ".greedy"
            case .beam: ".beam(\(max(1, width)))"
            }
        }

        if settings.policy != base.policy { lines.append("options.compute.policy = .\(settings.policy)") }
        for stage in Stage.allCases {
            if let pin = settings.chipPins[stage.rawValue], pin != .automatic {
                lines.append("options.compute.chips[.\(stage.rawValue)] = .\(pin.rawValue)")
            }
            if let pin = settings.precisionPins[stage.rawValue], pin != .automatic {
                lines.append("options.compute.precision[.\(stage.rawValue)] = .\(pin.rawValue)")
            }
        }

        if settings.search != base.search || (settings.search == .beam && settings.beamWidth != base.beamWidth) {
            lines.append("options.decoding.search = \(search(settings.search, settings.beamWidth))")
        }
        if settings.limitTokens { lines.append("options.decoding.maxTokens = \(max(1, settings.maxTokens))") }
        if settings.customRepeatGuard { lines.append("options.decoding.repeatGuard = \(max(0, settings.repeatGuard))") }
        if settings.language != base.language, !settings.language.isEmpty {
            lines.append("options.decoding.language = \(quoted(settings.language))")
        }
        if !settings.context.isEmpty { lines.append("options.decoding.context = \(quoted(settings.context))") }

        if let device = settings.inputDevice { lines.append("options.inputDevice = \(quoted(device))") }
        if !settings.chunkingAutomatic {
            lines.append("options.chunking = .maxPieceLength(.seconds(\(number(settings.maxPieceSeconds))))")
        }
        if settings.noise == "off" { lines.append("options.noise = .off") }
        if settings.echoCancellation != base.echoCancellation {
            lines.append("options.echoCancellation = .\(settings.echoCancellation.rawValue)")
        }
        if !settings.gainAutomatic { lines.append("options.gain = .fixed(decibels: \(number(settings.gainDecibels)))") }
        if settings.preRollMilliseconds != base.preRollMilliseconds {
            lines.append("options.preRoll = .milliseconds(\(Int(settings.preRollMilliseconds)))")
        }

        if settings.drafts != base.drafts { lines.append("options.listen.drafts = \(settings.drafts)") }
        if !settings.updateAutomatic {
            lines.append("options.listen.updateInterval = .fixed(.milliseconds(\(Int(settings.updateMilliseconds))))")
        }
        if settings.settleAfter != base.settleAfter { lines.append("options.listen.settleAfter = \(max(1, settings.settleAfter))") }
        if settings.pauseMilliseconds != base.pauseMilliseconds {
            lines.append("options.listen.pauseToEndPhrase = .milliseconds(\(Int(settings.pauseMilliseconds)))")
        }
        if settings.maxPhraseSeconds != base.maxPhraseSeconds {
            lines.append("options.listen.maxPhraseLength = .seconds(\(number(settings.maxPhraseSeconds)))")
        }
        if settings.finalPass != base.finalPass { lines.append("options.listen.finalPass = .\(settings.finalPass)") }
        if settings.draftSearch != base.draftSearch || (settings.draftSearch == .beam && settings.draftBeamWidth != base.draftBeamWidth) {
            lines.append("options.listen.decoding.draft.search = \(search(settings.draftSearch, settings.draftBeamWidth))")
        }

        if settings.keepWarm != base.keepWarm { lines.append("options.keepWarm = .\(settings.keepWarm)") }
        if settings.manualPermissions { lines.append("options.permissions = .manual") }

        if let directory = settings.modelDirectory, !directory.isEmpty {
            lines.append("options.model.directory = URL(fileURLWithPath: \(quoted(directory)))")
        }
        if !settings.revision.isEmpty, settings.revision != ModelOptions.pinnedRevision {
            lines.append("options.model.revision = \(quoted(settings.revision))")
        }
        if !settings.sha256.isEmpty { lines.append("options.model.sha256 = \(quoted(settings.sha256))") }
        if settings.offline { lines.append("options.model.offline = true") }
        return lines
    }

    static func setup(_ settings: PlaygroundSettings) -> String {
        let lines = optionLines(settings)
        if lines.isEmpty {
            return """
                import FourierKit

                let speech = Fourier()          // every option at its default
                try await speech.prepare()      // optional: download, load and warm up now
                """
        }
        return """
            import FourierKit

            var options = FourierOptions()
            \(lines.joined(separator: "\n"))

            let speech = Fourier(options)
            try await speech.prepare()      // optional: download, load and warm up now
            """
    }

    static func usage(_ mode: Mode) -> String {
        switch mode {
        case .transcribe:
            """
            let transcript = try await speech.transcribe(contentsOf: url)
            print(transcript.text)

            // or with progress and cancel
            let job = speech.transcriptionJob(contentsOf: url)
            for await fraction in job.progress { print(fraction) }
            let result = try await job.result
            """
        case .record:
            """
            let recorder = try await speech.record()
            for await level in recorder.levels { /* draw a meter */ }

            // when the user stops:
            let recording = await recorder.stop()
            let transcript = try await speech.transcribe(recording)
            print(transcript.text)
            """
        case .listen:
            """
            let live = try await speech.listen()

            let reader = Task {
                for try await event in live.events {
                    if case .update(let update) = event {
                        // settled never changes, draft may
                        print(update.settled, update.draft)
                    }
                }
            }

            // when the user stops:
            let transcript = try await live.finish()
            try await reader.value
            print(transcript.text)
            """
        }
    }

    static func ui(_ mode: Mode) -> String? {
        guard mode == .listen else { return nil }
        return """
            import FourierKitUI
            import SwiftUI

            // the whole live screen in one view
            FourierLiveView(speech)
            """
    }

    static func everything(settings: PlaygroundSettings, mode: Mode) -> String {
        [setup(settings), usage(mode)].joined(separator: "\n\n")
    }
}

/// the code pane: install, set up, use. each block copies with one click and keeps its case.
struct CodePane: View {
    @Environment(AppModel.self) private var app

    private var mode: Mode {
        if case .mode(let mode) = app.selection { return mode }
        if case .recent(let id) = app.selection, let item = app.recents.item(id) { return item.mode }
        return .listen
    }

    var body: some View {
        let settings = app.settings
        let changed = CodeSnippets.optionLines(settings).count
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                CodeBlock(
                    title: "1. add the package",
                    note: "in xcode: file, add package dependencies, then paste the url. or add it to Package.swift.",
                    code: CodeSnippets.packageURL
                )
                CodeBlock(title: nil, note: nil, code: CodeSnippets.package)
                if mode != .transcribe {
                    CodeBlock(title: "2. allow the microphone", note: "\(mode.title) asks for the microphone the first time it runs.", code: CodeSnippets.plist)
                }
                CodeBlock(
                    title: mode == .transcribe ? "2. set up" : "3. set up",
                    note: changed == 0
                        ? "every setting is at its default, so there is nothing to configure."
                        : "\(changed) \(changed == 1 ? "setting differs" : "settings differ") from the defaults. these lines match this playground's settings.",
                    code: CodeSnippets.setup(settings)
                )
                CodeBlock(
                    title: mode == .transcribe ? "3. \(mode.title)" : "4. \(mode.title)",
                    note: "the calls behind this screen.",
                    code: CodeSnippets.usage(mode)
                )
                if let ui = CodeSnippets.ui(mode) {
                    CodeBlock(title: "or use the default view", note: "prepare, permission, mic, level and live text in one view.", code: ui)
                }
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .fourierScrollEdges()
    }
}

private struct CodeBlock: View {
    let title: String?
    let note: String?
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let title {
                Text(title)
                    .fourierTextStyle(.caption)
                    .fourierInk(.tertiary)
            }
            if let note {
                Text(note)
                    .fourierTextStyle(.body)
                    .fourierInk(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(alignment: .top, spacing: 8) {
                // code keeps its case and uses the system's fixed-width face
                Text(code)
                    .font(.system(size: 12, design: .monospaced))
                    .lineSpacing(4)
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                Button(copied ? "copied" : "copy", systemImage: copied ? "checkmark" : "doc.on.doc") {
                    Pasteboard.copy(code)
                    copied = true
                    Task {
                        try? await Task.sleep(for: .seconds(1.6))
                        copied = false
                    }
                }
                .labelStyle(.iconOnly)
                .fourierGlassButton()
                .buttonBorderShape(.circle)
                .contentTransition(.symbolEffect(.replace))
                .help("copy this code")
                .accessibilityLabel("copy code")
            }
        }
    }
}
