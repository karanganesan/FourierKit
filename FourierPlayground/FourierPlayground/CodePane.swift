import FourierKit
import FourierKitUI
import SwiftUI

/// the swift you would paste into your own app to do what the playground is doing right now:
/// how to add the package, the options that differ from the defaults, and the calls for this mode.
enum CodeSnippets {
    static let packageURL = "https://github.com/karanganesan/FourierKit.git"

    /// the newest git tag is older than some calls these snippets use, so the line follows main. there is
    /// no way to tell at run time which tag first has them: pin one once a tag does.
    static var package: String {
        """
        // Package.swift
        dependencies: [
            // pin a tag for production once one includes these calls
            .package(url: "\(packageURL)", branch: "main"),
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

    /// one line per setting that is not at its default, for the calls of one mode. the lines rebuild
    /// what `PlaygroundSettings.options` builds for that mode, field by field:
    /// - transcribe reads compute, decoding, chunking, keep warm and the model
    /// - record also reads the input device, pre-roll, drafts and permissions, and drafts use `decoding` too
    /// - listen reads `listen.decoding` (draft and final) instead of `decoding`, never chunks, and has
    ///   its own keep warm next to the top-level one
    /// empty when everything the mode reads is at its default.
    static func optionLines(_ settings: PlaygroundSettings, mode: Mode) -> [String] {
        let base = PlaygroundSettings.defaults
        let live = mode == .listen
        let microphone = mode != .transcribe
        var lines: [String] = []
        func quoted(_ text: String) -> String {
            let escaped = text
                .replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n").replacingOccurrences(of: "\r", with: "\\r").replacingOccurrences(of: "\t", with: "\\t")
            return "\"" + escaped + "\""
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

        // compute: parsed the way `options` parses it, so a stale value never prints code that does not compile
        if let policy = ComputeOptions.PlacementPolicy(rawValue: settings.policy), policy != .automatic {
            lines.append("options.compute.policy = .\(policy.rawValue)")
        }
        for stage in Stage.allCases {
            if let pin = settings.chipPins[stage.rawValue], let chip = Chip(rawValue: pin.rawValue) {
                lines.append("options.compute.chips[.\(stage.rawValue)] = .\(chip.rawValue)")
            }
            if let pin = settings.precisionPins[stage.rawValue], pin != .automatic, let precision = Precision(rawValue: pin.rawValue) {
                lines.append("options.compute.precision[.\(stage.rawValue)] = .\(precision.rawValue)")
            }
        }

        // decoding. listen has a draft and a final pass: both share the limits, the language and the
        // context, and only the search differs
        let searchChanged = settings.search != base.search || (settings.search == .beam && settings.beamWidth != base.beamWidth)
        if live {
            if searchChanged { lines.append("options.listen.decoding.final.search = \(search(settings.search, settings.beamWidth))") }
            if settings.draftSearch != base.draftSearch || (settings.draftSearch == .beam && settings.draftBeamWidth != base.draftBeamWidth) {
                lines.append("options.listen.decoding.draft.search = \(search(settings.draftSearch, settings.draftBeamWidth))")
            }
            for pass in ["draft", "final"] {
                if settings.limitTokens { lines.append("options.listen.decoding.\(pass).maxTokens = \(max(1, settings.maxTokens))") }
                if settings.customRepeatGuard { lines.append("options.listen.decoding.\(pass).repeatGuard = \(max(0, settings.repeatGuard))") }
                if !settings.context.isEmpty { lines.append("options.listen.decoding.\(pass).context = \(quoted(settings.context))") }
            }
            // sets the language of the draft and the final pass together
            if settings.language != base.language, !settings.language.isEmpty {
                lines.append("options.listen.language = \(quoted(settings.language))")
            }
        } else {
            if searchChanged { lines.append("options.decoding.search = \(search(settings.search, settings.beamWidth))") }
            if settings.limitTokens { lines.append("options.decoding.maxTokens = \(max(1, settings.maxTokens))") }
            if settings.customRepeatGuard { lines.append("options.decoding.repeatGuard = \(max(0, settings.repeatGuard))") }
            if settings.language != base.language, !settings.language.isEmpty {
                lines.append("options.decoding.language = \(quoted(settings.language))")
            }
            if !settings.context.isEmpty { lines.append("options.decoding.context = \(quoted(settings.context))") }
        }

        // audio
        if microphone, let device = settings.inputDevice { lines.append("options.inputDevice = \(quoted(device))") }
        if !live, !settings.chunkingAutomatic {
            lines.append("options.chunking = .maxPieceLength(.seconds(\(number(settings.maxPieceSeconds))))")
        }
        if live {
            if settings.noise == "off" { lines.append("options.noise = .off") }
            if settings.echoCancellation != base.echoCancellation {
                lines.append("options.echoCancellation = .\(settings.echoCancellation.rawValue)")
            }
            if !settings.gainAutomatic { lines.append("options.gain = .fixed(decibels: \(number(settings.gainDecibels)))") }
        }
        if mode == .record, settings.recordDrafts != base.recordDrafts { lines.append("options.recordDrafts = \(settings.recordDrafts)") }
        if microphone, settings.preRollMilliseconds != base.preRollMilliseconds {
            lines.append("options.preRoll = .milliseconds(\(Int(settings.preRollMilliseconds)))")
        }

        if live {
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
            if let finalPass = ListenOptions.FinalPass(rawValue: settings.finalPass), finalPass != .perPhrase {
                lines.append("options.listen.finalPass = .\(finalPass.rawValue)")
            }
        }

        // lifecycle. a listen session reads its own keep warm, the playground sets both to one value
        if let warm = KeepWarm(rawValue: settings.keepWarm), warm != .automatic {
            lines.append("options.keepWarm = .\(warm.rawValue)")
            if live { lines.append("options.listen.keepWarm = .\(warm.rawValue)") }
        }
        if microphone, settings.manualPermissions { lines.append("options.permissions = .manual") }

        // model
        if let directory = settings.modelDirectory, !directory.isEmpty {
            lines.append("options.model.directory = URL(fileURLWithPath: \(quoted(directory)), isDirectory: true)")
        }
        if !settings.revision.isEmpty, settings.revision != ModelOptions.pinnedRevision {
            lines.append("options.model.revision = \(quoted(settings.revision))")
        }
        if !settings.sha256.isEmpty { lines.append("options.model.sha256 = \(quoted(settings.sha256))") }
        if settings.offline { lines.append("options.model.offline = true") }
        return lines
    }

    static func setup(_ settings: PlaygroundSettings, mode: Mode) -> String {
        let lines = optionLines(settings, mode: mode)
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
            let job = speech.transcriptionJob(contentsOf: url)

            // text as it is worked out: settled never changes, draft may
            Task {
                for await update in job.updates {
                    print(update.settled, update.draft)
                }
            }
            // 0 to 1, moving through the whole file
            Task {
                for await fraction in job.progress { print(fraction) }
            }

            let transcript = try await job.result     // the one to keep
            print(transcript.text)

            // or, when you only want the result
            let same = try await speech.transcribe(contentsOf: url)
            """
        case .record:
            """
            let recorder = try await speech.record()

            // while recording: a level for a meter, and draft text
            Task {
                for await level in recorder.levels { /* draw a meter */ }
            }
            Task {
                for await update in recorder.drafts {
                    print(update.settled, update.draft)
                }
            }

            // when the user stops: the raw audio, then one exact pass over it
            let recording = await recorder.stop()
            let job = speech.transcriptionJob(recording)
            for await update in job.updates { print(update.text) }
            let transcript = try await job.result     // the one to keep
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
        [setup(settings, mode: mode), usage(mode)].joined(separator: "\n\n")
    }
}

#if DEBUG
extension CodeSnippets {
    /// debug only. for settings that differ from the defaults, the option names a mode's snippet sets must be
    /// exactly the fields of `settings.options` that differ from `FourierOptions()` and that the mode reads.
    /// it compares names, not values: the snippet is text, so what the lines assign is checked by reading them.
    static func selfCheck() {
        let changes: [(String, (inout PlaygroundSettings) -> Void)] = [
            ("policy", { $0.policy = "npu" }),
            ("stale policy", { $0.policy = "gone" }),
            ("chip pin", { $0.chipPins["encoder"] = .gpu }),
            ("precision pin", { $0.precisionPins["decoder"] = .float32 }),
            ("beam", { $0.search = .beam; $0.beamWidth = 5 }),
            ("greedy", { $0.search = .greedy }),
            ("tokens", { $0.limitTokens = true; $0.maxTokens = 64 }),
            ("repeat guard", { $0.customRepeatGuard = true; $0.repeatGuard = 0 }),
            ("language", { $0.language = "French" }),
            ("empty language", { $0.language = "" }),
            ("context", { $0.context = "say \"hi\"\nand\\more" }),
            ("input device", { $0.inputDevice = "uid" }),
            ("chunking", { $0.chunkingAutomatic = false; $0.maxPieceSeconds = 12.5 }),
            ("noise", { $0.noise = "off" }),
            ("echo", { $0.echoCancellation = .on }),
            ("gain", { $0.gainAutomatic = false; $0.gainDecibels = -6.5 }),
            ("pre-roll", { $0.preRollMilliseconds = 250 }),
            ("record drafts", { $0.recordDrafts = false }),
            ("drafts", { $0.drafts = false }),
            ("update", { $0.updateAutomatic = false; $0.updateMilliseconds = 120 }),
            ("settle", { $0.settleAfter = 3 }),
            ("pause", { $0.pauseMilliseconds = 400 }),
            ("phrase", { $0.maxPhraseSeconds = 12 }),
            ("final pass", { $0.finalPass = "fullAudio" }),
            ("draft search", { $0.draftSearch = .beam; $0.draftBeamWidth = 4 }),
            ("keep warm", { $0.keepWarm = "never" }),
            ("permissions", { $0.manualPermissions = true }),
            ("directory", { $0.modelDirectory = "/tmp/m" }),
            ("revision", { $0.revision = "abc" }),
            ("pinned revision", { $0.revision = ModelOptions.pinnedRevision }),
            ("sha256", { $0.sha256 = "ff" }),
            ("offline", { $0.offline = true }),
        ]
        var everything = PlaygroundSettings()
        for (_, change) in changes { change(&everything) }
        everything.policy = "npu"
        everything.revision = "abc"
        let cases = changes + [("everything", { $0 = everything })]
        for mode in Mode.allCases {
            let base = fingerprint(FourierOptions(), mode: mode)
            for (name, change) in cases {
                var settings = PlaygroundSettings()
                change(&settings)
                let emitted = Set(optionLines(settings, mode: mode).map { $0.components(separatedBy: " = ")[0] })
                let now = fingerprint(settings.options, mode: mode)
                let differing = Set(now.keys.filter { now[$0] != base[$0] })
                assert(emitted == differing, "code pane, \(mode.title), \(name): snippet sets \(emitted.subtracting(differing).sorted()) that does not differ, and misses \(differing.subtracting(emitted).sorted())")
            }
        }
    }

    /// the fields of the options a mode reads, under the names the snippet assigns them by
    private static func fingerprint(_ options: FourierOptions, mode: Mode) -> [String: String] {
        var fields: [String: String] = [:]
        func put(_ name: String, _ value: Any) { fields[name] = String(describing: value) }
        put("options.compute.policy", options.compute.policy)
        for stage in Stage.allCases {
            put("options.compute.chips[.\(stage.rawValue)]", options.compute.chips[stage] as Any)
            put("options.compute.precision[.\(stage.rawValue)]", options.compute.precision[stage] as Any)
        }
        put("options.keepWarm", options.keepWarm)
        put("options.model.directory", options.model.directory?.path as Any)
        put("options.model.revision", options.model.revision)
        put("options.model.sha256", options.model.sha256 as Any)
        put("options.model.offline", options.model.offline)
        if mode != .listen {
            put("options.decoding.search", options.decoding.search)
            put("options.decoding.maxTokens", options.decoding.maxTokens as Any)
            put("options.decoding.repeatGuard", options.decoding.repeatGuard as Any)
            put("options.decoding.language", options.decoding.language)
            put("options.decoding.context", options.decoding.context as Any)
            put("options.chunking", options.chunking)
        }
        if mode != .transcribe {
            put("options.inputDevice", options.inputDevice as Any)
            put("options.preRoll", options.preRoll)
            put("options.permissions", options.permissions)
        }
        if mode == .record { put("options.recordDrafts", options.recordDrafts) }
        if mode == .listen {
            let listen = options.listen
            put("options.listen.keepWarm", listen.keepWarm)
            put("options.listen.decoding.final.search", listen.decoding.final.search)
            put("options.listen.decoding.draft.search", listen.decoding.draft.search)
            for (pass, decoding) in [("draft", listen.decoding.draft), ("final", listen.decoding.final)] {
                put("options.listen.decoding.\(pass).maxTokens", decoding.maxTokens as Any)
                put("options.listen.decoding.\(pass).repeatGuard", decoding.repeatGuard as Any)
                put("options.listen.decoding.\(pass).context", decoding.context as Any)
            }
            put("options.listen.language", [listen.decoding.draft.language, listen.decoding.final.language])
            put("options.noise", options.noise)
            put("options.echoCancellation", options.echoCancellation)
            put("options.gain", options.gain)
            put("options.listen.drafts", listen.drafts)
            put("options.listen.updateInterval", listen.updateInterval)
            put("options.listen.settleAfter", listen.settleAfter)
            put("options.listen.pauseToEndPhrase", listen.pauseToEndPhrase)
            put("options.listen.maxPhraseLength", listen.maxPhraseLength)
            put("options.listen.finalPass", listen.finalPass)
        }
        return fields
    }
}
#endif

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
        let changed = CodeSnippets.optionLines(settings, mode: mode).count
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
                    code: CodeSnippets.setup(settings, mode: mode)
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
