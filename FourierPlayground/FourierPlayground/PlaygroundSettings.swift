import FourierKit
import Foundation

/// every sdk option the playground exposes, saved as json in user defaults. `FourierOptions()` is the default.
struct PlaygroundSettings: Codable, Equatable {
    enum Search: String, Codable, CaseIterable, Identifiable {
        case automatic, greedy, beam
        var id: Self { self }
        var title: String {
            switch self {
            case .automatic: "checkpoint (beam 3)"
            case .greedy: "greedy"
            case .beam: "beam"
            }
        }
    }

    enum Switch: String, Codable, CaseIterable, Identifiable {
        case automatic, on, off
        var id: Self { self }
    }

    enum Pin: String, Codable, CaseIterable, Identifiable {
        case automatic, cpu, gpu, npu
        var id: Self { self }
        var title: String { self == .npu ? "neural engine" : rawValue }
    }

    enum PrecisionPin: String, Codable, CaseIterable, Identifiable {
        case automatic, float32, float16, bfloat16
        var id: Self { self }
    }

    // compute
    var policy = "automatic"
    var chipPins: [String: Pin] = [:]
    var precisionPins: [String: PrecisionPin] = [:]

    // decoding (transcribe and record)
    var search = Search.automatic
    var beamWidth = 3
    var limitTokens = false
    var maxTokens = 512
    var customRepeatGuard = false
    var repeatGuard = 8
    var language = "English"
    var context = ""

    // audio
    var inputDevice: String?
    var chunkingAutomatic = true
    var maxPieceSeconds = 30.0
    var noise = "automatic"
    var echoCancellation = Switch.automatic
    var gainAutomatic = true
    var gainDecibels = 0.0
    var preRollMilliseconds = 500.0

    // listen
    var drafts = true
    var updateAutomatic = true
    var updateMilliseconds = 300.0
    var settleAfter = 2
    var pauseMilliseconds = 600.0
    var maxPhraseSeconds = 24.0
    var finalPass = "perPhrase"
    var draftSearch = Search.greedy
    var draftBeamWidth = 2

    // lifecycle
    var keepWarm = "automatic"
    var manualPermissions = false

    // model
    var modelDirectory: String?
    var revision = ModelOptions.pinnedRevision
    var sha256 = ""
    var offline = false

    static let defaults = PlaygroundSettings()

    // MARK: options

    var options: FourierOptions {
        var options = FourierOptions()

        options.compute.policy = ComputeOptions.PlacementPolicy(rawValue: policy) ?? .automatic
        for stage in Stage.allCases {
            if let pin = chipPins[stage.rawValue], let chip = Chip(rawValue: pin.rawValue) { options.compute.chips[stage] = chip }
            if let pin = precisionPins[stage.rawValue], pin != .automatic, let precision = Precision(rawValue: pin.rawValue) {
                options.compute.precision[stage] = precision
            }
        }

        options.decoding = decoding(search: search, beamWidth: beamWidth)

        options.inputDevice = inputDevice
        options.chunking = chunkingAutomatic ? .automatic : .maxPieceLength(.seconds(maxPieceSeconds))
        options.noise = noise == "off" ? .off : .automatic
        options.echoCancellation = AudioSwitch(rawValue: echoCancellation.rawValue) ?? .automatic
        options.gain = gainAutomatic ? .automatic : .fixed(decibels: Float(gainDecibels))
        options.preRoll = .milliseconds(Int(preRollMilliseconds))

        options.listen.drafts = drafts
        options.listen.updateInterval = updateAutomatic ? .automatic : .fixed(.milliseconds(Int(updateMilliseconds)))
        options.listen.settleAfter = max(1, settleAfter)
        options.listen.pauseToEndPhrase = .milliseconds(Int(pauseMilliseconds))
        options.listen.maxPhraseLength = .seconds(maxPhraseSeconds)
        options.listen.finalPass = ListenOptions.FinalPass(rawValue: finalPass) ?? .perPhrase
        options.listen.decoding = ListenOptions.Decoding(
            draft: decoding(search: draftSearch, beamWidth: draftBeamWidth),
            final: decoding(search: search, beamWidth: beamWidth))

        let warm = KeepWarm(rawValue: keepWarm) ?? .automatic
        options.keepWarm = warm
        options.listen.keepWarm = warm
        options.permissions = manualPermissions ? .manual : .automatic

        let directory = modelDirectory.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let pinned = revision.isEmpty || revision == ModelOptions.pinnedRevision
        options.model = ModelOptions(
            directory: directory,
            revision: pinned ? nil : revision,
            sha256: sha256.isEmpty ? nil : sha256,
            offline: offline)
        return options
    }

    private func decoding(search: Search, beamWidth: Int) -> DecodingOptions {
        var decoding = DecodingOptions()
        switch search {
        case .automatic: decoding.search = .automatic
        case .greedy: decoding.search = .greedy
        case .beam: decoding.search = .beam(max(1, beamWidth))
        }
        decoding.maxTokens = limitTokens ? max(1, maxTokens) : nil
        decoding.repeatGuard = customRepeatGuard ? max(0, repeatGuard) : nil
        decoding.language = language.isEmpty ? "English" : language
        decoding.context = context.isEmpty ? nil : context
        return decoding
    }

    // MARK: storage

    private static let key = "settings"

    static func load() -> PlaygroundSettings {
        guard let data = UserDefaults.standard.data(forKey: key),
              let settings = try? JSONDecoder().decode(PlaygroundSettings.self, from: data)
        else { return PlaygroundSettings() }
        return settings
    }

    func save() {
        guard let data = try? JSONEncoder().encode(self) else { return }
        UserDefaults.standard.set(data, forKey: Self.key)
    }
}
