import FourierKit
import FourierKitUI
import Foundation
import Observation

enum Mode: String, Codable, CaseIterable, Identifiable, Sendable {
    case transcribe, record, listen
    var id: Self { self }

    var title: String { rawValue }

    var symbol: String {
        switch self {
        case .transcribe: "doc.text"
        case .record: "record.circle"
        case .listen: "waveform"
        }
    }
}

/// what the sidebar has selected, stored as a string for scene restoration
enum SidebarItem: Hashable, RawRepresentable {
    case mode(Mode)
    case recent(UUID)

    init?(rawValue: String) {
        if let mode = Mode(rawValue: rawValue) {
            self = .mode(mode)
        } else if rawValue.hasPrefix("recent."), let id = UUID(uuidString: String(rawValue.dropFirst(7))) {
            self = .recent(id)
        } else {
            return nil
        }
    }

    var rawValue: String {
        switch self {
        case .mode(let mode): mode.rawValue
        case .recent(let id): "recent.\(id.uuidString)"
        }
    }
}

/// the app's state: one sdk, the kit's models, the three modes and recent transcripts.
@Observable
@MainActor
final class AppModel {
    private(set) var speech: Fourier
    let state: FourierStateModel
    let listen: FourierListenModel
    let transcribe: TranscribeModel
    let record: RecordModel
    let recents = RecentStore()
    let launchOptions = LaunchOptions.current

    var settings: PlaygroundSettings {
        didSet {
            guard settings != oldValue else { return }
            settings.save()
            applySettings()
        }
    }

    /// true once the model was ready in this run; after that the modes never block on the prepare screen
    private(set) var hasBeenReady = false
    var selection: SidebarItem = .mode(.listen)
    var showStats = false {
        didSet { if showStats, showCode { showCode = false } }
    }
    var showSidebar = true
    /// the code pane and the stats pane share the trailing side, one at a time
    var showCode = false {
        didSet { if showCode { showStats = false } }
    }
    /// bumped to ask settings to open on a tab
    var settingsTab = "compute"

    init() {
        let settings = PlaygroundSettings.load()
        self.settings = settings
        let speech = Fourier(settings.options)
        self.speech = speech
        state = FourierStateModel(speech)
        listen = FourierListenModel(speech)
        transcribe = TranscribeModel(speech)
        record = RecordModel(speech)
        listen.onFinish = { [weak self] transcript in self?.save(transcript, mode: .listen, source: nil) }
        transcribe.onFinish = { [weak self] transcript, name in self?.save(transcript, mode: .transcribe, source: name) }
        record.onFinish = { [weak self] transcript in self?.save(transcript, mode: .record, source: nil) }
    }

    /// true while any mode is capturing or transcribing
    var isBusy: Bool { listen.isRunning || transcribe.isRunning || record.isRunning }

    var mode: Mode? {
        if case .mode(let mode) = selection { return mode }
        return nil
    }

    // MARK: launch

    func launch() async {
        if let mode = launchOptions.mode { selection = .mode(mode) }
        if launchOptions.showStats { showStats = true }
        if launchOptions.showCode { showCode = true }
        if let tab = launchOptions.settingsTab { settingsTab = tab }
        await recents.load()
        if let index = launchOptions.recent, recents.items.indices.contains(index) {
            selection = .recent(recents.items[index].id)
        }
        await state.prepare()
        if state.isReady || state.prepareError == nil { hasBeenReady = true }
        // automation waits for the first window layout to settle, so the run measures the work, not the launch
        if launchOptions.openFile != nil || launchOptions.listenFile != nil || launchOptions.recordFile != nil {
            try? await Task.sleep(for: .seconds(1.5))
        }
        runLaunchActions()
    }

    func markReady() {
        hasBeenReady = true
    }

    private func runLaunchActions() {
        if let url = launchOptions.openFile {
            transcribe.transcribe(url)
        }
        if let url = launchOptions.listenFile {
            var options = speech.options.listen
            options.audioSource = FileAudioSource(url: url, realtime: true)
            listen.options = options
            listen.start()
        }
        if let url = launchOptions.recordFile {
            record.source = FileAudioSource(url: url, realtime: true)
            record.start()
        }
    }

    // MARK: settings

    private func applySettings() {
        let speech = Fourier(settings.options)
        self.speech = speech
        state.speech = speech
        listen.speech = speech
        transcribe.speech = speech
        record.speech = speech
        listen.options = nil
    }

    func resetSettings() {
        settings = PlaygroundSettings.defaults
    }

    // MARK: transcripts

    private func save(_ transcript: Transcript, mode: Mode, source: String?) {
        guard !transcript.text.isEmpty else { return }
        recents.add(RecentTranscript(
            mode: mode, text: transcript.text, source: source,
            duration: transcript.duration.timeInterval, speed: transcript.metrics.realtimeSpeed))
    }

    /// the exact text the current screen shows, for copy, share and export
    var currentText: String? {
        switch selection {
        case .mode(.transcribe): transcribe.transcript?.text
        case .mode(.record): record.transcript?.text
        case .mode(.listen):
            listen.transcript?.text ?? listen.update.map { $0.text }.flatMap { $0.isEmpty ? nil : $0 }
        case .recent(let id): recents.item(id)?.text
        }
    }

    /// start or stop for the current mode
    func toggleCurrent() {
        switch selection {
        case .mode(.listen): listen.toggle()
        case .mode(.record): record.toggle()
        case .mode(.transcribe):
            if transcribe.isRunning { transcribe.cancel() } else { transcribe.choosing = true }
        case .recent: break
        }
    }

    var canToggleCurrent: Bool {
        switch selection {
        case .mode: true
        case .recent: false
        }
    }

    var toggleTitle: String {
        switch selection {
        case .mode(.listen): listen.isRunning ? "stop listening" : "start listening"
        case .mode(.record): record.isRunning ? "stop recording" : "start recording"
        case .mode(.transcribe): transcribe.isRunning ? "cancel" : "open audio file…"
        case .recent: "start"
        }
    }
}

extension Duration {
    var timeInterval: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
