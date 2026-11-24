import FourierKit
import FourierKitUI
import SwiftUI
import UniformTypeIdentifiers

// MARK: transcript text

/// a finished transcript: a quiet line of facts, then the text, lowercased for display only.
/// copy, share and export use the exact text through the toolbar and menus.
struct TranscriptText: View {
    let text: String
    let facts: [String]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !facts.isEmpty {
                    Text(facts.joined(separator: " · "))
                        .fourierTextStyle(.caption, numbers: .tabular)
                        .fourierInk(.tertiary)
                }
                Text(text.isEmpty ? "no speech was found." : text.lowercased())
                    .fourierTextStyle(.largeTitle)
                    .fourierInk(text.isEmpty ? .tertiary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityLabel(text.isEmpty ? "no speech was found." : text.lowercased())
            }
            .padding(.horizontal, 32)
            .padding(.vertical, 24)
            .frame(maxWidth: 880, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .fourierScrollEdges()
        #if os(macOS)
        .copyable([text])
        #endif
    }

    static func facts(duration: Duration, speed: Double, extra: [String] = []) -> [String] {
        var facts = extra
        facts.append(FourierFormat.clock(duration))
        if speed > 0 { facts.append("\(FourierFormat.speed(speed)) realtime") }
        return facts
    }
}

// MARK: transcribe

struct TranscribeScreen: View {
    @Environment(AppModel.self) private var app
    @State private var dropTargeted = false

    var body: some View {
        @Bindable var transcribe = app.transcribe
        Group {
            if let transcript = transcribe.transcript {
                TranscriptText(
                    text: transcript.text,
                    facts: TranscriptText.facts(
                        duration: transcript.duration, speed: transcript.metrics.realtimeSpeed,
                        extra: [transcribe.fileName].compactMap { $0 }))
            } else if transcribe.isRunning {
                progress
            } else {
                empty
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay {
            if dropTargeted {
                Color.clear
                    .fourierGlass()
                    .padding(16)
                    .allowsHitTesting(false)
                    .transition(.opacity)
            }
        }
        .animation(.smooth(duration: 0.16), value: dropTargeted)
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first(where: Self.isAudio) else { return false }
            transcribe.transcribe(url)
            return true
        } isTargeted: { dropTargeted = $0 }
        .fileImporter(isPresented: $transcribe.choosing, allowedContentTypes: [.audio, .movie]) { result in
            if case .success(let url) = result { transcribe.transcribe(url) }
        }
    }

    private var empty: some View {
        ContentUnavailableView {
            Label {
                Text(Self.emptyTitle).fourierTextStyle(.title)
            } icon: {
                Image(systemName: "waveform")
            }
        } description: {
            Text(app.transcribe.error.map(FourierFormat.error) ?? Self.emptyMessage)
                .fourierTextStyle(.body)
                .foregroundStyle(app.transcribe.error == nil ? Color.primary.opacity(0.64) : Color.red)
        } actions: {
            Button("choose file…") { app.transcribe.choosing = true }
                // ios draws a prominent glass button's label white, which vanishes on the white
                // accent in dark mode, so there it is the plain glass with the label in the text ink
                .fourierGlassButton(prominent: Self.prominentChoose)
                .controlSize(.large)
                .fourierTextStyle(.body)
                .keyboardShortcut(.defaultAction)
        }
    }

    private var progress: some View {
        VStack(spacing: 0) {
            StreamingText(update: app.transcribe.update)
            progressPanel
        }
    }

    private var progressPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(app.transcribe.fileName ?? "audio")
                .fourierTextStyle(.title)
                .lineLimit(1)
                .truncationMode(.middle)
            ProgressView(value: app.transcribe.fraction)
                .progressViewStyle(.linear)
                .tint(.primary)
                .animation(.smooth(duration: 0.4), value: app.transcribe.fraction)
                .accessibilityLabel("transcribing")
            HStack {
                Text("transcribing")
                Spacer()
                Text(FourierFormat.percent(app.transcribe.fraction))
            }
            .fourierTextStyle(.caption, numbers: .tabular)
            .fourierInk(.secondary)
            Button("cancel") { app.transcribe.cancel() }
                .fourierGlassButton()
                .fourierTextStyle(.body)
                .keyboardShortcut(.cancelAction)
        }
        .frame(maxWidth: 400)
        .fourierGlassPanel()
        .padding(32)
    }

    #if os(macOS)
    private static let prominentChoose = true
    private static let emptyTitle = "drop an audio file"
    private static let emptyMessage = "or choose one. it is turned into text on this mac."
    #else
    private static let prominentChoose = false
    private static let emptyTitle = "choose an audio file"
    private static let emptyMessage = "from files, or drop one here. it is turned into text on this device."
    #endif

    static func isAudio(_ url: URL) -> Bool {
        if let type = UTType(filenameExtension: url.pathExtension), type.conforms(to: .audio) || type.conforms(to: .movie) {
            return true
        }
        return TranscribeModel.audioExtensions.contains(url.pathExtension.lowercased())
    }
}

// MARK: record

struct RecordScreen: View {
    @Environment(AppModel.self) private var app
    @FocusState private var focused: Bool

    var body: some View {
        let record = app.record
        VStack(spacing: 0) {
            content(record)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            controls(record)
        }
        .focusable()
        .focusEffectDisabled()
        .focused($focused)
        .onKeyPress(.space) {
            guard canRecord else { return .ignored }
            record.toggle()
            return .handled
        }
        .onAppear { focused = true }
    }

    private var canRecord: Bool {
        let microphone = app.state.microphone
        return microphone.isSetupValid && microphone.permission != .denied && microphone.permission != .restricted
    }

    @ViewBuilder
    private func content(_ record: RecordModel) -> some View {
        if !canRecord {
            FourierPermissionView(app.state)
                .padding(32)
        } else {
            switch record.phase {
            case .recording, .starting:
                VStack(spacing: 24) {
                    FourierLevelMeter(level: record.level, isActive: record.phase == .recording, barCount: 64, barWidth: 4, ink: .primary)
                        .frame(height: 96)
                    TimelineView(.periodic(from: record.startedAt ?? .now, by: 1)) { context in
                        Text(FourierFormat.clock(.seconds(context.date.timeIntervalSince(record.startedAt ?? context.date))))
                            .fourierTextStyle(.largeTitle, numbers: .tabular)
                            .fourierInk(.primary)
                            .accessibilityLabel("recording time")
                    }
                }
                .padding(32)
            case .transcribing:
                VStack(spacing: 0) {
                StreamingText(update: record.update)
                VStack(alignment: .leading, spacing: 16) {
                    Text("transcribing the recording")
                        .fourierTextStyle(.title)
                    ProgressView(value: record.fraction)
                        .progressViewStyle(.linear)
                        .tint(.primary)
                        .animation(.smooth(duration: 0.4), value: record.fraction)
                }
                .frame(maxWidth: 400)
                .fourierGlassPanel()
                .padding(32)
                }
            case .idle:
                if let transcript = record.transcript {
                    TranscriptText(text: transcript.text, facts: TranscriptText.facts(duration: transcript.duration, speed: transcript.metrics.realtimeSpeed))
                } else {
                    VStack(spacing: 8) {
                        Text("record, then read it")
                            .fourierTextStyle(.title)
                        Text(record.error.map(FourierFormat.error) ?? "the recording stays on \(DeviceText.this) and is turned into text when you stop.")
                            .fourierTextStyle(.body)
                            .foregroundStyle(record.error == nil ? Color.primary.opacity(0.64) : Color.red)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: 400)
                    .padding(32)
                }
            }
        }
    }

    private func controls(_ record: RecordModel) -> some View {
        FourierControlBar {
            EmptyView()
        } center: {
            FourierMicButton(status: record.micStatus, diameter: 48, action: record.toggle) { status in
                // red means record, as in voice memos
                Image(systemName: status == .idle ? "circle.fill" : "stop.fill")
                    .font(.system(size: 24, weight: .semibold))
                    .foregroundStyle(status == .idle ? Color.red : Color.primary)
                    .contentTransition(.symbolEffect(.replace))
            }
            .plain()
            .accessibilityLabel(record.isRunning ? "stop recording" : "start recording")
            .disabled(!canRecord)
        } trailing: {
            Text(record.isRunning ? Self.stopHint : Self.recordHint)
                .fourierTextStyle(.caption)
                .fourierInk(.tertiary)
        }
        .frame(height: 96)
        .frame(maxWidth: .infinity)
    }

    #if os(macOS)
    private static let recordHint = "space to record"
    private static let stopHint = "space to stop"
    #else
    private static let recordHint = "tap to record"
    private static let stopHint = "tap to stop"
    #endif
}

/// the text of a file or a recording as it is transcribed: finished pieces in the text colour,
/// what is still being worked out in the draft colour, the newest words kept in view.
private struct StreamingText: View {
    let update: TranscriptionUpdate?

    var body: some View {
        ScrollView {
            FourierLiveText(
                settled: update?.settled ?? "",
                draft: (update?.settled.isEmpty == false && update?.draft.isEmpty == false ? " " : "") + (update?.draft ?? ""))
                .padding(.horizontal, 32)
                .padding(.vertical, 24)
                .frame(maxWidth: 960, alignment: .leading)
                .frame(maxWidth: .infinity)
        }
        .defaultScrollAnchor(.top, for: .alignment)
        .defaultScrollAnchor(.bottom, for: .sizeChanges)
        .fourierScrollEdges()
    }
}

// MARK: listen

struct ListenScreen: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        #if os(macOS)
        FourierLiveView(state: app.state, listen: app.listen)
        #else
        FourierLiveView(state: app.state, listen: app.listen, placeholder: "tap the button and start talking")
        #endif
    }
}

// MARK: recent

struct RecentScreen: View {
    @Environment(AppModel.self) private var app
    let id: UUID

    var body: some View {
        if let item = app.recents.item(id) {
            TranscriptText(
                text: item.text,
                facts: TranscriptText.facts(
                    duration: .seconds(item.duration), speed: item.speed,
                    extra: [item.mode.title, item.date.formatted(date: .abbreviated, time: .shortened).lowercased()]
                        + [item.source].compactMap { $0 }))
        } else {
            ContentUnavailableView {
                Label {
                    Text("transcript deleted").fourierTextStyle(.title)
                } icon: {
                    Image(systemName: "trash")
                }
            } description: {
                Text("undo brings it back.").fourierTextStyle(.body)
            } actions: {
                Button("listen") { app.selection = .mode(.listen) }
                    .fourierGlassButton()
                    .fourierTextStyle(.body)
            }
        }
    }
}
