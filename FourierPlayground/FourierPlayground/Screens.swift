import FourierKit
import FourierKitUI
import SwiftUI
import UniformTypeIdentifiers

// MARK: transcript text

/// a finished transcript on black: a quiet line of facts, then the text, lowercased for display only.
/// copy, share and export use the exact text through the toolbar and menus.
struct TranscriptText: View {
    let text: String
    let facts: [String]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if !facts.isEmpty {
                    Text(facts.joined(separator: " · "))
                        .fourierTextStyle(.footnote, numbers: .tabular)
                        .fourierInk(.tertiary)
                }
                Text(text.isEmpty ? "no speech was found." : text.lowercased())
                    .fourierTextStyle(.title)
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
        .copyable([text])
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
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(.white.opacity(0.08))
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
                Text("drop an audio file").fourierTextStyle(.title)
            } icon: {
                Image(systemName: "waveform")
            }
        } description: {
            Text(app.transcribe.error.map(FourierFormat.error) ?? "or choose one. it is turned into text on this mac.")
                .fourierTextStyle(.body)
                .foregroundStyle(app.transcribe.error == nil ? Color.secondary : Color.red)
        } actions: {
            Button("choose file…") { app.transcribe.choosing = true }
                .buttonStyle(.glassProminent)
                .controlSize(.large)
                .fourierTextStyle(.body.weight(500))
                .keyboardShortcut(.defaultAction)
        }
    }

    private var progress: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(app.transcribe.fileName ?? "audio")
                .fourierTextStyle(.title)
                .lineLimit(1)
                .truncationMode(.middle)
            ProgressView(value: app.transcribe.fraction)
                .progressViewStyle(.linear)
                .tint(.white)
                .animation(.smooth(duration: 0.4), value: app.transcribe.fraction)
                .accessibilityLabel("transcribing")
            HStack {
                Text("transcribing")
                Spacer()
                Text(FourierFormat.percent(app.transcribe.fraction))
            }
            .fourierTextStyle(.footnote, numbers: .tabular)
            .fourierInk(.secondary)
            Button("cancel") { app.transcribe.cancel() }
                .buttonStyle(.glass)
                .fourierTextStyle(.body.weight(500))
                .keyboardShortcut(.cancelAction)
        }
        .frame(maxWidth: 400)
        .padding(32)
    }

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
                            .fourierTextStyle(.display, numbers: .tabular)
                            .fourierInk(.primary)
                            .accessibilityLabel("recording time")
                    }
                }
                .padding(32)
            case .transcribing:
                VStack(alignment: .leading, spacing: 16) {
                    Text("transcribing the recording")
                        .fourierTextStyle(.title)
                    ProgressView(value: record.fraction)
                        .progressViewStyle(.linear)
                        .tint(.white)
                        .animation(.smooth(duration: 0.4), value: record.fraction)
                }
                .frame(maxWidth: 400)
                .padding(32)
            case .idle:
                if let transcript = record.transcript {
                    TranscriptText(text: transcript.text, facts: TranscriptText.facts(duration: transcript.duration, speed: transcript.metrics.realtimeSpeed))
                } else {
                    VStack(spacing: 8) {
                        Text("record, then read it")
                            .fourierTextStyle(.title)
                        Text(record.error.map(FourierFormat.error) ?? "the recording stays on this mac and is turned into text when you stop.")
                            .fourierTextStyle(.body)
                            .foregroundStyle(record.error == nil ? Color.secondary : Color.red)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: 400)
                    .padding(32)
                }
            }
        }
    }

    private func controls(_ record: RecordModel) -> some View {
        VStack(spacing: 8) {
            FourierMicButton(status: record.micStatus, diameter: 80, action: record.toggle) { status in
                if status == .idle {
                    Color.clear
                } else {
                    FourierMicSymbol(status: status, size: 28)
                }
            }
            .overlay {
                // red means record, as in voice memos. drawn above the glass so the colour stays solid
                if record.phase == .idle {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 32, height: 32)
                        .allowsHitTesting(false)
                        .transition(.scale(scale: 0.5).combined(with: .opacity))
                }
            }
            .animation(.smooth(duration: 0.24), value: record.phase)
            .accessibilityLabel(record.isRunning ? "stop recording" : "start recording")
            .disabled(!canRecord)
            Text(record.isRunning ? "space to stop" : "space to record")
                .fourierTextStyle(.footnote)
                .fourierInk(.tertiary)
        }
        .frame(height: 144)
        .frame(maxWidth: .infinity)
    }
}

// MARK: listen

struct ListenScreen: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        FourierLiveView(state: app.state, listen: app.listen)
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
                    .buttonStyle(.glass)
                    .fourierTextStyle(.body.weight(500))
            }
        }
    }
}
