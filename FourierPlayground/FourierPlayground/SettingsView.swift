import FourierKit
import FourierKitUI
import SwiftUI

/// every sdk option, grouped as in the plan, with a reset to defaults. changes apply to the next run.
struct SettingsView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        TabView(selection: $app.settingsTab) {
            Tab("compute", systemImage: "cpu", value: "compute") { ComputeSettings() }
            Tab("decoding", systemImage: "textformat", value: "decoding") { DecodingSettings() }
            Tab("audio", systemImage: "waveform", value: "audio") { AudioSettings() }
            Tab("listen", systemImage: "ear", value: "listen") { ListenSettings() }
            Tab("lifecycle", systemImage: "bolt", value: "lifecycle") { LifecycleSettings() }
            Tab("model", systemImage: "shippingbox", value: "model") { ModelSettings() }
        }
        .frame(width: 560, height: 520)
        .fourierTextStyle(.body)
    }
}

/// a grouped form with the reset button at its foot
private struct SettingsForm<Content: View>: View {
    @Environment(AppModel.self) private var app
    @ViewBuilder let content: Content

    var body: some View {
        Form {
            content
            Section {
                HStack {
                    Text(app.isBusy ? "changes apply when the current run ends" : "changes apply to the next run")
                        .fourierTextStyle(.footnote)
                        .fourierInk(.tertiary)
                    Spacer()
                    Button("reset to defaults") { app.resetSettings() }
                        .disabled(app.settings == .defaults)
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color.black)
    }
}

private struct Hint: View {
    let text: String
    init(_ text: String) { self.text = text }
    var body: some View {
        Text(text)
            .fourierTextStyle(.footnote)
            .fourierInk(.tertiary)
            .fixedSize(horizontal: false, vertical: true)
    }
}

// MARK: compute

private struct ComputeSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        SettingsForm {
            Section {
                Picker("placement", selection: $app.settings.policy) {
                    Text("fastest measured").tag("automatic")
                    Text("prefer neural engine").tag("npu")
                    Text("lowest power").tag("efficiency")
                }
            } footer: {
                Hint("where each stage runs. a pin below wins over this.")
            }
            Section {
                ForEach(Stage.allCases, id: \.self) { stage in
                    Picker(FourierFormat.stage(stage), selection: chipBinding(stage)) {
                        ForEach(PlaygroundSettings.Pin.allCases) { Text($0.title).tag($0) }
                    }
                }
            } header: {
                Text("chip pins")
            }
            Section {
                ForEach(Stage.allCases, id: \.self) { stage in
                    Picker(FourierFormat.stage(stage), selection: precisionBinding(stage)) {
                        ForEach(PlaygroundSettings.PrecisionPin.allCases) { Text($0.rawValue).tag($0) }
                    }
                }
            } header: {
                Text("precision pins")
            } footer: {
                Hint("a pin no shipped path can run makes prepare fail with a clear error.")
            }
        }
    }

    private func chipBinding(_ stage: Stage) -> Binding<PlaygroundSettings.Pin> {
        Binding {
            app.settings.chipPins[stage.rawValue] ?? .automatic
        } set: { pin in
            app.settings.chipPins[stage.rawValue] = pin == .automatic ? nil : pin
        }
    }

    private func precisionBinding(_ stage: Stage) -> Binding<PlaygroundSettings.PrecisionPin> {
        Binding {
            app.settings.precisionPins[stage.rawValue] ?? .automatic
        } set: { pin in
            app.settings.precisionPins[stage.rawValue] = pin == .automatic ? nil : pin
        }
    }
}

// MARK: decoding

private struct DecodingSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        SettingsForm {
            Section {
                Picker("search", selection: $app.settings.search) {
                    ForEach(PlaygroundSettings.Search.allCases) { Text($0.title).tag($0) }
                }
                if app.settings.search == .beam {
                    Stepper("beam width \(app.settings.beamWidth)", value: $app.settings.beamWidth, in: 1...8)
                }
                Toggle("limit tokens", isOn: $app.settings.limitTokens)
                if app.settings.limitTokens {
                    Stepper("at most \(app.settings.maxTokens) tokens", value: $app.settings.maxTokens, in: 16...512, step: 16)
                }
                Toggle("custom repeat guard", isOn: $app.settings.customRepeatGuard)
                if app.settings.customRepeatGuard {
                    Stepper(app.settings.repeatGuard == 0 ? "repeat guard off" : "no repeated \(app.settings.repeatGuard)-word runs",
                            value: $app.settings.repeatGuard, in: 0...16)
                }
            } header: {
                Text("search")
            } footer: {
                Hint("the checkpoint's own settings are the exact path. used by transcribe, record and listen's final pass.")
            }
            Section {
                TextField("language", text: $app.settings.language, prompt: Text("english"))
                TextField("context", text: $app.settings.context, prompt: Text("names and terms the model should know"), axis: .vertical)
                    .lineLimit(2...4)
            } header: {
                Text("text")
            } footer: {
                Hint("the model is english only. context changes the model's input, so it leaves the exact path.")
            }
        }
    }
}

// MARK: audio

private struct AudioSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        SettingsForm {
            Section("input") {
                Picker("microphone", selection: $app.settings.inputDevice) {
                    Text("system default").tag(String?.none)
                    ForEach(app.state.microphone.devices) { device in
                        Text(device.name.lowercased()).tag(Optional(device.id))
                    }
                }
            }
            Section {
                Picker("noise suppression", selection: $app.settings.noise) {
                    Text("automatic").tag("automatic")
                    Text("off").tag("off")
                }
                Picker("echo cancellation", selection: $app.settings.echoCancellation) {
                    ForEach(PlaygroundSettings.Switch.allCases) { Text($0.rawValue).tag($0) }
                }
                Toggle("automatic gain", isOn: $app.settings.gainAutomatic)
                if !app.settings.gainAutomatic {
                    LabeledContent("gain") {
                        Slider(value: $app.settings.gainDecibels, in: -12...24, step: 1)
                        Text(String(format: "%+.0f db", app.settings.gainDecibels))
                            .fourierTextStyle(.body, numbers: .tabular)
                            .frame(width: 56, alignment: .trailing)
                    }
                }
                LabeledContent("pre-roll") {
                    Slider(value: $app.settings.preRollMilliseconds, in: 0...1000, step: 50)
                    Text("\(Int(app.settings.preRollMilliseconds)) ms")
                        .fourierTextStyle(.body, numbers: .tabular)
                        .frame(width: 64, alignment: .trailing)
                }
            } header: {
                Text("live audio")
            } footer: {
                Hint("cleanup applies to listen only. files and recordings go in untouched.")
            }
            Section {
                Toggle("cut long audio automatically", isOn: $app.settings.chunkingAutomatic)
                if !app.settings.chunkingAutomatic {
                    LabeledContent("longest piece") {
                        Slider(value: $app.settings.maxPieceSeconds, in: 1...120, step: 1)
                        Text("\(Int(app.settings.maxPieceSeconds)) s")
                            .fourierTextStyle(.body, numbers: .tabular)
                            .frame(width: 56, alignment: .trailing)
                    }
                }
            } header: {
                Text("long files")
            } footer: {
                Hint("audio up to 120 s is one pass. longer audio is cut at quiet points.")
            }
        }
    }
}

// MARK: listen

private struct ListenSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        SettingsForm {
            Section("live text") {
                Toggle("show drafts", isOn: $app.settings.drafts)
                Picker("draft search", selection: $app.settings.draftSearch) {
                    ForEach(PlaygroundSettings.Search.allCases) { Text($0.title).tag($0) }
                }
                if app.settings.draftSearch == .beam {
                    Stepper("draft beam width \(app.settings.draftBeamWidth)", value: $app.settings.draftBeamWidth, in: 1...8)
                }
                Toggle("automatic update rate", isOn: $app.settings.updateAutomatic)
                if !app.settings.updateAutomatic {
                    LabeledContent("update every") {
                        Slider(value: $app.settings.updateMilliseconds, in: 100...1000, step: 50)
                        Text("\(Int(app.settings.updateMilliseconds)) ms")
                            .fourierTextStyle(.body, numbers: .tabular)
                            .frame(width: 64, alignment: .trailing)
                    }
                }
                Stepper("settle after \(app.settings.settleAfter) agreeing passes", value: $app.settings.settleAfter, in: 1...5)
            }
            Section("phrases") {
                LabeledContent("pause to end a phrase") {
                    Slider(value: $app.settings.pauseMilliseconds, in: 200...2000, step: 50)
                    Text("\(Int(app.settings.pauseMilliseconds)) ms")
                        .fourierTextStyle(.body, numbers: .tabular)
                        .frame(width: 64, alignment: .trailing)
                }
                LabeledContent("longest phrase") {
                    Slider(value: $app.settings.maxPhraseSeconds, in: 5...60, step: 1)
                    Text("\(Int(app.settings.maxPhraseSeconds)) s")
                        .fourierTextStyle(.body, numbers: .tabular)
                        .frame(width: 64, alignment: .trailing)
                }
            }
            Section {
                Picker("final pass", selection: $app.settings.finalPass) {
                    Text("each phrase as it closes").tag("perPhrase")
                    Text("all audio at the end").tag("fullAudio")
                }
            } header: {
                Text("final transcript")
            } footer: {
                Hint("the final text comes from exact passes and is the one that is saved.")
            }
        }
    }
}

// MARK: lifecycle

private struct LifecycleSettings: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        SettingsForm {
            Section {
                Picker("keep warm", selection: $app.settings.keepWarm) {
                    Text("automatic").tag("automatic")
                    Text("always").tag("always")
                    Text("never").tag("never")
                }
            } header: {
                Text("engine")
            } footer: {
                Hint("automatic keeps the model loaded and lets go after 5 idle minutes, under memory pressure or when the mac is hot.")
            }
            Section {
                Toggle("ask for the microphone myself", isOn: $app.settings.manualPermissions)
                LabeledContent("microphone") {
                    Text(permissionText)
                        .fourierInk(.secondary)
                }
            } header: {
                Text("permissions")
            } footer: {
                Hint("by default the first record or listen asks, never app launch.")
            }
            Section {
                HStack {
                    Button("warm now") { Task { try? await app.speech.warm() } }
                    Button("cool now") { Task { await app.speech.cool() } }
                    Spacer()
                    Text(FourierFormat.engine(app.state.state.engine))
                        .fourierInk(.secondary)
                }
            }
        }
    }

    private var permissionText: String {
        switch app.state.microphone.permission {
        case .notDetermined: "not asked yet"
        case .granted: "allowed"
        case .denied: "off"
        case .restricted: "blocked"
        }
    }
}

// MARK: model

private struct ModelSettings: View {
    @Environment(AppModel.self) private var app
    @State private var choosingFolder = false

    var body: some View {
        @Bindable var app = app
        SettingsForm {
            Section {
                LabeledContent("folder") {
                    Text(app.settings.modelDirectory ?? ModelOptions.defaultDirectory.path)
                        .fourierTextStyle(.footnote)
                        .fourierInk(.secondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                }
                HStack {
                    Button("choose folder…") { choosingFolder = true }
                    if app.settings.modelDirectory != nil {
                        Button("use default") { app.settings.modelDirectory = nil }
                    }
                }
            } header: {
                Text("location")
            }
            Section {
                TextField("revision", text: $app.settings.revision)
                    .fourierTextStyle(.footnote, numbers: .tabular)
                TextField("sha256", text: $app.settings.sha256, prompt: Text("pinned"))
                    .fourierTextStyle(.footnote, numbers: .tabular)
                Toggle("offline", isOn: $app.settings.offline)
            } header: {
                Text("checkpoint")
            } footer: {
                Hint("a revision other than the pinned one needs its own sha256. offline never touches the network.")
            }
        }
        .fileImporter(isPresented: $choosingFolder, allowedContentTypes: [.folder]) { result in
            if case .success(let url) = result { app.settings.modelDirectory = url.path }
        }
    }
}
