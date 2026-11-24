import FourierKit
import SwiftUI

/// prepare, permission, mic, level and live text in one drop-in view.
///
/// `FourierLiveView(speech)` just works: it prepares the model, asks for the microphone on the first press,
/// and streams text. space starts and stops while the view has focus. pass your own models to drive the same
/// session from a toolbar or menu, or to keep the transcript.
public struct FourierLiveView<Accessory: View>: View {
    @State private var owned: OwnedModels?
    private let injected: (state: FourierStateModel, listen: FourierListenModel)?
    private let placeholder: String
    private let textStyle: FourierTextStyle
    private let spaceToggles: Bool
    private let accessory: Accessory

    @State private var hasBeenReady = false
    @FocusState private var focused: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// a self-contained live view for `speech`
    /// - parameters:
    ///   - options: listen options, nil uses `speech.options.listen`
    ///   - placeholder: shown before the first words
    ///   - textStyle: the live text's step of the type scale
    ///   - spaceToggles: space starts and stops while the view has focus
    ///   - accessory: extra content beside the controls, e.g. a copy button
    public init(
        _ speech: Fourier,
        options: ListenOptions? = nil,
        placeholder: String = "press space and start talking",
        textStyle: FourierTextStyle = .largeTitle,
        spaceToggles: Bool = true,
        @ViewBuilder accessory: () -> Accessory
    ) {
        owned = OwnedModels(state: FourierStateModel(speech), listen: FourierListenModel(speech, options: options))
        injected = nil
        self.placeholder = placeholder
        self.textStyle = textStyle
        self.spaceToggles = spaceToggles
        self.accessory = accessory()
    }

    /// a live view over models the app owns, so other controls can share the session
    public init(
        state: FourierStateModel,
        listen: FourierListenModel,
        placeholder: String = "press space and start talking",
        textStyle: FourierTextStyle = .largeTitle,
        spaceToggles: Bool = true,
        @ViewBuilder accessory: () -> Accessory
    ) {
        owned = nil
        injected = (state, listen)
        self.placeholder = placeholder
        self.textStyle = textStyle
        self.spaceToggles = spaceToggles
        self.accessory = accessory()
    }

    private var state: FourierStateModel { injected?.state ?? owned!.state }
    private var listen: FourierListenModel { injected?.listen ?? owned!.listen }

    public var body: some View {
        let state = state
        let listen = listen
        VStack(spacing: 0) {
            content(state: state, listen: listen)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            controls(state: state, listen: listen)
        }
        .focusable(spaceToggles)
        .focusEffectDisabled()
        .focused($focused)
        .onKeyPress(.space) {
            guard spaceToggles, canListen(state) else { return .ignored }
            listen.toggle()
            return .handled
        }
        .onAppear { focused = true }
        .onChange(of: state.isReady, initial: true) { _, ready in
            if ready { hasBeenReady = true }
        }
        .task { await state.observe() }
    }

    // MARK: parts

    @ViewBuilder
    private func content(state: FourierStateModel, listen: FourierListenModel) -> some View {
        if showsPrepare(state, listen) {
            FourierPrepareView(state)
                .padding(32)
                .transition(.opacity)
        } else if usesMicrophone(listen), let blocked = blockedPermission(state) {
            FourierPermissionView(state)
                .padding(32)
                .id(blocked)
        } else {
            ScrollView {
                FourierLiveText(listen, placeholder: placeholder, style: textStyle)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 24)
                    .frame(maxWidth: 960, alignment: .leading)
                    .frame(maxWidth: .infinity)
            }
            // short text sits at the top; once it overflows, new lines keep the newest words in view
            .defaultScrollAnchor(.top, for: .alignment)
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
            .scrollIndicators(.automatic)
            .scrollEdgeEffectStyle(.soft, for: .top)
        }
    }

    private func controls(state: FourierStateModel, listen: FourierListenModel) -> some View {
        FourierControlBar {
            if listen.phase == .listening {
                FourierLevelMeter(level: listen.level, isActive: true, barCount: 16)
                    .frame(width: 120, height: 24)
                    .padding(.leading, 8)
                    .transition(.opacity)
            }
        } center: {
            FourierMicButton(listen, diameter: 48)
                .plain()
                .disabled(!canListen(state))
        } trailing: {
            status(listen)
        }
        .overlay(alignment: .trailing) {
            accessory
                .padding(.trailing, 24)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 96)
        .animation(reduceMotion ? nil : .smooth, value: listen.phase)
    }

    private func status(_ listen: FourierListenModel) -> some View {
        Group {
            if let error = listen.error {
                Text(FourierFormat.error(error))
                    .foregroundStyle(.red)
                    .lineLimit(3)
            } else {
                switch listen.phase {
                case .idle:
                    Text(listen.transcript == nil ? "space to start" : "space to start again")
                        .fourierInk(.tertiary)
                case .starting:
                    Text("starting")
                        .fourierInk(.tertiary)
                case .listening:
                    TimelineView(.periodic(from: listen.startedAt ?? .now, by: 1)) { context in
                        Text(FourierFormat.clock(.seconds(context.date.timeIntervalSince(listen.startedAt ?? context.date))))
                            .fourierInk(.secondary)
                    }
                case .finishing:
                    Text("finishing")
                        .fourierInk(.tertiary)
                }
            }
        }
        .fourierTextStyle(.caption, numbers: .tabular)
        .contentTransition(.opacity)
    }

    // MARK: rules

    private func showsPrepare(_ state: FourierStateModel, _ listen: FourierListenModel) -> Bool {
        !hasBeenReady && !state.isReady && listen.phase == .idle
    }

    private func usesMicrophone(_ listen: FourierListenModel) -> Bool {
        (listen.options?.audioSource ?? listen.speech.options.audioSource).usesMicrophone
    }

    private func blockedPermission(_ state: FourierStateModel) -> MicrophoneState.Permission? {
        let microphone = state.microphone
        if !microphone.isSetupValid { return .restricted }
        switch microphone.permission {
        case .denied, .restricted: return microphone.permission
        case .granted, .notDetermined: return nil
        }
    }

    private func canListen(_ state: FourierStateModel) -> Bool {
        !usesMicrophone(listen) || blockedPermission(state) == nil
    }
}

extension FourierLiveView where Accessory == EmptyView {
    public init(
        _ speech: Fourier,
        options: ListenOptions? = nil,
        placeholder: String = "press space and start talking",
        textStyle: FourierTextStyle = .largeTitle,
        spaceToggles: Bool = true
    ) {
        self.init(speech, options: options, placeholder: placeholder, textStyle: textStyle, spaceToggles: spaceToggles) { EmptyView() }
    }

    public init(
        state: FourierStateModel,
        listen: FourierListenModel,
        placeholder: String = "press space and start talking",
        textStyle: FourierTextStyle = .largeTitle,
        spaceToggles: Bool = true
    ) {
        self.init(state: state, listen: listen, placeholder: placeholder, textStyle: textStyle, spaceToggles: spaceToggles) { EmptyView() }
    }
}

@MainActor
private final class OwnedModels {
    let state: FourierStateModel
    let listen: FourierListenModel

    init(state: FourierStateModel, listen: FourierListenModel) {
        self.state = state
        self.listen = listen
    }
}
