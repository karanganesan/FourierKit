import FourierKit
import SwiftUI

/// the microphone's permission, with the one action that helps.
///
/// never asked: an allow button. denied: what to do and a button to open system settings.
/// restricted: the reason, with no dead-end button. setup missing: what the app is missing.
/// granted: `granted` content, empty by default.
public struct FourierPermissionView<Granted: View>: View {
    private let model: FourierStateModel
    private let reason: String
    private let granted: Granted

    /// - parameters:
    ///   - reason: one line on why the app needs the microphone
    ///   - granted: shown once access is granted
    public init(
        _ model: FourierStateModel,
        reason: String = "speech is turned into text on this mac. nothing leaves it.",
        @ViewBuilder granted: () -> Granted
    ) {
        self.model = model
        self.reason = reason
        self.granted = granted()
    }

    public var body: some View {
        let microphone = model.microphone
        Group {
            if !microphone.isSetupValid {
                message(
                    symbol: "exclamationmark.triangle",
                    title: "microphone setup missing",
                    detail: "this app has no microphone usage description or audio entitlement, so it cannot ask for access.")
            } else {
                switch microphone.permission {
                case .granted:
                    granted
                case .notDetermined:
                    message(symbol: "mic", title: "allow the microphone", detail: reason) {
                        Button("allow microphone") { Task { await model.requestMicrophone() } }
                            .buttonStyle(.glassProminent)
                            .fourierTextStyle(.body.weight(500))
                            .controlSize(.large)
                    }
                case .denied:
                    message(
                        symbol: "mic.slash",
                        title: "microphone is off",
                        detail: "turn it on in system settings > privacy & security > microphone, then come back.") {
                            Button("open system settings") { model.openMicrophoneSettings() }
                                .buttonStyle(.glass)
                                .fourierTextStyle(.body.weight(500))
                                .controlSize(.large)
                        }
                case .restricted:
                    message(
                        symbol: "lock",
                        title: "microphone is blocked",
                        detail: "a device policy blocks the microphone on this mac. ask the person who manages it.")
                }
            }
        }
        .task { await model.refreshMicrophone() }
    }

    private func message(symbol: String, title: String, detail: String) -> some View {
        message(symbol: symbol, title: title, detail: detail) { EmptyView() }
    }

    private func message(symbol: String, title: String, detail: String, @ViewBuilder action: () -> some View) -> some View {
        VStack(spacing: 16) {
            Image(systemName: symbol)
                .font(.system(size: 32, weight: .light))
                .fourierInk(.secondary)
                .accessibilityHidden(true)
            VStack(spacing: 8) {
                Text(title)
                    .fourierTextStyle(.title)
                    .fourierInk(.primary)
                Text(detail.lowercased())
                    .fourierTextStyle(.body)
                    .fourierInk(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            action()
        }
        .frame(maxWidth: 400)
        .accessibilityElement(children: .contain)
    }
}

extension FourierPermissionView where Granted == EmptyView {
    public init(_ model: FourierStateModel, reason: String = "speech is turned into text on this mac. nothing leaves it.") {
        self.init(model, reason: reason) { EmptyView() }
    }
}
