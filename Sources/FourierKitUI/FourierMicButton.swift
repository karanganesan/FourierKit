import FourierKit
import SwiftUI

/// what a mic button shows.
public enum FourierMicStatus: Sendable, Hashable {
    case idle
    /// preparing, asking or starting capture
    case starting
    case running
    /// waiting for the final text
    case finishing

    public init(_ phase: FourierListenModel.Phase) {
        switch phase {
        case .idle: self = .idle
        case .starting: self = .starting
        case .listening: self = .running
        case .finishing: self = .finishing
        }
    }
}

/// a round glass start and stop button whose symbol follows the live state.
public struct FourierMicButton<Label: View>: View {
    private let status: FourierMicStatus
    private let diameter: CGFloat
    private let action: () -> Void
    private let label: (FourierMicStatus) -> Label
    private var isPlain = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// - parameters:
    ///   - status: idle, starting, running or finishing
    ///   - diameter: the button's size, on the 8 pt grid
    ///   - action: starts when idle, stops when running
    ///   - label: custom content for each status
    public init(
        status: FourierMicStatus,
        diameter: CGFloat = 64,
        action: @escaping () -> Void,
        @ViewBuilder label: @escaping (FourierMicStatus) -> Label
    ) {
        self.status = status
        self.diameter = diameter
        self.action = action
        self.label = label
    }

    /// the button without glass of its own, for use inside a glass bar such as `FourierControlBar`
    public func plain() -> FourierMicButton {
        var copy = self
        copy.isPlain = true
        return copy
    }

    public var body: some View {
        Group {
            if isPlain {
                button.buttonStyle(.borderless)
            } else {
                // system glass, untinted: light in light mode, dark in dark mode
                button.buttonStyle(.glass).buttonBorderShape(.circle)
            }
        }
        .foregroundStyle(.primary)
        .disabled(status == .finishing)
        .accessibilityLabel(accessibilityLabel)
        .animation(reduceMotion ? nil : .smooth, value: status)
    }

    private var button: some View {
        Button(action: action) {
            label(status)
                .frame(width: diameter, height: diameter)
                .contentShape(.circle)
        }
    }

    private var accessibilityLabel: String {
        switch status {
        case .idle: "start"
        case .starting: "starting, press to cancel"
        case .running: "stop"
        case .finishing: "finishing"
        }
    }
}

extension FourierMicButton where Label == FourierMicSymbol {
    public init(status: FourierMicStatus, diameter: CGFloat = 64, action: @escaping () -> Void) {
        self.init(status: status, diameter: diameter, action: action) { status in
            FourierMicSymbol(status: status, size: diameter * 0.375)
        }
    }

    /// a button bound to a listen model: starts and stops its session
    public init(_ listen: FourierListenModel, diameter: CGFloat = 64) {
        self.init(status: FourierMicStatus(listen.phase), diameter: diameter) { listen.toggle() }
    }
}

/// the default mic button symbol: a mic when idle, a stop square while running, pulsing while it waits.
public struct FourierMicSymbol: View {
    let status: FourierMicStatus
    let size: CGFloat
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    public init(status: FourierMicStatus, size: CGFloat = 24) {
        self.status = status
        self.size = size
    }

    public var body: some View {
        Image(systemName: status == .idle ? "mic.fill" : "stop.fill")
            .font(.system(size: size, weight: .semibold))
            .contentTransition(.symbolEffect(.replace))
            .symbolEffect(.pulse, options: .repeating, isActive: !reduceMotion && (status == .starting || status == .finishing))
    }
}
