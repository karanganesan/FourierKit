import SwiftUI

/// the kit's text and fill levels: the foreground colour at fixed 8% opacity steps.
///
/// increased contrast lifts every step so secondary text stays readable.
public enum FourierInk: Sendable, Hashable {
    /// 100%
    case primary
    /// 64%, labels and supporting text
    case secondary
    /// 56%, live drafts before they settle
    case draft
    /// 40%, hints
    case tertiary
    /// 24%, placeholders and quiet marks
    case quaternary
    /// 16%, fills
    case fill
    /// 8%, the faintest fill
    case faintFill

    public func opacity(increasedContrast: Bool) -> Double {
        switch (self, increasedContrast) {
        case (.primary, _): 1
        case (.secondary, false): 0.64
        case (.secondary, true): 0.88
        case (.draft, false): 0.56
        case (.draft, true): 0.80
        case (.tertiary, false): 0.40
        case (.tertiary, true): 0.64
        case (.quaternary, false): 0.24
        case (.quaternary, true): 0.48
        case (.fill, false): 0.16
        case (.fill, true): 0.32
        case (.faintFill, false): 0.08
        case (.faintFill, true): 0.24
        }
    }
}

extension View {
    /// sets the foreground to the primary colour at one of the kit's opacity steps.
    public func fourierInk(_ ink: FourierInk) -> some View {
        modifier(InkModifier(ink: ink))
    }
}

private struct InkModifier: ViewModifier {
    let ink: FourierInk
    @Environment(\.colorSchemeContrast) private var contrast

    func body(content: Content) -> some View {
        content.foregroundStyle(.primary.opacity(ink.opacity(increasedContrast: contrast == .increased)))
    }
}

extension EnvironmentValues {
    var increasedContrast: Bool { colorSchemeContrast == .increased }
}
