import SwiftUI

/// the kit's text and fill levels: the label colour of the current appearance at fixed opacity.
///
/// three levels for text and two for fills, and nothing in between. the label colour is dark in
/// light mode and light in dark mode, so every level adapts on its own. increased contrast lifts
/// each step so secondary text stays readable.
public enum FourierInk: Sendable, Hashable {
    /// 100%: text, values, symbols
    case primary
    /// 64%: labels, descriptions, live drafts before they settle
    case secondary
    /// 40%: hints, section headers, placeholders
    case tertiary
    /// 16%: tracks and fills
    case fill
    /// 8%: the faintest fill
    case faintFill

    /// the same level as `secondary`. a draft is secondary text that has not settled yet.
    public static let draft = FourierInk.secondary

    public func opacity(increasedContrast: Bool) -> Double {
        switch (self, increasedContrast) {
        case (.primary, _): 1
        case (.secondary, false): 0.64
        case (.secondary, true): 0.88
        case (.tertiary, false): 0.40
        case (.tertiary, true): 0.64
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
