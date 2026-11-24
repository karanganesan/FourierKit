import SwiftUI

extension View {
    /// puts loading, progress and permission content on a floating liquid glass panel, so
    /// something that is waiting reads as one object above the window and not as loose text.
    public func fourierGlassPanel(cornerRadius: CGFloat = 24, padding: CGFloat = 24) -> some View {
        self
            .padding(padding)
            .glassEffect(.regular, in: .rect(cornerRadius: cornerRadius, style: .continuous))
    }
}

/// the floating bar of live controls: one capsule of system glass that hugs what is in it and
/// grows smoothly when a level meter or a clock joins the main button.
///
/// the glass is the system's own material. it is light in light mode and dark in dark mode, and
/// nothing in the bar is tinted, so text and symbols take the label colour of the appearance.
public struct FourierControlBar<Leading: View, Center: View, Trailing: View>: View {
    private let leading: Leading
    private let center: Center
    private let trailing: Trailing

    public init(
        @ViewBuilder leading: () -> Leading,
        @ViewBuilder center: () -> Center,
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.leading = leading()
        self.center = center()
        self.trailing = trailing()
    }

    public var body: some View {
        HStack(spacing: 8) {
            leading
            center
            trailing
        }
        .padding(.leading, 16)
        .padding(.trailing, 24)
        .frame(height: 64)
        .glassEffect(.regular, in: .capsule)
    }
}
