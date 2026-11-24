import SwiftUI

/// the one place glass comes from. every surface in the kit and the app, from the window itself
/// to a panel, a bar or a button, takes its glass from here, so they all match and one edit
/// changes them all.
public enum FourierGlass {
    /// the system's regular liquid glass, untinted. it is light in light mode and dark in dark
    /// mode, the same glass the toolbar's buttons are made of.
    public static var surface: Glass { .regular }
    /// the material of the window itself, behind every pane
    public static var window: Material { .ultraThinMaterial }
    /// corner radius of a floating panel, on the 8 pt grid
    public static let panelRadius: CGFloat = 24
    /// padding inside a floating panel
    public static let panelPadding: CGFloat = 24
    /// height of the floating control bar
    public static let barHeight: CGFloat = 64
}

extension View {
    /// the kit's glass behind this view, in any shape
    public func fourierGlass(in shape: some Shape = .rect(cornerRadius: FourierGlass.panelRadius, style: .continuous)) -> some View {
        glassEffect(FourierGlass.surface, in: shape)
    }

    /// the whole window as one translucent sheet, edge to edge, with nothing painted on it.
    ///
    /// liquid glass bends what is inside its own window, so it cannot be the window: a window made
    /// of it has nothing behind it to bend. the window uses the system material that blurs what is
    /// behind the window instead, which is the closest match and also light or dark with the system.
    public func fourierGlassWindow() -> some View {
        containerBackground(FourierGlass.window, for: .window)
    }

    /// puts loading, progress and permission content on a floating panel of the same glass
    public func fourierGlassPanel(cornerRadius: CGFloat = FourierGlass.panelRadius, padding: CGFloat = FourierGlass.panelPadding) -> some View {
        self
            .padding(padding)
            .fourierGlass(in: .rect(cornerRadius: cornerRadius, style: .continuous))
    }

    /// a button made of the same glass. prominent is for the one main action of a screen.
    @ViewBuilder
    public func fourierGlassButton(prominent: Bool = false) -> some View {
        if prominent {
            buttonStyle(.glassProminent)
        } else {
            buttonStyle(.glass)
        }
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
        .frame(height: FourierGlass.barHeight)
        .fourierGlass(in: .capsule)
    }
}
