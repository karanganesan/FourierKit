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
    /// height of the fade at the top and bottom of anything that scrolls
    public static let edgeFade: CGFloat = 40
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
        #if os(macOS)
        containerBackground(FourierGlass.window, for: .window)
        #else
        // an ios scene has no window background to set, so the material fills the screen behind the view
        background { Rectangle().fill(FourierGlass.window).ignoresSafeArea() }
        #endif
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

extension View {
    /// soft edges on anything that scrolls up and down: what passes under the top or the bottom
    /// fades out progressively, gone at the very edge and whole a band's height in. an edge only
    /// shows while content is past it, and is as strong as how far past it is, so there is nothing
    /// at the top of a list or at its end. the top band reaches up under the toolbar, so rows never
    /// collide with its buttons or the window controls.
    public func fourierScrollEdges(top: Bool = true, bottom: Bool = true) -> some View {
        modifier(ScrollEdges(top: top, bottom: bottom))
    }
}

private struct ScrollEdges: ViewModifier {
    let top: Bool
    let bottom: Bool
    // only the bands observe the position. the scrolling content never re-renders for it.
    @State private var state = ScrollEdgeState()

    func body(content: Content) -> some View {
        content
            .onScrollGeometryChange(for: ScrollEdgeState.Position.self) { geometry in
                ScrollEdgeState.Position(
                    y: geometry.contentOffset.y + geometry.contentInsets.top,
                    end: max(0, geometry.contentSize.height + geometry.contentInsets.bottom - geometry.containerSize.height))
            } action: { _, position in
                state.moved(position)
            }
            // the content itself fades out at an edge, instead of a blurred band laid over it. on a
            // see-through window a band of material is lighter than what is around it and shows as a strip.
            .mask { ScrollEdgeMask(state: state, top: top, bottom: bottom) }
    }
}

@Observable
private final class ScrollEdgeState {
    struct Position: Equatable {
        var y: CGFloat
        var end: CGFloat
    }

    var topLevel = 0.0
    var bottomLevel = 0.0

    /// position alone sets the edges, with no animation of their own
    func moved(_ position: Position) {
        let scrolls = position.end > 4
        let height = FourierGlass.edgeFade
        let top = scrolls ? min(1, max(0, position.y) / height) : 0
        let bottom = scrolls ? min(1, max(0, position.end - position.y) / height) : 0
        guard top != topLevel || bottom != bottomLevel else { return }
        var still = Transaction()
        still.disablesAnimations = true
        withTransaction(still) {
            if top != topLevel { topLevel = top }
            if bottom != bottomLevel { bottomLevel = bottom }
        }
    }
}

/// fades what scrolls to nothing at each edge, as far as there is more content past that edge.
/// above the top band, under the toolbar, content is hidden entirely once it has scrolled there.
private struct ScrollEdgeMask: View {
    let state: ScrollEdgeState
    let top: Bool
    let bottom: Bool

    /// an eased ramp between two opacities, so the fade has no visible start
    private func ramp(from: Double, to: Double) -> [Gradient.Stop] {
        stride(from: 0.0, through: 1.0, by: 0.125).map { t in
            .init(color: .black.opacity(from + (to - from) * (t * t * (3 - 2 * t))), location: t)
        }
    }

    var body: some View {
        let topClear = top ? 1 - state.topLevel : 1
        let bottomClear = bottom ? 1 - state.bottomLevel : 1
        GeometryReader { proxy in
            VStack(spacing: 0) {
                // whatever bar covers the top: nothing shows through it once content has scrolled under
                Color.black.opacity(topClear).frame(height: proxy.safeAreaInsets.top)
                LinearGradient(stops: ramp(from: topClear, to: 1), startPoint: .top, endPoint: .bottom)
                    .frame(height: FourierGlass.edgeFade)
                Color.black
                LinearGradient(stops: ramp(from: 1, to: bottomClear), startPoint: .top, endPoint: .bottom)
                    .frame(height: FourierGlass.edgeFade)
                Color.black.opacity(bottomClear).frame(height: proxy.safeAreaInsets.bottom)
            }
            .ignoresSafeArea()
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
    private let action: (() -> Void)?

    /// - parameter action: when given, a click or a tap anywhere on the bar runs it, so the whole
    ///   capsule is the target and not only the button inside it
    public init(
        action: (() -> Void)? = nil,
        @ViewBuilder leading: () -> Leading,
        @ViewBuilder center: () -> Center,
        @ViewBuilder trailing: () -> Trailing
    ) {
        self.action = action
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
        .contentShape(.capsule)
        .onTapGesture { action?() }
    }
}
