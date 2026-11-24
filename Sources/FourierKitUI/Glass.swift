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
    /// height of the progressive blur at the top and bottom of anything that scrolls
    public static let edgeBlur: CGFloat = 40
    /// its strength, as a share of the system's lightest blur at full
    public static let edgeBlurStrength: Double = 0.8
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

extension View {
    /// soft edges on anything that scrolls up and down: what passes under the top or the bottom
    /// blurs progressively, strongest at the very edge and gone a band's height in. an edge only
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
            .overlay(alignment: .top) { if top { ScrollEdgeBand(state: state, edge: .top) } }
            .overlay(alignment: .bottom) { if bottom { ScrollEdgeBand(state: state, edge: .bottom) } }
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
        let height = FourierGlass.edgeBlur
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

private struct ScrollEdgeBand: View {
    let state: ScrollEdgeState
    let edge: VerticalEdge

    var body: some View {
        let level = edge == .top ? state.topLevel : state.bottomLevel
        GeometryReader { proxy in
            // the band is its own height plus whatever bar covers this edge, so it starts at the
            // window's edge and still reaches a full band's height into the visible content
            let inset = edge == .top ? proxy.safeAreaInsets.top : proxy.safeAreaInsets.bottom
            EdgeBand(edge: edge)
                .frame(height: FourierGlass.edgeBlur + inset)
                .frame(maxHeight: .infinity, alignment: edge == .top ? .top : .bottom)
                .opacity(level)
                .ignoresSafeArea(edges: edge == .top ? .top : .bottom)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// one edge's band: blurs that build toward the edge. every layer fades in on an eased ramp from
/// where it starts to the edge, each stronger than the last and starting nearer the edge, so the
/// blur deepens as the edge nears and there is no line where the band begins.
private struct EdgeBand: View {
    let edge: VerticalEdge
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    /// the system's blurs, lightest first, and where each begins: 0 is the band's inner side, 1 the edge
    private let layers: [(Material, Double)] = [(.ultraThinMaterial, 0.0), (.thinMaterial, 0.35), (.regularMaterial, 0.65)]

    private func ramp(from start: Double) -> [Gradient.Stop] {
        stride(from: 0.0, through: 1.0, by: 0.125).map { t in
            .init(color: .black.opacity(t * t * (3 - 2 * t)), location: start + (1 - start) * t)
        }
    }

    var body: some View {
        let toward: (UnitPoint, UnitPoint) = edge == .top ? (.bottom, .top) : (.top, .bottom)
        if !reduceTransparency {
            ZStack {
                ForEach(layers.indices, id: \.self) { index in
                    Rectangle()
                        .fill(layers[index].0)
                        .mask(LinearGradient(stops: ramp(from: layers[index].1), startPoint: toward.0, endPoint: toward.1))
                        .opacity(FourierGlass.edgeBlurStrength)
                }
            }
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
