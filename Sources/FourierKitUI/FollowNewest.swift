import SwiftUI

extension View {
    /// keeps the end of a growing scroll view in view, for text that streams in.
    ///
    /// apply it to a `ScrollView`. while the reader is at the end, new content scrolls into view as it
    /// arrives. scrolling up to read lets go, and scrolling back to the end picks it up again.
    public func fourierFollowsNewest() -> some View {
        modifier(FollowNewest())
    }
}

private struct FollowNewest: ViewModifier {
    private struct Metrics: Equatable {
        var content: CGFloat
        var container: CGFloat
        /// how far the end of the content is below the bottom of what is visible
        var below: CGFloat
    }

    @State private var position = ScrollPosition(edge: .top)
    // not view state that draws: only the scroll callback reads it
    @State private var following = Following()

    func body(content: Content) -> some View {
        content
            .scrollPosition($position)
            .onScrollGeometryChange(for: Metrics.self) { geometry in
                let visible = geometry.containerSize.height - geometry.contentInsets.top - geometry.contentInsets.bottom
                return Metrics(
                    content: geometry.contentSize.height.rounded(),
                    container: visible.rounded(),
                    below: (geometry.contentSize.height - visible - geometry.contentOffset.y - geometry.contentInsets.top).rounded())
            } action: { old, new in
                if new.content != old.content || new.container != old.container {
                    // the text or the window changed size, not the reader's place in it
                    if new.content <= new.container { following.isOn = true }
                    if following.isOn, new.below > 0 { position.scrollTo(edge: .bottom) }
                } else {
                    // the reader moved: follow only while they stay within a line or two of the end
                    following.isOn = new.below < 48
                }
            }
    }
}

private final class Following {
    var isOn = true
}
