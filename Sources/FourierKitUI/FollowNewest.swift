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
        /// the offset that puts the end of the content at the bottom of what is visible
        var end: CGFloat
    }

    @State private var position = ScrollPosition(edge: .top)
    // not view state that draws: only the scroll callback reads it
    @State private var following = Following()

    func body(content: Content) -> some View {
        content
            .scrollPosition($position)
            .onScrollGeometryChange(for: Metrics.self) { geometry in
                let visible = geometry.containerSize.height - geometry.contentInsets.top - geometry.contentInsets.bottom
                let end = geometry.contentSize.height - geometry.containerSize.height + geometry.contentInsets.bottom
                return Metrics(
                    content: geometry.contentSize.height.rounded(),
                    container: visible.rounded(),
                    below: (end - geometry.contentOffset.y).rounded(),
                    end: end)
            } action: { old, new in
                if new.content != old.content || new.container != old.container {
                    // the text or the window changed size, not the reader's place in it
                    if new.content <= new.container { following.isOn = true }
                    if following.isOn, new.below > 0 {
                        // an offset, not the bottom edge: asking for the same edge twice does nothing.
                        // no animation, so a frame in between is never taken for the reader's own scroll
                        var transaction = Transaction()
                        transaction.disablesAnimations = true
                        withTransaction(transaction) { position.scrollTo(y: new.end) }
                    }
                } else if new.below > old.below {
                    // the reader moved up: let go once they are more than a line or two from the end
                    if new.below >= 48 { following.isOn = false }
                } else if new.below < 48 {
                    // back at the end: follow again
                    following.isOn = true
                }
            }
    }
}

private final class Following {
    var isOn = true
}
