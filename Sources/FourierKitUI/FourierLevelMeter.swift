import SwiftUI

/// a soft row of rounded bars drawn from the real input level: the newest level enters on the right
/// and older ones move left, so it reads as a waveform of what was heard.
///
/// it redraws at most 30 times a second and only while `isActive`. with reduce motion on it shows one
/// steady bar that follows the level instead of moving.
public struct FourierLevelMeter: View {
    private let level: Float
    private let isActive: Bool
    private let barCount: Int
    private let barWidth: CGFloat
    private let spacing: CGFloat
    private let ink: FourierInk

    @State private var history = LevelHistory()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.colorSchemeContrast) private var contrast

    /// - parameters:
    ///   - level: input level, 0 (silence) to 1 (full scale)
    ///   - isActive: false freezes the meter and stops redrawing
    ///   - barCount: how many past levels are shown
    ///   - barWidth: width of each bar; bars are as tall as the view allows
    ///   - ink: the bars' opacity step
    public init(level: Float, isActive: Bool = true, barCount: Int = 24, barWidth: CGFloat = 4,
                spacing: CGFloat = 4, ink: FourierInk = .secondary) {
        self.level = level
        self.isActive = isActive
        self.barCount = barCount
        self.barWidth = barWidth
        self.spacing = spacing
        self.ink = ink
    }

    public var body: some View {
        Group {
            if reduceMotion {
                steady
            } else {
                moving
            }
        }
        .frame(width: CGFloat(barCount) * (barWidth + spacing) - spacing)
        .accessibilityElement()
        .accessibilityLabel("input level")
        .accessibilityValue(FourierFormat.percent(Double(level)))
    }

    private var color: Color {
        .primary.opacity(ink.opacity(increasedContrast: contrast == .increased))
    }

    private var moving: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !isActive)) { timeline in
            let levels = history.sample(level, at: timeline.date, count: barCount, active: isActive)
            // fixed-size bars scaled by a transform: each frame changes only layer transforms, so nothing is
            // redrawn on the cpu and the main thread never waits on a busy gpu to upload a backing store
            HStack(alignment: .center, spacing: spacing) {
                ForEach(levels.indices, id: \.self) { index in
                    Capsule()
                        .fill(color)
                        .frame(width: barWidth)
                        .scaleEffect(x: 1, y: Self.scale(levels[index], minimum: 0.12), anchor: .center)
                }
            }
            .frame(maxHeight: .infinity)
        }
    }

    private var steady: some View {
        GeometryReader { proxy in
            Capsule()
                .fill(color)
                .frame(width: max(barWidth, proxy.size.width * CGFloat(Self.shape(level))), height: barWidth)
                .frame(maxHeight: .infinity)
        }
    }

    static func scale(_ level: Float, minimum: CGFloat) -> CGFloat {
        max(minimum, CGFloat(shape(level)))
    }

    /// spreads the level so normal speech sits mid-height and only loud input reaches the top
    static func shape(_ level: Float) -> Float {
        pow(max(0, min(1, level)), 1.6)
    }
}

/// the past levels, kept outside view state so sampling never invalidates the view
@MainActor
private final class LevelHistory {
    private var values: [Float] = []
    private var lastSample: Date = .distantPast

    func sample(_ level: Float, at date: Date, count: Int, active: Bool) -> [Float] {
        if values.count != count { values = Array(repeating: 0, count: count) }
        if active, date.timeIntervalSince(lastSample) >= 1.0 / 32 {
            lastSample = date
            values.removeFirst()
            values.append(level)
        }
        return values
    }
}
