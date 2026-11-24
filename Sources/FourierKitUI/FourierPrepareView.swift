import FourierKit
import SwiftUI

/// one smooth bar from download to warm-up, with the stage, bytes and time left.
///
/// shows the model's progress whether or not this view started the prepare. pass `startsPrepare: true`
/// (the default) to begin preparing when it appears.
public struct FourierPrepareView<Accessory: View>: View {
    private let model: FourierStateModel
    private let title: String
    private let message: String?
    private let startsPrepare: Bool
    private let accessory: Accessory

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// - parameters:
    ///   - title: the line above the bar
    ///   - message: one line below the title, e.g. why the download is needed
    ///   - startsPrepare: begins preparing when the view appears
    ///   - accessory: extra content under the bar, e.g. a cancel button
    public init(
        _ model: FourierStateModel,
        title: String = "getting ready",
        message: String? = "the speech model runs on this mac. it downloads once, then stays here.",
        startsPrepare: Bool = true,
        @ViewBuilder accessory: () -> Accessory
    ) {
        self.model = model
        self.title = title
        self.message = message
        self.startsPrepare = startsPrepare
        self.accessory = accessory()
    }

    public var body: some View {
        let progress = model.prepare
        let fraction = model.isReady ? 1 : progress?.fraction ?? 0
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text(title.lowercased())
                    .fourierTextStyle(.title)
                    .fourierInk(.primary)
                if let message {
                    Text(message.lowercased())
                        .fourierTextStyle(.body)
                        .fourierInk(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            ProgressView(value: fraction)
                .progressViewStyle(.linear)
                .tint(.primary)
                .animation(reduceMotion ? nil : .smooth(duration: 0.4), value: fraction)
                .accessibilityLabel("model progress")
                .accessibilityValue(FourierFormat.percent(fraction))

            if let error = model.prepareError {
                Text(FourierFormat.error(error))
                    .fourierTextStyle(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
                Button("try again") { Task { await model.prepare() } }
                    .buttonStyle(.glass)
                    .fourierTextStyle(.body)
            } else {
                // fixed slots: the stage on the left, bytes and time on the right, one line each
                HStack(alignment: .firstTextBaseline, spacing: 16) {
                    Text(stageText(progress))
                        .contentTransition(.opacity)
                    Spacer(minLength: 8)
                    Text(detailText(progress))
                        .multilineTextAlignment(.trailing)
                }
                .fourierTextStyle(.caption, numbers: .tabular)
                .fourierInk(.secondary)
                .lineLimit(1)
                .animation(reduceMotion ? nil : .smooth, value: progress?.stage)
            }
            accessory
        }
        .frame(maxWidth: 400)
        .fourierGlassPanel()
        .task(id: startsPrepare) {
            guard startsPrepare, !model.isReady else { return }
            await model.prepare()
        }
        .accessibilityElement(children: .contain)
    }

    private func stageText(_ progress: PrepareProgress?) -> String {
        if model.isReady { return "ready" }
        return progress?.stage.name ?? "checking files"
    }

    private func detailText(_ progress: PrepareProgress?) -> String {
        guard let progress, !model.isReady else { return " " }
        var parts: [String] = []
        if progress.stage == .download, progress.bytesTotal > 0 {
            parts.append("\(FourierFormat.bytes(progress.bytesDone)) of \(FourierFormat.bytes(progress.bytesTotal))")
        }
        if let remaining = FourierFormat.remaining(progress.timeRemaining) { parts.append(remaining) }
        return parts.isEmpty ? FourierFormat.percent(progress.fraction) : parts.joined(separator: " · ")
    }
}

extension FourierPrepareView where Accessory == EmptyView {
    public init(
        _ model: FourierStateModel,
        title: String = "getting ready",
        message: String? = "the speech model runs on this mac. it downloads once, then stays here.",
        startsPrepare: Bool = true
    ) {
        self.init(model, title: title, message: message, startsPrepare: startsPrepare) { EmptyView() }
    }
}
