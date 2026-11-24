import Charts
import FourierKit
import SwiftUI

/// what the engine is doing right now: speed with a small live chart, latency, packing, chips, memory and machine.
///
/// numbers refresh about four times a second in fixed slots. metrics are collected only while this view is on screen.
public struct FourierStatsView<Footer: View>: View {
    private let model: FourierStateModel
    private let footer: Footer

    /// - parameter footer: extra rows at the end, e.g. app-specific numbers
    public init(_ model: FourierStateModel, @ViewBuilder footer: () -> Footer) {
        self.model = model
        self.footer = footer()
    }

    public var body: some View {
        let metrics = model.metrics
        let state = model.state
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                speed(metrics)

                StatsGroup("latency") {
                    StatsRow("first text", FourierFormat.latency(metrics.timeToFirstText))
                    StatsRow("draft delay", FourierFormat.latency(metrics.draftDelay))
                    StatsRow("settle delay", FourierFormat.latency(metrics.settleDelay))
                }

                StatsGroup("packing") {
                    StatsRow("workload", FourierFormat.workload(metrics.workload))
                    StatsRow("batch", metrics.maxBatchWidth > 0 ? "\(metrics.batchWidth) of \(metrics.maxBatchWidth)" : "\(metrics.batchWidth)")
                    StatsRow("running pieces", "\(metrics.activePieces)")
                    StatsRow("queued pieces", "\(metrics.queuedPieces)")
                    StatsBar("slot use", value: metrics.maxBatchWidth > 0 ? Double(metrics.batchWidth) / Double(metrics.maxBatchWidth) : 0)
                }

                StatsGroup("chips") {
                    ForEach(Stage.allCases, id: \.self) { stage in
                        StatsRow(FourierFormat.stage(stage), state.placement[stage].map(FourierFormat.chip) ?? "–")
                    }
                }

                StatsGroup("busy") {
                    ForEach([Chip.npu, .gpu, .cpu], id: \.self) { chip in
                        StatsBar(FourierFormat.chip(chip), value: metrics.chipBusy[chip] ?? 0)
                    }
                }

                StatsGroup("memory") {
                    StatsRow("app", FourierFormat.bytes(metrics.appMemory))
                    StatsRow("weights", FourierFormat.bytes(metrics.weightMemory))
                    StatsRow("decoder cache", FourierFormat.bytes(metrics.cacheMemory))
                }

                StatsGroup("machine") {
                    StatsRow("engine", FourierFormat.engine(state.engine))
                    StatsRow("chip", model.machine.chip.isEmpty ? "–" : model.machine.chip)
                    StatsRow("memory", model.machine.memory > 0 ? FourierFormat.bytes(model.machine.memory) : "–")
                    StatsRow("heat", model.machine.thermalText)
                    StatsRow("power", model.machine.powerText)
                    StatsRow("pressure", FourierFormat.pressure(state.pressure))
                }

                footer
            }
            .padding(16)
        }
        .scrollIndicators(.automatic)
        .fourierScrollEdges()
        .task { await model.observeMetrics() }
    }

    private func speed(_ metrics: FourierMetrics) -> some View {
        StatsGroup("speed") {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(FourierFormat.speed(metrics.realtimeSpeed))
                    .fourierTextStyle(.largeTitle, numbers: .tabular)
                    .fourierInk(.primary)
                Text("realtime")
                    .fourierTextStyle(.caption)
                    .fourierInk(.tertiary)
            }
            .accessibilityElement(children: .combine)
            StatsRow("tokens per second", FourierFormat.rate(metrics.tokensPerSecond))
            SpeedChart(samples: model.speedHistory)
                .frame(height: 48)
                .padding(.vertical, 8)
            StatsRow("decoder steps per second", FourierFormat.rate(metrics.decoderStepsPerSecond))
        }
    }
}

extension FourierStatsView where Footer == EmptyView {
    public init(_ model: FourierStateModel) {
        self.init(model) { EmptyView() }
    }
}

// MARK: parts

/// the last 30 s of tokens per second as a soft area, no axes
private struct SpeedChart: View {
    let samples: [FourierMetricSample]
    @Environment(\.colorSchemeContrast) private var contrast

    var body: some View {
        let increased = contrast == .increased
        let top = max(1, samples.map(\.tokensPerSecond).max() ?? 1)
        Chart(samples) { sample in
            AreaMark(x: .value("time", sample.id), y: .value("tokens", sample.tokensPerSecond))
                .foregroundStyle(Color.primary.opacity(FourierInk.faintFill.opacity(increasedContrast: increased)))
                .interpolationMethod(.monotone)
            LineMark(x: .value("time", sample.id), y: .value("tokens", sample.tokensPerSecond))
                .foregroundStyle(Color.primary.opacity(FourierInk.secondary.opacity(increasedContrast: increased)))
                .lineStyle(StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round))
                .interpolationMethod(.monotone)
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: 0...(top * 1.15))
        .chartXScale(domain: (samples.last?.id ?? 0) - 119 ... max(samples.last?.id ?? 0, 1))
        .accessibilityLabel("tokens per second over the last 30 seconds")
        .accessibilityValue(FourierFormat.rate(samples.last?.tokensPerSecond ?? 0))
    }
}

/// a titled group of rows
struct StatsGroup<Content: View>: View {
    let title: String
    let content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(title)
                .fourierTextStyle(.caption)
                .fourierInk(.tertiary)
                .padding(.bottom, 8)
                .accessibilityAddTraits(.isHeader)
            content
        }
    }
}

/// a label and a value in a fixed, right-aligned slot
struct StatsRow: View {
    let label: String
    let value: String

    init(_ label: String, _ value: String) {
        self.label = label
        self.value = value
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .fourierInk(.secondary)
                .lineLimit(1)
            Spacer(minLength: 8)
            Text(value)
                .fourierInk(.primary)
                .lineLimit(1)
                .multilineTextAlignment(.trailing)
        }
        .fourierTextStyle(.body, numbers: .tabular)
        .frame(minHeight: 24)
        .accessibilityElement(children: .combine)
    }
}

/// a label, a soft bar and a percentage
struct StatsBar: View {
    let label: String
    let value: Double

    init(_ label: String, value: Double) {
        self.label = label
        self.value = value
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .fourierInk(.secondary)
                .lineLimit(1)
                .frame(width: 104, alignment: .leading)
            ProgressView(value: max(0, min(1, value)))
                .progressViewStyle(.linear)
                .controlSize(.small)
                .tint(.primary)
                .accessibilityHidden(true)
            Text(FourierFormat.percent(value))
                .fourierInk(.primary)
                .frame(width: 40, alignment: .trailing)
        }
        .fourierTextStyle(.body, numbers: .tabular)
        .frame(minHeight: 24)
        .accessibilityElement(children: .combine)
    }
}
