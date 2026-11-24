import FourierKit
import Foundation

/// short lowercase strings for numbers the views show. public so apps can match the kit.
public enum FourierFormat {
    /// "1.2 gb", "640 mb"
    public static func bytes(_ bytes: Int64) -> String {
        let value = Double(bytes)
        if value >= 1e9 { return String(format: "%.1f gb", value / 1e9) }
        if value >= 1e6 { return String(format: "%.0f mb", value / 1e6) }
        if value >= 1e3 { return String(format: "%.0f kb", value / 1e3) }
        return "\(bytes) b"
    }

    public static func bytes(_ bytes: UInt64) -> String {
        self.bytes(Int64(clamping: bytes))
    }

    /// "820 ms", "1.4 s"
    public static func latency(_ duration: Duration?) -> String {
        guard let duration else { return "–" }
        let seconds = duration.timeInterval
        if seconds < 1 { return String(format: "%.0f ms", seconds * 1000) }
        return String(format: "%.1f s", seconds)
    }

    /// "0:07", "12:30", "1:02:03"
    public static func clock(_ duration: Duration) -> String {
        let total = max(0, Int(duration.timeInterval.rounded(.down)))
        let hours = total / 3600, minutes = (total / 60) % 60, seconds = total % 60
        if hours > 0 { return String(format: "%d:%02d:%02d", hours, minutes, seconds) }
        return String(format: "%d:%02d", minutes, seconds)
    }

    /// "about 3 min left", "about 40 s left"
    public static func remaining(_ duration: Duration?) -> String? {
        guard let duration else { return nil }
        let seconds = duration.timeInterval
        if seconds < 1 { return "almost done" }
        if seconds < 60 { return "about \(Int(seconds.rounded(.up))) s left" }
        return "about \(Int((seconds / 60).rounded(.up))) min left"
    }

    /// "12.3×"
    public static func speed(_ speed: Double) -> String {
        speed <= 0 ? "–" : String(format: speed < 10 ? "%.1f×" : "%.0f×", speed)
    }

    /// "42 /s"
    public static func rate(_ value: Double) -> String {
        value <= 0 ? "–" : String(format: value < 10 ? "%.1f" : "%.0f", value)
    }

    public static func percent(_ value: Double) -> String {
        String(format: "%.0f%%", max(0, min(1, value)) * 100)
    }

    public static func workload(_ workload: Workload) -> String {
        switch workload {
        case .idle: "idle"
        case .short: "short clip"
        case .medium: "medium clip"
        case .long: "long clip"
        case .veryLong: "very long audio"
        case .manyFiles: "many files"
        case .realtime: "live"
        case .realtimePlusFiles: "live and files"
        case .underLoad: "under load"
        }
    }

    public static func engine(_ engine: EngineState) -> String {
        switch engine {
        case .cold: "not loaded"
        case .preparing: "preparing"
        case .loading: "loading"
        case .warm: "ready"
        case .busy: "working"
        case .cooling: "cooling down"
        case .failed: "failed"
        }
    }

    public static func pressure(_ pressure: SystemPressure) -> String {
        switch pressure {
        case .normal: "normal"
        case .busy: "memory tight"
        case .hot: "hot"
        case .lowPower: "low power"
        }
    }

    public static func stage(_ stage: Stage) -> String {
        switch stage {
        case .mel: "mel"
        case .encoder: "encoder"
        case .decoder: "decoder"
        case .search: "search"
        }
    }

    public static func chip(_ chip: Chip) -> String {
        switch chip {
        case .cpu: "cpu"
        case .gpu: "gpu"
        case .npu: "neural engine"
        }
    }

    /// an error's description, lowercase for display
    public static func error(_ error: any Error) -> String {
        if let error = error as? FourierError { return error.description.lowercased() }
        return error.localizedDescription.lowercased()
    }
}

extension Duration {
    var timeInterval: Double {
        let parts = components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
