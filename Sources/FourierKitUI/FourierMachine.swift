import Foundation
import IOKit.ps

/// what the mac is like right now: chip, memory, heat and power. read off the main actor.
public struct FourierMachine: Sendable, Hashable {
    public enum Power: String, Sendable, Hashable {
        case adapter, battery, unknown
    }

    /// e.g. "apple m4"
    public var chip: String
    public var memory: UInt64
    public var thermal: ProcessInfo.ThermalState
    public var power: Power
    public var lowPowerMode: Bool

    public init(chip: String = "", memory: UInt64 = 0, thermal: ProcessInfo.ThermalState = .nominal,
                power: Power = .unknown, lowPowerMode: Bool = false) {
        self.chip = chip
        self.memory = memory
        self.thermal = thermal
        self.power = power
        self.lowPowerMode = lowPowerMode
    }

    /// reads the machine now. cheap, but it calls into iokit, so call it off the main actor.
    public static func read() -> FourierMachine {
        FourierMachine(
            chip: chipName,
            memory: ProcessInfo.processInfo.physicalMemory,
            thermal: ProcessInfo.processInfo.thermalState,
            power: powerSource(),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
    }

    private static let chipName: String = {
        var size = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return "apple silicon" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return "apple silicon" }
        let name = bytes.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
        return name.lowercased()
    }()

    private static func powerSource() -> Power {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() as String?
        else { return .unknown }
        switch type {
        case kIOPMACPowerKey: return .adapter
        case kIOPMBatteryPowerKey: return .battery
        default: return .unknown
        }
    }

    public var thermalText: String {
        switch thermal {
        case .nominal: "cool"
        case .fair: "warm"
        case .serious: "hot"
        case .critical: "very hot"
        @unknown default: "unknown"
        }
    }

    public var powerText: String {
        let source = switch power {
        case .adapter: "power adapter"
        case .battery: "battery"
        case .unknown: "unknown"
        }
        return lowPowerMode ? "\(source), low power" : source
    }
}
