import Foundation
import Metal
#if os(macOS)
import IOKit.ps
#endif

/// what the device is like right now: chip, memory, heat and power. read off the main actor.
public struct FourierMachine: Sendable, Hashable {
    public enum Power: String, Sendable, Hashable {
        case adapter, battery, unknown
    }

    /// e.g. "apple m4", or "apple a18 pro" on a phone
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

    /// reads the machine now. cheap, but it calls into iokit on a mac, so call it off the main actor.
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
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return gpuChipName }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return gpuChipName }
        let name = bytes.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
        return name.lowercased()
    }()

    /// ios has no cpu brand string; the gpu is named after the chip, e.g. "apple a18 pro gpu"
    private static var gpuChipName: String {
        guard let name = MTLCreateSystemDefaultDevice()?.name.lowercased() else { return "apple silicon" }
        return name.hasSuffix(" gpu") ? String(name.dropLast(4)) : name
    }

    #if os(macOS)
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
    #else
    /// ios reports charging only through uidevice on the main actor, with monitoring switched on
    /// app-wide, which a read off the main actor must not do.
    private static func powerSource() -> Power { .unknown }
    #endif

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
