#if DEBUG
import Foundation
import FourierKitUI
import os

/// debug only: measures how long the main thread stays busy between sleeps while a mode is transcribing,
/// recording or listening, and logs every stall over 8 ms plus the worst one when the work ends.
///
/// it watches the main run loop (wake to next sleep), so it costs two timestamps per loop and never blocks.
/// lines go to stderr and the unified log as `watchdog: …`.
final class MainThreadWatchdog: @unchecked Sendable {
    static let shared = MainThreadWatchdog()

    private let threshold = 0.008
    private let logger = Logger(subsystem: "com.karanganesan.fourierkit", category: "watchdog")
    // touched only on the main thread
    private var observer: CFRunLoopObserver?
    private var wokeAt: Double?
    private var measuring = false
    private var label = ""
    private var worst = 0.0
    private var stalls = 0
    private var loops = 0
    private var allTimeWorst = 0.0

    func start() {
        guard observer == nil else { return }
        let observer = CFRunLoopObserverCreateWithHandler(
            kCFAllocatorDefault, CFRunLoopActivity.afterWaiting.rawValue | CFRunLoopActivity.beforeWaiting.rawValue,
            true, 0
        ) { [unowned self] _, activity in
            let now = ProcessInfo.processInfo.systemUptime
            if activity == .afterWaiting {
                wokeAt = now
            } else if let woke = wokeAt {
                wokeAt = nil
                guard measuring else { return }
                loops += 1
                let busy = now - woke
                if busy > worst { worst = busy }
                if busy > threshold {
                    stalls += 1
                    report(String(format: "watchdog: main thread busy %.1f ms while %@", busy * 1000, label))
                }
            }
        }
        CFRunLoopAddObserver(CFRunLoopGetMain(), observer, .commonModes)
        self.observer = observer
        report("watchdog: on, threshold 8 ms")
        report("font: allo variable \(FourierFont.isAlloAvailable ? "loaded" : "not loaded: \(FourierFont.alloLoadError ?? "")")")
    }

    /// call on the main thread when work starts and ends
    func setMeasuring(_ on: Bool, label: String) {
        guard on != measuring else { return }
        if on {
            self.label = label
            worst = 0
            stalls = 0
            loops = 0
            measuring = true
        } else {
            measuring = false
            allTimeWorst = max(allTimeWorst, worst)
            report(String(format: "watchdog: %@ done, worst main thread busy %.1f ms, %d of %d loops over 8 ms, worst this run %.1f ms",
                          self.label, worst * 1000, stalls, loops, allTimeWorst * 1000))
        }
    }

    private func report(_ line: String) {
        logger.log("\(line, privacy: .public)")
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
}
#endif
