import Foundation

/// deterministic splitting of long 16 khz audio at its quietest points.
enum AudioSplitter {
    static let sampleRate = 16_000.0
    /// energy is measured on 10 ms frames
    static let frameLength = 160
    /// a cut sits at the centre of the quietest 200 ms region
    static let regionFrames = 20

    /// audio no longer than `maxSeconds` comes back as one range, untouched. longer audio is cut
    /// greedily: each piece is at most `maxSeconds`, cut at the centre of the quietest 200 ms region
    /// in the last third of the allowed window. a pure function of the samples.
    static func pieces(samples: [Float], maxSeconds: Double = 120) -> [Range<Int>] {
        let total = samples.count
        let maxSamples = max(1, Int(maxSeconds * sampleRate))
        guard total > maxSamples else { return [0..<total] }

        let energy = FrameEnergy(samples)
        var ranges = [Range<Int>]()
        var start = 0
        while total - start > maxSamples {
            let end = start + maxSamples
            let searchStart = start + (maxSamples * 2) / 3
            var cut = energy.quietestCentre(in: searchStart..<end) ?? end
            if cut <= start { cut = end }
            ranges.append(start..<cut)
            start = cut
        }
        ranges.append(start..<total)
        return ranges
    }

    /// halves one piece at the quietest 200 ms region in its middle third, e.g. when it hit the token limit.
    /// nil when the range is too short to split.
    static func split(samples: [Float], range: Range<Int>) -> (first: Range<Int>, second: Range<Int>)? {
        let range = range.clamped(to: 0..<samples.count)
        guard range.count >= 2 else { return nil }
        let third = range.count / 3
        let window = (range.lowerBound + third)..<(range.upperBound - third)
        let energy = FrameEnergy(samples, within: range)
        var cut = energy.quietestCentre(in: window) ?? (range.lowerBound + range.count / 2)
        if cut <= range.lowerBound || cut >= range.upperBound { cut = range.lowerBound + range.count / 2 }
        return (range.lowerBound..<cut, cut..<range.upperBound)
    }

    /// prefix sums of per-frame energy on the absolute 10 ms grid.
    private struct FrameEnergy {
        /// frame index of `prefix[0]`
        let firstFrame: Int
        /// `prefix[i]` is the energy of frames `firstFrame ..< firstFrame + i`
        let prefix: [Double]

        init(_ samples: [Float], within range: Range<Int>? = nil) {
            let range = range ?? 0..<samples.count
            let n = AudioSplitter.frameLength
            let first = range.lowerBound / n
            let last = range.upperBound / n  // whole frames only
            var prefix = [Double](repeating: 0, count: max(0, last - first) + 1)
            samples.withUnsafeBufferPointer { s in
                var running = 0.0
                for f in first..<max(first, last) {
                    var sum = 0.0
                    for i in (f * n)..<(f * n + n) {
                        let x = Double(s[i])
                        sum += x * x
                    }
                    running += sum
                    prefix[f - first + 1] = running
                }
            }
            firstFrame = first
            self.prefix = prefix
        }

        /// centre sample of the lowest-energy whole 200 ms region inside `window`; the latest wins a tie.
        func quietestCentre(in window: Range<Int>) -> Int? {
            let n = AudioSplitter.frameLength, r = AudioSplitter.regionFrames
            let first = max(firstFrame, (window.lowerBound + n - 1) / n)
            let last = min(firstFrame + prefix.count - 1, window.upperBound / n) - r
            guard last >= first else { return nil }
            var best = first
            var bestEnergy = Double.infinity
            for f in first...last {
                let e = prefix[f - firstFrame + r] - prefix[f - firstFrame]
                if e <= bestEnergy {
                    bestEnergy = e
                    best = f
                }
            }
            return best * n + r * n / 2
        }
    }
}
