import Accelerate
import Foundation

/// log-mel features for one clip, laid out like torch's `[128, T]`.
struct MelFeatures: Sendable {
    var bins: Int
    /// valid frames, the sum of the reference's `input_features_mask`
    var frames: Int
    /// `frames` rounded up to a multiple of 100, the extra columns are zero
    var paddedFrames: Int
    /// row-major `[bin][paddedFrame]`
    var data: [Float]
}

/// the qwen3-asr feature extractor, as the processor calls it for one clip.
enum LogMel {
    static let sampleRate = 16_000
    static let fftLength = 400
    static let hopLength = 160
    static let bins = 128
    static let spectrumBins = fftLength / 2 + 1
    /// shorter clips are zero-padded to this many samples, and the mask counts the padding
    static let minimumSamples = 8_000
    /// mel frames per encoder chunk (`2 * n_window`)
    static let chunkFrames = 100

    // MARK: clip

    static func features(samples: [Float]) -> MelFeatures {
        let length = max(samples.count, minimumSamples)
        // a centred stft has length/hop + 1 frames and the reference drops the last one
        let frames = length / hopLength
        let paddedFrames = (frames + chunkFrames - 1) / chunkFrames * chunkFrames
        var data = [Float](repeating: 0, count: bins * paddedFrames)

        let signal = centerPadded(samples, length: length)
        signal.withUnsafeBufferPointer { signal in
            data.withUnsafeMutableBufferPointer { out in
                let source = SendablePointer(signal.baseAddress!)
                let target = SendableMutablePointer(out.baseAddress!)
                let cores = DeviceProfile.cpu.activeProcessorCount
                let chunk = max(64, (frames + cores * 4 - 1) / (cores * 4))
                let chunks = (frames + chunk - 1) / chunk
                DispatchQueue.concurrentPerform(iterations: chunks) { index in
                    let start = index * chunk
                    let range = start..<min(frames, start + chunk)
                    logMelFrames(
                        signal: source.pointer, frames: range,
                        output: target.pointer + start, rowStride: paddedFrames)
                }
            }
        }

        normalize(&data, frames: frames, rowStride: paddedFrames)
        return MelFeatures(bins: bins, frames: frames, paddedFrames: paddedFrames, data: data)
    }

    /// `_get_audio_token_length`: 13 tokens per full 100-frame chunk, plus three stride-2 convs over the rest.
    static func audioTokenCount(frames: Int) -> Int {
        let remainder = frames % chunkFrames
        // python floor division: a zero remainder adds no tokens
        let afterFirst = floorDiv(remainder - 1, 2) + 1
        let afterSecond = floorDiv(afterFirst - 1, 2) + 1
        return floorDiv(afterSecond - 1, 2) + 1 + (frames / chunkFrames) * 13
    }

    // MARK: stages

    /// per-frame stage: windowed dft, power, mel, clamp and log10.
    /// frame `t` reads `signal[t*hop ..< t*hop + 400]` of the centre-padded signal, and bin `b`
    /// of frame `frames.lowerBound + i` goes to `output[b * rowStride + i]`.
    static func logMelFrames(
        signal: UnsafePointer<Float>, frames: Range<Int>,
        output: UnsafeMutablePointer<Float>, rowStride: Int
    ) {
        let count = frames.count
        guard count > 0 else { return }
        let tables = Tables.shared
        let n = fftLength, k = spectrumBins

        let windowed = UnsafeMutablePointer<Float>.allocate(capacity: count * n)
        let spectrum = UnsafeMutablePointer<Float>.allocate(capacity: count * 2 * k)
        let power = UnsafeMutablePointer<Float>.allocate(capacity: count * k)
        defer {
            windowed.deallocate()
            spectrum.deallocate()
            power.deallocate()
        }

        tables.window.withUnsafeBufferPointer { window in
            for i in 0..<count {
                let start = signal + (frames.lowerBound + i) * hopLength
                vDSP_vmul(start, 1, window.baseAddress!, 1, windowed + i * n, 1, vDSP_Length(n))
            }
        }

        // [count, 400] x [400, 402]: each row is 201 real parts then 201 imaginary parts
        tables.dft.withUnsafeBufferPointer { dft in
            vDSP_mmul(windowed, 1, dft.baseAddress!, 1, spectrum, 1, vDSP_Length(count), vDSP_Length(2 * k), vDSP_Length(n))
        }

        for i in 0..<count {
            let re = spectrum + i * 2 * k
            let im = re + k
            vDSP_vmma(re, 1, re, 1, im, 1, im, 1, power + i * k, 1, vDSP_Length(k))
        }

        // [count, 201] x [201, 128], then transposed into the strided output rows
        let mel = spectrum  // the spectrum is no longer needed, and count * 402 >= count * 128
        tables.melByBin.withUnsafeBufferPointer { filters in
            vDSP_mmul(power, 1, filters.baseAddress!, 1, mel, 1, vDSP_Length(count), vDSP_Length(bins), vDSP_Length(k))
        }
        for b in 0..<bins {
            vDSP_mmov(mel + b, output + b * rowStride, 1, vDSP_Length(count), vDSP_Length(bins), vDSP_Length(1))
        }

        var floor: Float = 1e-10
        var length = Int32(count)
        for b in 0..<bins {
            let row = output + b * rowStride
            vDSP_vthr(row, 1, &floor, row, 1, vDSP_Length(count))
            vvlog10f(row, row, &length)
        }
    }

    /// clip-level stage: floor at the clip's loudest value minus 8, then `(x + 4) / 4`.
    /// only the first `frames` columns of each row are touched.
    static func normalize(_ data: inout [Float], frames: Int, rowStride: Int) {
        guard frames > 0 else { return }
        data.withUnsafeMutableBufferPointer { buffer in
            let base = buffer.baseAddress!
            let rows = buffer.count / rowStride
            var peak = -Float.infinity
            for b in 0..<rows {
                var rowPeak: Float = 0
                vDSP_maxv(base + b * rowStride, 1, &rowPeak, vDSP_Length(frames))
                peak = max(peak, rowPeak)
            }
            normalize(base, rows: rows, frames: frames, rowStride: rowStride, peak: peak)
        }
    }

    /// the same floor and scale with a known peak, for callers that track the loudest value themselves.
    static func normalize(
        _ base: UnsafeMutablePointer<Float>, rows: Int, frames: Int, rowStride: Int, peak: Float
    ) {
        var floor = peak - 8
        var four: Float = 4
        var quarter: Float = 0.25
        for b in 0..<rows {
            let row = base + b * rowStride
            vDSP_vthr(row, 1, &floor, row, 1, vDSP_Length(frames))
            vDSP_vsadd(row, 1, &four, row, 1, vDSP_Length(frames))
            // dividing by four is exact, so this multiply matches the reference bit for bit
            vDSP_vsmul(row, 1, &quarter, row, 1, vDSP_Length(frames))
        }
    }

    /// zero-pads to `length`, then reflect-pads 200 samples on each side as `torch.stft(center=True)` does.
    static func centerPadded(_ samples: [Float], length: Int) -> [Float] {
        let half = fftLength / 2
        var x = samples
        if x.count < length { x.append(contentsOf: repeatElement(0, count: length - x.count)) }
        var out = [Float](repeating: 0, count: length + 2 * half)
        for i in 0..<half { out[i] = x[half - i] }
        out.replaceSubrange(half..<(half + length), with: x)
        for j in 0..<half { out[half + length + j] = x[length - 2 - j] }
        return out
    }

    // MARK: tables

    final class Tables: Sendable {
        static let shared = Tables()

        /// periodic hann window of 400
        let window: [Float]
        /// `[400, 402]`: cos for the 201 bins, then -sin
        let dft: [Float]
        /// `[201, 128]` slaney mel filters, the layout `mel_filter_bank` returns
        let melByBin: [Float]

        private init() {
            let n = LogMel.fftLength, k = LogMel.spectrumBins
            window = (0..<n).map { Float(0.5 - 0.5 * __cospi(2 * Double($0) / Double(n))) }

            var dft = [Float](repeating: 0, count: n * 2 * k)
            for t in 0..<n {
                for f in 0..<k {
                    // exact argument reduction keeps the zeros and symmetries exact
                    let r = Double((t * f) % n) * 2 / Double(n)
                    dft[t * 2 * k + f] = Float(__cospi(r))
                    dft[t * 2 * k + k + f] = Float(-__sinpi(r))
                }
            }
            self.dft = dft
            melByBin = LogMel.slaneyFilters(
                spectrumBins: k, filters: LogMel.bins, minFrequency: 0, maxFrequency: 8000,
                sampleRate: Double(LogMel.sampleRate))
        }
    }

    /// `mel_filter_bank(norm="slaney", mel_scale="slaney")` in double, returned as float `[bin][filter]`.
    static func slaneyFilters(
        spectrumBins: Int, filters: Int, minFrequency: Double, maxFrequency: Double, sampleRate: Double
    ) -> [Float] {
        func toMel(_ hz: Double) -> Double {
            hz >= 1000 ? 15 + Foundation.log(hz / 1000) * (27 / Foundation.log(6.4)) : 3 * hz / 200
        }
        func toHertz(_ mel: Double) -> Double {
            mel >= 15 ? 1000 * exp(Foundation.log(6.4) / 27 * (mel - 15)) : 200 * mel / 3
        }
        let low = toMel(minFrequency), high = toMel(maxFrequency)
        // numpy linspace: start + i * step, with the end pinned
        let step = (high - low) / Double(filters + 1)
        let edges = (0..<(filters + 2)).map { i -> Double in
            toHertz(i == filters + 1 ? high : low + Double(i) * step)
        }
        let nyquist = Double(Int(sampleRate) / 2)
        let binStep = nyquist / Double(spectrumBins - 1)
        let frequencies = (0..<spectrumBins).map { i -> Double in
            i == spectrumBins - 1 ? nyquist : Double(i) * binStep
        }

        var out = [Float](repeating: 0, count: filters * spectrumBins)
        for m in 0..<filters {
            let lower = edges[m + 1] - edges[m]
            let upper = edges[m + 2] - edges[m + 1]
            let scale = 2 / (edges[m + 2] - edges[m])
            for i in 0..<spectrumBins {
                let down = -(edges[m] - frequencies[i]) / lower
                let up = (edges[m + 2] - frequencies[i]) / upper
                out[i * filters + m] = Float(max(0, min(down, up)) * scale)
            }
        }
        return out
    }

    private static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
    }
}

/// raw pointers handed to `concurrentPerform`; each iteration touches its own columns.
struct SendablePointer<T>: @unchecked Sendable {
    let pointer: UnsafePointer<T>
    init(_ pointer: UnsafePointer<T>) { self.pointer = pointer }
}

struct SendableMutablePointer<T>: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<T>
    init(_ pointer: UnsafeMutablePointer<T>) { self.pointer = pointer }
}
