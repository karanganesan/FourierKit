import AVFoundation
import Accelerate
import Foundation

enum AudioFileError: Error, Sendable, Equatable {
    case unreadable(String)
    case conversionFailed(String)
}

/// decodes any file core audio can open into 16 khz mono float32.
enum AudioFileReader {
    static let targetSampleRate: Double = 16_000

    /// blocking. a 16 khz mono file comes back with its samples untouched.
    static func read(_ url: URL) throws -> [Float] {
        let file: AVAudioFile
        do {
            // deinterleaved float32 at the file's own rate and channel count
            file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
        } catch {
            throw AudioFileError.unreadable(error.localizedDescription)
        }
        let format = file.processingFormat
        let total = file.length
        guard total > 0 else { return [] }
        guard total < Int64(Int32.max) else { throw AudioFileError.unreadable("file is too long") }

        let channels = Int(format.channelCount)
        var mono = [Float]()
        mono.reserveCapacity(Int(total))

        let block: AVAudioFrameCount = 1 << 18
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: block) else {
            throw AudioFileError.conversionFailed("could not allocate a read buffer")
        }
        while file.framePosition < total {
            do {
                try file.read(into: buffer, frameCount: block)
            } catch {
                throw AudioFileError.unreadable(error.localizedDescription)
            }
            let frames = Int(buffer.frameLength)
            if frames == 0 { break }
            guard let data = buffer.floatChannelData else {
                throw AudioFileError.conversionFailed("no float channel data")
            }
            appendDownmix(data, channels: channels, frames: frames, to: &mono)
        }

        if format.sampleRate == targetSampleRate { return mono }
        return Resampler.resample(mono, from: format.sampleRate)
    }

    /// averages channels; a single channel is copied as is.
    static func appendDownmix(
        _ data: UnsafePointer<UnsafeMutablePointer<Float>>, channels: Int, frames: Int,
        to out: inout [Float]
    ) {
        if channels == 1 {
            out.append(contentsOf: UnsafeBufferPointer(start: data[0], count: frames))
            return
        }
        let start = out.count
        out.append(contentsOf: repeatElement(0, count: frames))
        out.withUnsafeMutableBufferPointer { buffer in
            let dst = buffer.baseAddress! + start
            for c in 0..<channels {
                vDSP_vadd(dst, 1, data[c], 1, dst, 1, vDSP_Length(frames))
            }
            var scale = 1 / Float(channels)
            vDSP_vsmul(dst, 1, &scale, dst, 1, vDSP_Length(frames))
        }
    }
}

/// mono sample-rate conversion to 16 khz with core audio's best converter.
enum Resampler {
    /// blocking. the output has `round(count * 16000 / sampleRate)` samples and is time-aligned with the input.
    static func resample(_ samples: [Float], from sampleRate: Double) -> [Float] {
        let target = AudioFileReader.targetSampleRate
        guard sampleRate != target else { return samples }
        guard !samples.isEmpty, sampleRate > 0 else { return [] }

        let expected = Int((Double(samples.count) * target / sampleRate).rounded())
        guard
            let inFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false),
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: target, channels: 1, interleaved: false),
            let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else { return [] }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        // .normal assumes silence before the start, so the output is not delayed
        converter.primeMethod = .normal

        let feed = ConverterFeed(samples: samples, format: inFormat)
        var out = [Float]()
        out.reserveCapacity(expected + 1024)
        let capacity: AVAudioFrameCount = 1 << 16
        guard let buffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return [] }

        while true {
            buffer.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: buffer, error: &error) { count, outStatus in
                feed.next(count, outStatus)
            }
            if buffer.frameLength > 0, let data = buffer.floatChannelData {
                out.append(contentsOf: UnsafeBufferPointer(start: data[0], count: Int(buffer.frameLength)))
            }
            if status == .endOfStream || status == .error { break }
            if status == .inputRanDry && buffer.frameLength == 0 { break }
        }

        if out.count > expected {
            out.removeLast(out.count - expected)
        } else if out.count < expected {
            out.append(contentsOf: repeatElement(0, count: expected - out.count))
        }
        return out
    }
}

/// hands the whole input to the converter in slices, then signals end of stream.
private final class ConverterFeed: @unchecked Sendable {
    private let samples: [Float]
    private let format: AVAudioFormat
    private var position = 0

    init(samples: [Float], format: AVAudioFormat) {
        self.samples = samples
        self.format = format
    }

    func next(
        _ count: AVAudioPacketCount, _ status: UnsafeMutablePointer<AVAudioConverterInputStatus>
    ) -> AVAudioBuffer? {
        let remaining = samples.count - position
        guard remaining > 0 else {
            status.pointee = .endOfStream
            return nil
        }
        let n = min(Int(count), remaining)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(n)) else {
            status.pointee = .endOfStream
            return nil
        }
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress! + position, count: n)
        }
        buffer.frameLength = AVAudioFrameCount(n)
        position += n
        status.pointee = .haveData
        return buffer
    }
}
