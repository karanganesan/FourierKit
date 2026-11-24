import AVFoundation
import Accelerate
import AudioToolbox
import CoreAudio
import CoreServices
import Foundation
import Security
import Synchronization

// MARK: types

enum MicrophonePermission: Sendable, Equatable {
    case notDetermined, granted, denied, restricted
}

struct InputDeviceInfo: Sendable, Hashable, Identifiable {
    var id: AudioDeviceID
    var uid: String
    var name: String
    var isDefault: Bool
}

enum CaptureError: Error, Sendable, Equatable {
    /// the user has not been asked yet; call `MicrophoneAccess.request()` first
    case permissionNotRequested
    case permissionDenied
    case permissionRestricted
    /// `NSMicrophoneUsageDescription` is missing from the app's info.plist
    case missingUsageDescription
    /// the app is sandboxed without `com.apple.security.device.audio-input`
    case missingAudioInputEntitlement
    case noInputDevice
    case engineFailed(String)
}

enum CaptureState: Sendable, Equatable {
    case off, starting, on, interrupted
    case failed(CaptureError)
}

enum CaptureEvent: Sendable, Equatable {
    case stateChanged(CaptureState)
    /// the input route changed and capture was rebuilt; audio around the change may be missing
    case configurationChanged(audioMayBeLost: Bool)
    /// the preferred device is gone, so capture follows the system default
    case preferredDeviceUnavailable(uid: String)
    /// voice processing could not start, so capture runs on the plain input
    case voiceProcessingUnavailable(String)
    /// the reader fell behind the ring buffer and this many samples were skipped
    case samplesDropped(count: Int)
}

struct CaptureOptions: Sendable, Equatable {
    /// apple's voice processing: noise suppression, echo cancellation and automatic gain
    var voiceProcessing = true
    /// the system's automatic gain, only with voice processing
    var automaticGainControl = true
    /// core audio device uid, nil follows the system default input
    var preferredDeviceUID: String? = nil
    /// audio kept from before speech starts
    var preRollSeconds = 0.5
    /// ring buffer length; the reader may fall this far behind before audio is lost
    var bufferSeconds = 30.0
    /// chunk sizes delivered on `chunks`
    var minimumChunkSeconds = 0.02
    var maximumChunkSeconds = 0.04

    init() {}
}

// MARK: permission and setup

enum MicrophoneAccess {
    static let usageDescriptionKey = "NSMicrophoneUsageDescription"
    static let audioInputEntitlement = "com.apple.security.device.audio-input"
    static let sandboxEntitlement = "com.apple.security.app-sandbox"
    static let settingsURL = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!

    /// never prompts.
    static var permission: MicrophonePermission {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: .granted
        case .denied: .denied
        case .restricted: .restricted
        case .notDetermined: .notDetermined
        @unknown default: .denied
        }
    }

    /// shows the system prompt only when the user has not been asked yet.
    static func request() async -> MicrophonePermission {
        let current = permission
        guard current == .notDetermined else { return current }
        _ = await AVCaptureDevice.requestAccess(for: .audio)
        return permission
    }

    /// setup mistakes that would make the system kill the app or silently deny it, empty when fine.
    static func setupProblems(bundle: Bundle = .main) -> [CaptureError] {
        var problems = [CaptureError]()
        let usage = bundle.object(forInfoDictionaryKey: usageDescriptionKey) as? String
        if usage?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true {
            problems.append(.missingUsageDescription)
        }
        if isSandboxed && entitlement(audioInputEntitlement) != true {
            problems.append(.missingAudioInputEntitlement)
        }
        return problems
    }

    static var isSandboxed: Bool { entitlement(sandboxEntitlement) == true }

    /// a boolean entitlement of this process, nil when absent or not a boolean.
    static func entitlement(_ key: String) -> Bool? {
        guard let task = SecTaskCreateFromSelf(nil),
            let value = SecTaskCopyValueForEntitlement(task, key as CFString, nil)
        else { return nil }
        return value as? Bool
    }

    /// opens privacy & security > microphone in system settings.
    @discardableResult
    static func openSettings() -> Bool {
        LSOpenCFURLRef(settingsURL as CFURL, nil) == noErr
    }
}

// MARK: devices

enum InputDevices {
    static func list() -> [InputDeviceInfo] {
        let defaultID = defaultInputID()
        return allDeviceIDs().compactMap { id in
            guard inputChannelCount(id) > 0, let uid = uid(of: id) else { return nil }
            return InputDeviceInfo(id: id, uid: uid, name: name(of: id) ?? uid, isDefault: id == defaultID)
        }
    }

    static func defaultInputID() -> AudioDeviceID? {
        var id = AudioDeviceID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &id)
        return status == noErr && id != kAudioObjectUnknown ? id : nil
    }

    static func deviceID(uid: String) -> AudioDeviceID? {
        list().first { $0.uid == uid }?.id
    }

    /// emits the current list at once, then again whenever devices or the default input change.
    static func changes() -> AsyncStream<[InputDeviceInfo]> {
        let (stream, continuation) = AsyncStream.makeStream(
            of: [InputDeviceInfo].self, bufferingPolicy: .bufferingNewest(1))
        let queue = DispatchQueue(label: "fourierkit.audio.devices")
        let listener: AudioObjectPropertyListenerBlock = { _, _ in
            continuation.yield(list())
        }
        let selectors = [kAudioHardwarePropertyDevices, kAudioHardwarePropertyDefaultInputDevice]
        for selector in selectors {
            var address = AudioObjectPropertyAddress(
                mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain)
            AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, queue, listener)
        }
        let registration = ListenerRegistration(listener: listener, queue: queue, selectors: selectors)
        continuation.onTermination = { _ in registration.remove() }
        continuation.yield(list())
        return stream
    }

    private final class ListenerRegistration: @unchecked Sendable {
        let listener: AudioObjectPropertyListenerBlock
        let queue: DispatchQueue
        let selectors: [AudioObjectPropertySelector]

        init(listener: @escaping AudioObjectPropertyListenerBlock, queue: DispatchQueue, selectors: [AudioObjectPropertySelector]) {
            self.listener = listener
            self.queue = queue
            self.selectors = selectors
        }

        func remove() {
            for selector in selectors {
                var address = AudioObjectPropertyAddress(
                    mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                    mElement: kAudioObjectPropertyElementMain)
                AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, queue, listener)
            }
        }
    }

    private static func allDeviceIDs() -> [AudioDeviceID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let system = AudioObjectID(kAudioObjectSystemObject)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size) == noErr, size > 0 else { return [] }
        var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
        guard AudioObjectGetPropertyData(system, &address, 0, nil, &size, &ids) == noErr else { return [] }
        return ids
    }

    private static func inputChannelCount(_ id: AudioDeviceID) -> Int {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioObjectPropertyScopeInput, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size) == noErr, size > 0 else { return 0 }
        let raw = UnsafeMutableRawPointer.allocate(
            byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }
        guard AudioObjectGetPropertyData(id, &address, 0, nil, &size, raw) == noErr else { return 0 }
        let list = UnsafeMutableAudioBufferListPointer(raw.assumingMemoryBound(to: AudioBufferList.self))
        return list.reduce(0) { $0 + Int($1.mNumberChannels) }
    }

    private static func uid(of id: AudioDeviceID) -> String? {
        stringProperty(id, kAudioDevicePropertyDeviceUID)
    }

    private static func name(of id: AudioDeviceID) -> String? {
        stringProperty(id, kAudioObjectPropertyName)
    }

    private static func stringProperty(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = withUnsafeMutablePointer(to: &value) {
            AudioObjectGetPropertyData(id, &address, 0, nil, &size, $0)
        }
        guard status == noErr, let value else { return nil }
        return value.takeRetainedValue() as String
    }
}

// MARK: capture

/// microphone capture into a lock-free ring of 16 khz mono samples.
///
/// one instance is one session: `start()`, then `stop()`, which ends the streams.
/// `start()` and `stop()` block briefly, so call them off the main thread.
/// the render callback only downmixes, converts, writes to the ring and stores the level.
final class MicrophoneCapture: @unchecked Sendable {
    static let sampleRate = 16_000.0

    let options: CaptureOptions
    let ring: RingBuffer

    /// 16 khz mono chunks of about 20–40 ms, in order, from the moment `start()` succeeded
    let chunks: AsyncStream<[Float]>
    /// rms input level (0...1), newest value only
    let levels: AsyncStream<Float>
    let events: AsyncStream<CaptureEvent>

    private let chunkContinuation: AsyncStream<[Float]>.Continuation
    private let levelContinuation: AsyncStream<Float>.Continuation
    private let eventContinuation: AsyncStream<CaptureEvent>.Continuation

    private let level = LevelMeter()
    private let stateBox = Mutex<CaptureState>(.off)

    // everything below is only touched on `control`
    private let control = DispatchQueue(label: "fourierkit.audio.capture", qos: .userInitiated)
    private var engine: AVAudioEngine?
    private var processor: RenderProcessor?
    private var observer: NSObjectProtocol?
    private var reader: Task<Void, Never>?
    private var deviceListener: AudioObjectPropertyListenerBlock?
    private var stopped = false

    init(options: CaptureOptions = CaptureOptions()) {
        self.options = options
        ring = RingBuffer(minimumCapacity: Int(max(options.bufferSeconds, options.preRollSeconds + 1) * Self.sampleRate))
        (chunks, chunkContinuation) = AsyncStream.makeStream(of: [Float].self, bufferingPolicy: .unbounded)
        (levels, levelContinuation) = AsyncStream.makeStream(of: Float.self, bufferingPolicy: .bufferingNewest(1))
        (events, eventContinuation) = AsyncStream.makeStream(of: CaptureEvent.self, bufferingPolicy: .unbounded)
    }

    deinit {
        chunkContinuation.finish()
        levelContinuation.finish()
        eventContinuation.finish()
    }

    var state: CaptureState { stateBox.withLock { $0 } }

    /// newest input level, readable from any thread.
    var currentLevel: (peak: Float, rms: Float) { level.read() }

    /// absolute 16 khz position one past the newest captured sample.
    var position: Int { ring.totalWritten }

    /// the `preRollSeconds` of audio before `position`, or less if not that much is held.
    func preRoll(before position: Int) -> (start: Int, samples: [Float])? {
        let length = Int(options.preRollSeconds * Self.sampleRate)
        let start = max(ring.oldestAvailable, position - length, 0)
        guard let samples = ring.read(from: start, count: position - start) else { return nil }
        return (start, samples)
    }

    /// checks permission and setup, then starts the engine. does not prompt; request permission first.
    func start() throws {
        try control.sync {
            guard !stopped, engine == nil else { return }
            if let problem = MicrophoneAccess.setupProblems().first {
                try fail(problem)
            }
            switch MicrophoneAccess.permission {
            case .granted: break
            case .restricted: try fail(.permissionRestricted)
            case .notDetermined: try fail(.permissionNotRequested)
            case .denied: try fail(.permissionDenied)
            }
            setState(.starting)
            do {
                try startEngine()
            } catch let error as CaptureError {
                try fail(error)
            } catch {
                try fail(.engineFailed(error.localizedDescription))
            }
            watchDevices()
            startReader()
            setState(.on)
        }
    }

    /// stops capture, delivers the audio still in the ring, and ends the streams.
    func stop() {
        let reader: Task<Void, Never>? = control.sync {
            guard !stopped else { return nil }
            stopped = true
            unwatchDevices()
            tearDown()
            setState(.off)
            return self.reader
        }
        // the reader drains what is left and finishes the streams itself
        reader?.cancel()
        if reader == nil { finishStreams() }
    }

    // MARK: engine (control queue)

    /// starts with voice processing if asked, and falls back to the plain input if the system refuses it.
    private func startEngine() throws {
        guard options.voiceProcessing else { return try buildAndStart(voiceProcessing: false) }
        do {
            try buildAndStart(voiceProcessing: true)
        } catch {
            tearDown()
            eventContinuation.yield(.voiceProcessingUnavailable(String(describing: error)))
            try buildAndStart(voiceProcessing: false)
        }
    }

    private func buildAndStart(voiceProcessing: Bool) throws {
        guard InputDevices.defaultInputID() != nil || options.preferredDeviceUID != nil else {
            throw CaptureError.noInputDevice
        }
        let engine = AVAudioEngine()
        // stored first so a failure below is cleaned up by `tearDown()`
        self.engine = engine
        let input = engine.inputNode

        if let uid = options.preferredDeviceUID {
            if let id = InputDevices.deviceID(uid: uid) {
                try setDevice(id, on: input)
            } else {
                eventContinuation.yield(.preferredDeviceUnavailable(uid: uid))
            }
        }

        if voiceProcessing {
            try input.setVoiceProcessingEnabled(true)
            input.isVoiceProcessingAGCEnabled = options.automaticGainControl
            input.voiceProcessingOtherAudioDuckingConfiguration =
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration(enableAdvancedDucking: false, duckingLevel: .min)
            // voice processing needs the output side of the graph; touching it before enabling crashes
            _ = engine.mainMixerNode
        }

        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw CaptureError.noInputDevice }
        guard
            let processor = RenderProcessor(
                inputFormat: format, ring: ring, level: level,
                // with voice processing the input has several channels and the processed voice is the first
                firstChannelOnly: voiceProcessing)
        else { throw CaptureError.engineFailed("could not create the audio converter") }
        self.processor = processor

        // a sink node is called on the render thread with each hardware buffer, unlike a tap,
        // which arrives late and in large blocks
        let sink = AVAudioSinkNode { _, frames, list in
            processor.process(list, frames: Int(frames))
            return noErr
        }
        engine.attach(sink)
        engine.connect(input, to: sink, format: format)
        observer = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            self.control.async { [weak self] in self?.rebuild(reason: .configurationChange) }
        }

        engine.prepare()
        try engine.start()
    }

    private func setDevice(_ id: AudioDeviceID, on input: AVAudioInputNode) throws {
        guard let unit = input.audioUnit else { throw CaptureError.engineFailed("input has no audio unit") }
        var device = id
        let status = AudioUnitSetProperty(
            unit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
            &device, UInt32(MemoryLayout<AudioDeviceID>.size))
        if status != noErr { throw CaptureError.engineFailed("could not select the input device (\(status))") }
    }

    private func tearDown() {
        if let observer { NotificationCenter.default.removeObserver(observer) }
        observer = nil
        engine?.stop()
        engine = nil
        processor = nil
    }

    private enum RebuildReason { case configurationChange, deviceListChange }

    private func rebuild(reason: RebuildReason) {
        guard !stopped else { return }
        switch reason {
        case .configurationChange:
            // the engine has already stopped itself
            guard engine != nil else { return }
        case .deviceListChange:
            // a running engine gets its own configuration change; this only revives a failed capture
            guard engine == nil, case .failed = state else { return }
        }
        setState(.interrupted)
        tearDown()
        eventContinuation.yield(.configurationChanged(audioMayBeLost: true))
        do {
            try startEngine()
            setState(.on)
        } catch let error as CaptureError {
            tearDown()
            setState(.failed(error))
        } catch {
            tearDown()
            setState(.failed(.engineFailed(error.localizedDescription)))
        }
    }

    /// while a session runs, device changes can revive a capture that failed, e.g. when a mic is plugged back in.
    private func watchDevices() {
        guard deviceListener == nil else { return }
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.rebuild(reason: .deviceListChange)
        }
        for selector in Self.deviceSelectors {
            var address = AudioObjectPropertyAddress(
                mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
            AudioObjectAddPropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, control, listener)
        }
        deviceListener = listener
    }

    private func unwatchDevices() {
        guard let listener = deviceListener else { return }
        for selector in Self.deviceSelectors {
            var address = AudioObjectPropertyAddress(
                mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
            AudioObjectRemovePropertyListenerBlock(AudioObjectID(kAudioObjectSystemObject), &address, control, listener)
        }
        deviceListener = nil
    }

    private static let deviceSelectors = [kAudioHardwarePropertyDevices, kAudioHardwarePropertyDefaultInputDevice]

    private func fail(_ error: CaptureError) throws -> Never {
        tearDown()
        setState(.failed(error))
        throw error
    }

    private func setState(_ new: CaptureState) {
        let changed = stateBox.withLock { state -> Bool in
            guard state != new else { return false }
            state = new
            return true
        }
        if changed { eventContinuation.yield(.stateChanged(new)) }
    }

    private func finishStreams() {
        chunkContinuation.finish()
        levelContinuation.finish()
        eventContinuation.finish()
    }

    // MARK: reader (background task)

    private func startReader() {
        let ring = ring
        let level = level
        let chunkOut = chunkContinuation
        let levelOut = levelContinuation
        let eventOut = eventContinuation
        let minimum = max(1, Int(options.minimumChunkSeconds * Self.sampleRate))
        let maximum = max(minimum, Int(options.maximumChunkSeconds * Self.sampleRate))
        let startPosition = ring.totalWritten

        reader = Task.detached(priority: .userInitiated) {
            var position = startPosition
            var lastLevelStamp = -1
            var scratch = [Float](repeating: 0, count: maximum)

            func drain(final: Bool) {
                while true {
                    let available = ring.totalWritten - position
                    guard available >= (final ? 1 : minimum) else { return }
                    let n = min(available, maximum)
                    let result = scratch.withUnsafeMutableBufferPointer {
                        ring.read(from: position, count: n, into: $0.baseAddress!)
                    }
                    switch result {
                    case .read(let count):
                        chunkOut.yield(Array(scratch[0..<count]))
                        position += count
                    case .overrun(let oldest):
                        // skip ahead with some slack so the writer does not lap us again at once
                        let resume = oldest + ring.capacity / 4
                        eventOut.yield(.samplesDropped(count: resume - position))
                        position = resume
                    }
                }
            }

            while !Task.isCancelled {
                drain(final: false)
                let (stamp, rms) = level.latest()
                if stamp != lastLevelStamp {
                    lastLevelStamp = stamp
                    levelOut.yield(rms)
                }
                try? await Task.sleep(for: .milliseconds(5))
            }
            drain(final: true)
            chunkOut.finish()
            levelOut.finish()
            eventOut.finish()
        }
    }
}

// MARK: realtime path

/// newest peak and rms, published from the render thread without locks.
final class LevelMeter: Sendable {
    private let peakBits = Atomic<UInt32>(0)
    private let rmsBits = Atomic<UInt32>(0)
    private let stamp = Atomic<Int>(0)

    func store(peak: Float, rms: Float) {
        peakBits.store(peak.bitPattern, ordering: .relaxed)
        rmsBits.store(rms.bitPattern, ordering: .relaxed)
        stamp.wrappingAdd(1, ordering: .releasing)
    }

    func read() -> (peak: Float, rms: Float) {
        _ = stamp.load(ordering: .acquiring)
        return (Float(bitPattern: peakBits.load(ordering: .relaxed)), Float(bitPattern: rmsBits.load(ordering: .relaxed)))
    }

    func latest() -> (stamp: Int, rms: Float) {
        let s = stamp.load(ordering: .acquiring)
        return (s, Float(bitPattern: rmsBits.load(ordering: .relaxed)))
    }
}

/// owns everything the render callback touches. all buffers and the converter's input block are made up
/// front, so steady-state callbacks allocate nothing on our side.
final class RenderProcessor: @unchecked Sendable {
    private let ring: RingBuffer
    private let level: LevelMeter
    private let channels: Int
    private let firstChannelOnly: Bool
    private let converter: AVAudioConverter?
    private let mono: AVAudioPCMBuffer
    private let output: AVAudioPCMBuffer
    private let capacity: AVAudioFrameCount
    private let pending: PendingInput
    private let inputBlock: AVAudioConverterInputBlock

    init?(inputFormat: AVAudioFormat, ring: RingBuffer, level: LevelMeter, firstChannelOnly: Bool) {
        self.ring = ring
        self.level = level
        self.firstChannelOnly = firstChannelOnly
        channels = Int(inputFormat.channelCount)
        // larger callbacks are processed in slices of this size
        capacity = AVAudioFrameCount(max(4096, Int(inputFormat.sampleRate * 0.1)))

        guard
            let monoFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: inputFormat.sampleRate, channels: 1, interleaved: false),
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: MicrophoneCapture.sampleRate, channels: 1,
                interleaved: false),
            let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: capacity)
        else { return nil }
        let ratio = MicrophoneCapture.sampleRate / inputFormat.sampleRate
        guard
            let output = AVAudioPCMBuffer(
                pcmFormat: outFormat, frameCapacity: AVAudioFrameCount(Double(capacity) * ratio) + 64)
        else { return nil }
        self.mono = mono
        self.output = output
        let pending = PendingInput(mono)
        self.pending = pending
        // hands the converter the current slice once per call, then reports no data so it keeps its state
        inputBlock = { _, status in
            guard pending.isPending else {
                status.pointee = .noDataNow
                return nil
            }
            pending.isPending = false
            status.pointee = .haveData
            return pending.buffer
        }

        if inputFormat.sampleRate == MicrophoneCapture.sampleRate {
            converter = nil
        } else {
            guard let converter = AVAudioConverter(from: monoFormat, to: outFormat) else { return nil }
            converter.sampleRateConverterQuality = AVAudioQuality.high.rawValue
            converter.primeMethod = .normal
            self.converter = converter
        }
    }

    /// called on the realtime render thread. no locks, no waiting, no allocation on our side.
    func process(_ list: UnsafePointer<AudioBufferList>, frames total: Int) {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        guard buffers.count > 0, total > 0 else { return }
        var offset = 0
        var peak: Float = 0
        var sumSquares: Float = 0
        while offset < total {
            let n = min(Int(capacity), total - offset)
            let dst = mono.floatChannelData![0]
            guard downmix(buffers, offset: offset, count: n, into: dst) else { return }
            mono.frameLength = AVAudioFrameCount(n)

            var slicePeak: Float = 0
            var sliceSquares: Float = 0
            vDSP_maxmgv(dst, 1, &slicePeak, vDSP_Length(n))
            vDSP_svesq(dst, 1, &sliceSquares, vDSP_Length(n))
            peak = max(peak, slicePeak)
            sumSquares += sliceSquares

            if let converter {
                pending.isPending = true
                output.frameLength = 0
                var error: NSError?
                _ = converter.convert(to: output, error: &error, withInputFrom: inputBlock)
                pending.isPending = false
                if output.frameLength > 0 {
                    ring.write(output.floatChannelData![0], count: Int(output.frameLength))
                }
            } else {
                ring.write(dst, count: n)
            }
            offset += n
        }
        level.store(peak: peak, rms: (sumSquares / Float(total)).squareRoot())
    }

    /// averages the used channels into `dst`; false if the buffer list is not float audio we can read.
    private func downmix(
        _ buffers: UnsafeMutableAudioBufferListPointer, offset: Int, count n: Int,
        into dst: UnsafeMutablePointer<Float>
    ) -> Bool {
        let used = firstChannelOnly ? 1 : channels
        if buffers.count == 1 && buffers[0].mNumberChannels > 1 {
            // one interleaved buffer
            let stride = Int(buffers[0].mNumberChannels)
            guard let raw = buffers[0].mData else { return false }
            let src = raw.assumingMemoryBound(to: Float.self) + offset * stride
            let mixed = min(used, stride)
            vDSP_mmov(src, dst, 1, vDSP_Length(n), vDSP_Length(stride), 1)
            for c in 1..<max(1, mixed) {
                vDSP_vadd(dst, 1, src + c, vDSP_Stride(stride), dst, 1, vDSP_Length(n))
            }
            scale(dst, count: n, channels: mixed)
        } else {
            let mixed = min(used, buffers.count)
            for c in 0..<mixed {
                guard let raw = buffers[c].mData else { return false }
                let src = raw.assumingMemoryBound(to: Float.self) + offset
                if c == 0 {
                    dst.update(from: src, count: n)
                } else {
                    vDSP_vadd(dst, 1, src, 1, dst, 1, vDSP_Length(n))
                }
            }
            scale(dst, count: n, channels: mixed)
        }
        return true
    }

    private func scale(_ dst: UnsafeMutablePointer<Float>, count n: Int, channels: Int) {
        guard channels > 1 else { return }
        var factor = 1 / Float(channels)
        vDSP_vsmul(dst, 1, &factor, dst, 1, vDSP_Length(n))
    }
}

/// the slice waiting for the converter; only the render thread touches it.
private final class PendingInput: @unchecked Sendable {
    let buffer: AVAudioPCMBuffer
    var isPending = false

    init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }
}
