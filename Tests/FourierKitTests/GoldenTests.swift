import Foundation
import Testing
@_spi(Golden) import FourierKit

/// the golden test: three clips, many dimensions, and nothing else.
///
/// the frozen reference in golden/ is hugging face transformers on the original weights. every
/// must-pass dimension is checked on all three clips, and the scored dimensions are appended to
/// docs/performance.jsonl. run it with `swift test -c release` on a quiet machine.
@Suite(.serialized)
struct GoldenTests {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let recorded = 10

    struct Clip {
        var name: String
        var samples: [Float]
        var seconds: Double
        var tokens: [Int32]
        var finalText: String
        var stopToken: Int32
        var steps: Int
        var smallestGap: Double
        var topScores: [Float]
        var topIndices: [Int32]
    }

    static func modelDirectory() -> URL {
        if let override = ProcessInfo.processInfo.environment["FOURIER_MODEL_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return GoldenProbe.defaultModelDirectory
    }

    static func loadClips() throws -> ([Clip], [String: Any]) {
        let data = try Data(contentsOf: root.appendingPathComponent("golden/manifest.json"))
        let manifest = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let entries = manifest["clips"] as! [String: [String: Any]]
        var clips: [Clip] = []
        for name in ["short", "medium", "long"] {
            let entry = try #require(entries[name], "golden/manifest.json has no \(name) clip")
            let tensorURL = root.appendingPathComponent(entry["tensors"] as! String)
            guard FileManager.default.fileExists(atPath: tensorURL.path) else {
                throw GoldenFailure("\(tensorURL.path) is missing. fetch the golden tensors (see CONTRIBUTING.md) or remake them with reference/python/make_golden.py")
            }
            let samples = try GoldenProbe.readAudio(root.appendingPathComponent(entry["file"] as! String))
            #expect(samples.count == entry["samples"] as! Int, "\(name): the clip does not decode to the frozen samples")
            clips.append(Clip(
                name: name,
                samples: samples,
                seconds: Double(samples.count) / 16_000,
                tokens: (entry["token_ids"] as! [Int]).map(Int32.init),
                finalText: entry["final_text"] as! String,
                stopToken: Int32(entry["stop_token"] as! Int),
                steps: entry["steps"] as! Int,
                smallestGap: entry["smallest_gap"] as! Double,
                topScores: try GoldenProbe.floats(in: tensorURL, named: "top_scores"),
                topIndices: try GoldenProbe.ints(in: tensorURL, named: "top_indices")
            ))
        }
        return (clips, manifest)
    }

    struct GoldenFailure: Error, CustomStringConvertible {
        var description: String
        init(_ description: String) { self.description = description }
    }

    typealias Run = GoldenProbe.Run

    static func run(_ clip: Clip) async throws -> Run {
        try await GoldenProbe.transcribe(samples: clip.samples, candidates: recorded)
    }

    /// how much of the safety margin a run uses, over every comparison beam search makes.
    ///
    /// a comparison keeps its order when our gap is off by less than half the reference gap, so
    /// the worst ratio of gap error to half the gap must stay under 1. gaps are compared, not raw
    /// scores, because an error shared by both sides of a comparison cannot change its outcome.
    static func scoreError(_ clip: Clip, _ result: Run) -> (used: Double, error: Double, sameCandidates: Bool) {
        var used = 0.0
        var worst = 0.0
        var same = result.scores.count == clip.steps
        for step in 0..<min(clip.steps, result.scores.count) {
            let ours = result.scores[step]
            for rank in 0..<min(9, ours.count) {
                let reference = clip.topScores[step * recorded + rank]
                if result.indices[step][rank] != clip.topIndices[step * recorded + rank] { same = false }
                guard reference > -1e8 else { continue }
                worst = max(worst, Double(abs(ours[rank] - reference)))
                let next = clip.topScores[step * recorded + rank + 1]
                guard next > -1e8, rank + 1 < ours.count else { continue }
                let wanted = Double(reference) - Double(next)
                let got = Double(ours[rank]) - Double(ours[rank + 1])
                used = max(used, abs(got - wanted) / (wanted / 2))
            }
        }
        return (used, worst, same)
    }

    static func sameScores(_ a: Run, _ b: Run) -> Bool {
        guard a.tokens == b.tokens, a.indices == b.indices, a.scores.count == b.scores.count else { return false }
        for (x, y) in zip(a.scores, b.scores) where x.map(\.bitPattern) != y.map(\.bitPattern) { return false }
        return true
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
    }

    static func peakMemory() -> UInt64 {
        var info = rusage_info_current()
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, $0) }
        }
        return status == 0 ? info.ri_lifetime_max_phys_footprint : 0
    }

    @Test(.timeLimit(.minutes(60)))
    func golden() async throws {
        var failures: [String] = []
        func expect(_ passed: Bool, _ message: @autoclosure () -> String = "a golden check failed", line: Int = #line) {
            guard !passed else { return }
            let text = "line \(line): \(message())"
            failures.append(text)
            Issue.record(Comment(rawValue: text))
        }
        let (clips, manifest) = try Self.loadClips()
        let directory = Self.modelDirectory()
        let now = GoldenProbe.now

        var clock = now()
        let loaded = try await GoldenProbe.load(modelDirectory: directory)
        let coldLoad = loaded.seconds
        let beams = loaded.beams
        let generation = manifest["generation"] as! [String: Any]
        expect(beams == generation["num_beams"] as! Int)
        expect(loaded.bans == generation["banned_tokens"] as! Int)

        var log: [String: Any] = [:]
        var perClip: [String: [String: Any]] = [:]
        var alone: [String: Run] = [:]

        // each clip alone: tokens, text, stop, the safety margin, cold and warm repeats
        for clip in clips {
            var results: [Run] = []
            var walls: [Double] = []
            for _ in 0..<5 {
                clock = now()
                results.append(try await Self.run(clip))
                walls.append(now() - clock)
            }
            let first = results[0]
            expect(first.tokens == clip.tokens, "\(clip.name): token ids differ from the reference")
            expect(first.text == clip.finalText, "\(clip.name): final text differs from the reference")
            expect(first.tokens.last == clip.stopToken && first.tokens.count == clip.tokens.count, "\(clip.name): stops at a different token or length")
            expect(first.reachedStop)
            for (run, result) in results.enumerated().dropFirst() {
                expect(Self.sameScores(first, result), "\(clip.name): run \(run) is not bit-identical to the first run")
            }
            let (used, error, sameCandidates) = Self.scoreError(clip, first)
            expect(sameCandidates, "\(clip.name): the ranked candidates differ from the reference")
            expect(used < 1, "\(clip.name): a beam comparison uses \(used) of its safety margin")
            alone[clip.name] = first

            let warm = Array(walls.dropFirst())
            let wall = Self.median(warm)
            let best = results[1...].min { $0.total < $1.total }!
            perClip[clip.name] = [
                "seconds": clip.seconds,
                "tokens": clip.tokens.count,
                "steps": first.steps,
                "cold_wall": walls[0],
                "warm_wall": wall,
                "speed": clip.seconds / wall,
                "tokens_per_second": Double(clip.tokens.count) / wall,
                "time_to_first_text": Self.median(results[1...].map(\.firstToken)),
                "mel": best.mel,
                "encode": best.encode,
                "prefill": best.prefill,
                "decode": best.decode,
                "decoder_steps_per_second": Double(first.steps - 1) / max(best.decode, 1e-9),
                "score_error": error,
                "smallest_gap": clip.smallestGap,
                "margin_used": used,
            ]
        }

        // any batch or load: all three in one wide batch, then six pieces through a batch that
        // only fits two, so slots are refilled mid-flight. scores must stay bit-identical.
        await GoldenProbe.setMaxRows(12)
        clock = now()
        let together = try await withThrowingTaskGroup(of: (String, Run).self) { group in
            for clip in clips { group.addTask { (clip.name, try await Self.run(clip)) } }
            var out: [String: Run] = [:]
            for try await (name, result) in group { out[name] = result }
            return out
        }
        let batchWall = now() - clock
        let audioSeconds = clips.reduce(0) { $0 + $1.seconds }
        for clip in clips {
            expect(Self.sameScores(alone[clip.name]!, together[clip.name]!), "\(clip.name): scores change when it shares a batch")
        }
        expect(together.values.contains { $0.widestBatch > beams }, "the clips never shared a batch")

        await GoldenProbe.setMaxRows(beams * 2)
        let refilled = try await withThrowingTaskGroup(of: (String, Run).self) { group in
            for round in 0..<2 {
                for clip in clips where clip.name != "long" || round == 0 {
                    group.addTask { (clip.name, try await Self.run(clip)) }
                }
            }
            var out: [(String, Run)] = []
            for try await pair in group { out.append(pair) }
            return out
        }
        for (name, result) in refilled {
            expect(Self.sameScores(alone[name]!, result), "\(name): scores change after a slot refill")
        }

        // sustained throughput: the three clips, three times over, at the widest batch
        await GoldenProbe.setMaxRows(12)
        clock = now()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<3 {
                for clip in clips { group.addTask { _ = try await Self.run(clip) } }
            }
            try await group.waitForAll()
        }
        let sustainedWall = now() - clock

        // warm load: drop the engine and map it again
        await GoldenProbe.unload()
        let warmLoad = try await GoldenProbe.load(modelDirectory: directory).seconds
        let reloaded = try await Self.run(clips[0])
        expect(Self.sameScores(alone[clips[0].name]!, reloaded), "short: scores change after a reload")

        log["date"] = ISO8601DateFormatter().string(from: Date())
        log["label"] = ProcessInfo.processInfo.environment["FOURIER_RUN_LABEL"] ?? "golden run"
        log["clips"] = perClip
        log["batch_throughput"] = audioSeconds / batchWall
        log["sustained_throughput"] = audioSeconds * 3 / sustainedWall
        log["cold_load"] = coldLoad
        log["warm_load"] = warmLoad
        log["peak_memory_bytes"] = Self.peakMemory()
        log["thermal_state"] = ProcessInfo.processInfo.thermalState.rawValue
        log["placement"] = ["mel": "cpu", "encoder": "gpu", "decoder": "gpu", "search": "cpu"]
        log["passed"] = failures.isEmpty
        log["failures"] = failures
        let line = try JSONSerialization.data(withJSONObject: log, options: [.sortedKeys])
        let logURL = Self.root.appendingPathComponent("docs/performance.jsonl")
        if let handle = try? FileHandle(forWritingTo: logURL) {
            try handle.seekToEnd()
            try handle.write(contentsOf: line + Data("\n".utf8))
            try handle.close()
        } else {
            try (line + Data("\n".utf8)).write(to: logURL)
        }
        print(String(data: try JSONSerialization.data(withJSONObject: log, options: [.sortedKeys, .prettyPrinted]), encoding: .utf8)!)
    }

    /// every mode, through the public api: a file, a recording, and the clip played in at real
    /// speed as if spoken. live numbers go to the same measurement log.
    @Test(.timeLimit(.minutes(30)))
    func modes() async throws {
        var failures: [String] = []
        func expect(_ passed: Bool, _ message: @autoclosure () -> String, line: Int = #line) {
            guard !passed else { return }
            let text = "line \(line): \(message())"
            failures.append(text)
            Issue.record(Comment(rawValue: text))
        }
        let (clips, _) = try Self.loadClips()
        var options = FourierOptions()
        if ProcessInfo.processInfo.environment["FOURIER_MODEL_DIR"] != nil { options.model.directory = Self.modelDirectory() }
        let speech = Fourier(options)
        try await speech.prepare()
        var live: [String: [String: Any]] = [:]

        for clip in clips {
            let url = Self.root.appendingPathComponent("golden/clips/\(clip.name).wav")

            let fromFile = try await speech.transcribe(contentsOf: url)
            expect(fromFile.text == clip.finalText, "\(clip.name): a file through the api differs from the reference")
            expect(fromFile.pieces.count == 1, "\(clip.name): a clip that fits one pass was split")

            var recording = options
            recording.audioSource = FileAudioSource(url: url, realtime: false)
            let recorder = try await Fourier(recording).record()
            let deadline = GoldenProbe.now() + 60
            while recorder.duration < .seconds(clip.seconds), GoldenProbe.now() < deadline {
                try await Task.sleep(for: .milliseconds(20))
            }
            let recorded = await recorder.stop()
            expect(recorded.samples == clip.samples, "\(clip.name): the recording is not the samples that went in")
            let fromRecording = try await speech.transcribe(recorded)
            expect(fromRecording.text == clip.finalText, "\(clip.name): a recording through the api differs from the reference")

            var listening = ListenOptions()
            listening.audioSource = FileAudioSource(url: url, realtime: true)
            let session = try await speech.listen(listening)
            let started = GoldenProbe.now()
            let watcher = Task { () -> (broken: String?, updates: Int, draftChanges: Int, firstText: Double?, delays: [Double]) in
                var settled = ""
                var draft = ""
                var broken: String?
                var updates = 0
                var draftChanges = 0
                var firstText: Double?
                var delays: [Double] = []
                for try await event in session.events {
                    switch event {
                    case .update(let update):
                        updates += 1
                        if !update.settled.hasPrefix(settled) { broken = "settled text was taken back: \(settled) became \(update.settled)" }
                        if update.text != update.settled + update.draft { broken = "text is not settled + draft" }
                        if update.draft != draft { draftChanges += 1 }
                        if firstText == nil, !update.text.isEmpty { firstText = GoldenProbe.now() - started }
                        settled = update.settled
                        draft = update.draft
                        delays.append(Double(update.delay.components.seconds) + Double(update.delay.components.attoseconds) * 1e-18)
                    case .warning(.inputEnded):
                        return (broken, updates, draftChanges, firstText, delays)
                    default:
                        break
                    }
                }
                return (broken, updates, draftChanges, firstText, delays)
            }
            let watched = try await watcher.value
            let finishStarted = GoldenProbe.now()
            let final = try await session.finish()
            let finishWait = GoldenProbe.now() - finishStarted
            expect(watched.broken == nil, "\(clip.name): \(watched.broken ?? "")")
            expect(!final.pieces.isEmpty, "\(clip.name): the live session heard nothing")
            for piece in final.pieces {
                let range = piece.samples.clamped(to: 0..<clip.samples.count)
                let alone = try await speech.transcribe(samples: Array(clip.samples[range]))
                expect(alone.text == piece.text, "\(clip.name): a live phrase differs from transcribe on the same split points")
            }
            expect(final.text == final.pieces.map(\.text).filter { !$0.isEmpty }.joined(separator: " "), "\(clip.name): the live final is not its phrases joined")
            live[clip.name] = [
                "updates": watched.updates,
                "draft_changes": watched.draftChanges,
                "first_text": watched.firstText ?? -1,
                "median_delay": Self.median(watched.delays),
                "worst_delay": watched.delays.max() ?? -1,
                "phrases": final.pieces.count,
                "finish_wait": finishWait,
                "final_equals_reference": final.text == clip.finalText,
            ]
        }

        let log: [String: Any] = [
            "date": ISO8601DateFormatter().string(from: Date()),
            "label": ProcessInfo.processInfo.environment["FOURIER_RUN_LABEL"] ?? "golden run",
            "kind": "modes",
            "live": live,
            "passed": failures.isEmpty,
            "failures": failures,
        ]
        let line = try JSONSerialization.data(withJSONObject: log, options: [.sortedKeys]) + Data("\n".utf8)
        let logURL = Self.root.appendingPathComponent("docs/performance.jsonl")
        if let handle = try? FileHandle(forWritingTo: logURL) {
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            try handle.close()
        } else {
            try line.write(to: logURL)
        }
        print(String(data: try JSONSerialization.data(withJSONObject: log, options: [.sortedKeys, .prettyPrinted]), encoding: .utf8)!)
    }
}
