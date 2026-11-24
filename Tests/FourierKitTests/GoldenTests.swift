import CryptoKit
import Foundation
import Testing
@_spi(Golden) import FourierKit

/// the golden test: three clips, many dimensions, and nothing else.
///
/// the frozen reference in golden/ is hugging face transformers on the original weights. every
/// must-pass dimension is checked on all three clips, and the scored dimensions are appended to
/// docs/performance.jsonl, passed or failed. run it with `swift test -c release` on a quiet machine.
///
/// what the reference pins: tokens, text, the stop, the ranked top 10 candidates of every step, and
/// (through the safety margin) the order of those candidates, of the finished set and of the stop
/// heuristic. mel, encoder output and first logits are diagnostics, with bounds that only catch a
/// broken stage. clip shapes the reference never saw (lengths that are not a multiple of 160
/// samples, clips under 0.5 s) are only run, see `unverifiedClipShapesRunAndStop`.
@Suite(.serialized)
struct GoldenTests {
    static let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    static let recorded = 10
    /// the most of a comparison's gap the engine's error may use, and what make_golden.py rejects
    /// a clip at when the manifest does not say
    static let marginLimit = 0.5

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
        var tensorURL: URL
        var melFrames: Int
        var audioTokens: Int
    }

    struct GoldenFailure: Error, CustomStringConvertible {
        var description: String
        init(_ description: String) { self.description = description }
    }

    /// collects failures so a failed or thrown run can still be logged
    final class Checks {
        var failures: [String] = []
        func expect(_ passed: Bool, _ message: @autoclosure () -> String, line: Int = #line) {
            guard !passed else { return }
            let text = "line \(line): \(message())"
            failures.append(text)
            Issue.record(Comment(rawValue: text))
        }
    }

    static func modelDirectory() -> URL {
        if let override = ProcessInfo.processInfo.environment["FOURIER_MODEL_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return GoldenProbe.defaultModelDirectory
    }

    /// the engine reads the model from disk and never downloads it. a machine without it fails here.
    static func requireModel(_ directory: URL) throws {
        let missing = GoldenProbe.missingModelFiles(in: directory)
        guard missing.isEmpty else {
            throw GoldenFailure("the model is not at \(directory.path) (missing \(missing.joined(separator: ", "))). prepare it first by running prepare() once with network access, or set FOURIER_MODEL_DIR to a prepared copy")
        }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func loadClips() throws -> ([Clip], [String: Any]) {
        let data = try Data(contentsOf: root.appendingPathComponent("golden/manifest.json"))
        let manifest = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let remake = "fetch the golden files again (see CONTRIBUTING.md) or remake them with reference/python/make_golden.py"
        // a set that was written without the 64-bit noise check proves nothing about exactness
        if manifest["validated"] as? Bool == false {
            throw GoldenFailure("golden/manifest.json is marked unvalidated: it was made without the 64-bit noise check. remake it with reference/python/make_golden.py and no --unvalidated")
        }
        let limit = manifest["margin_threshold"] as? Double ?? marginLimit
        let entries = manifest["clips"] as! [String: [String: Any]]
        var clips: [Clip] = []
        for name in ["short", "medium", "long"] {
            let entry = try #require(entries[name], "golden/manifest.json has no \(name) clip")
            // each clip must have passed the 64-bit run: same tokens, same candidates, a margin under the limit
            guard let wide = entry["float64"] as? [String: Any], wide["same_tokens"] as? Bool == true,
                  wide["same_candidates"] as? Bool == true, let used = wide["margin_used"] as? Double, used < limit
            else {
                throw GoldenFailure("\(name): golden/manifest.json has no passing 64-bit check (same tokens, same candidates, margin used under \(limit)), so the clip is not validated. remake it with reference/python/make_golden.py")
            }
            let tensorURL = root.appendingPathComponent(entry["tensors"] as! String)
            guard FileManager.default.fileExists(atPath: tensorURL.path) else {
                throw GoldenFailure("\(tensorURL.path) is missing. \(remake)")
            }
            let fileURL = root.appendingPathComponent(entry["file"] as! String)
            guard sha256(try Data(contentsOf: fileURL)) == entry["file_sha256"] as! String else {
                throw GoldenFailure("\(entry["file"] as! String) does not match the manifest's file hash. \(remake)")
            }
            guard sha256(try Data(contentsOf: tensorURL)) == entry["tensors_sha256"] as! String else {
                throw GoldenFailure("\(entry["tensors"] as! String) does not match the manifest's tensor hash. \(remake)")
            }
            let samples = try GoldenProbe.readAudio(fileURL)
            guard sha256(samples.withUnsafeBytes { Data($0) }) == entry["samples_sha256"] as! String else {
                throw GoldenFailure("\(name): the clip decodes to different samples than the manifest's samples hash. \(remake)")
            }
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
                topIndices: try GoldenProbe.ints(in: tensorURL, named: "top_indices"),
                tensorURL: tensorURL,
                melFrames: entry["mel_frames"] as! Int,
                audioTokens: entry["audio_tokens"] as! Int
            ))
        }
        // note: the manifest's reference_seconds is one instrumented transformers run (extra top-k,
        // output scores and logits, and the first call's warm-up on the first clip), set against our
        // warm median. it is not a plain warm generate. new manifests also carry
        // reference_seconds_plain, which is.
        return (clips, manifest)
    }

    typealias Run = GoldenProbe.Run

    static func run(_ clip: Clip, beams: Int? = nil) async throws -> Run {
        guard let beams else { return try await GoldenProbe.transcribe(samples: clip.samples, candidates: recorded) }
        return try await GoldenProbe.transcribe(samples: clip.samples, beams: beams, candidates: recorded)
    }

    /// the parts of the manifest's generation block the margin needs
    struct Generation {
        var beams: Int
        var eos: Set<Int32>
        var lengthPenalty: Double

        init(_ manifest: [String: Any]) {
            let generation = manifest["generation"] as! [String: Any]
            beams = generation["num_beams"] as! Int
            eos = Set((generation["eos_token_id"] as! [Int]).map(Int32.init))
            lengthPenalty = generation["length_penalty"] as! Double
        }
    }

    struct Margin {
        /// the largest score difference to the reference over the recorded candidates
        var error = 0.0
        /// top-k: the worst gap error over half the reference gap, between every pair of neighbours
        /// among the 10 recorded candidates, so the 9th/10th cut that decides which are kept is in it
        var used = 0.0
        /// the same ratio over the finished-set ranking and the stop heuristic
        var finishedUsed = 0.0
        var sameCandidates = true
    }

    /// how much of the safety margin a run uses.
    ///
    /// a comparison keeps its order when our gap is off by less than half the reference gap, so
    /// the worst ratio of gap error to half the gap must stay under 1. gaps are compared, not raw
    /// scores, because an error shared by both sides of a comparison cannot change its outcome.
    ///
    /// covered, all from the recorded top 10 of each step: the order of neighbouring candidates
    /// (which, by transitivity, fixes every pair, so also the picks among the non-finished ones and
    /// the order of the kept 9), the cut between the 9th and 10th, the ranking of every finished
    /// hypothesis against every other by length-penalised score (adjacent pairs of all of them), and
    /// the stop heuristic (the best running beam against the worst of the finished set). not
    /// covered: finished hypotheses that fall outside the recorded 10, and the order among
    /// finished-then-masked candidates, which is never used because at least `num_beams` candidates
    /// of the kept 9 are always running.
    static func scoreError(_ clip: Clip, _ result: Run, _ generation: Generation) -> Margin {
        scoreError(clip, scores: result.scores, indices: result.indices, generation)
    }

    static func scoreError(_ clip: Clip, scores: [[Float]], indices: [[Int32]], _ generation: Generation) -> Margin {
        var margin = Margin()
        func ratio(_ wanted: Double, _ got: Double) -> Double {
            wanted == 0 ? .infinity : abs(got - wanted) / (abs(wanted) / 2)
        }
        margin.sameCandidates = scores.count == clip.steps && indices.count == clip.steps
        let keep = max(2, 1 + generation.eos.count) * generation.beams
        var finished: [(reference: Double, ours: Double)] = []
        for step in 0..<min(clip.steps, scores.count) {
            let ours = scores[step]
            guard ours.count == recorded, indices[step].count == recorded else { margin.sameCandidates = false; continue }
            for rank in 0..<recorded {
                let reference = clip.topScores[step * recorded + rank]
                if indices[step][rank] != clip.topIndices[step * recorded + rank] { margin.sameCandidates = false }
                guard reference > -1e8 else { continue }
                margin.error = max(margin.error, Double(abs(ours[rank] - reference)))
                guard rank + 1 < recorded else { continue }
                let next = clip.topScores[step * recorded + rank + 1]
                guard next > -1e8 else { continue }
                let wanted = Double(reference) - Double(next)
                margin.used = max(margin.used, wanted > 0 ? ratio(wanted, Double(ours[rank]) - Double(ours[rank + 1])) : .infinity)
            }

            // finished set: a hypothesis that hit a stop token among the best num_beams enters with
            // its score over length ** length_penalty. this step's length is step + 1.
            let penalty = Foundation.pow(Double(step + 1), generation.lengthPenalty)
            func isStop(_ rank: Int) -> Bool {
                generation.eos.contains(Int32(Int(clip.topIndices[step * recorded + rank]) % GoldenProbe.vocabulary))
            }
            for rank in 0..<generation.beams where isStop(rank) && clip.topScores[step * recorded + rank] > -1e8 {
                finished.append((Double(clip.topScores[step * recorded + rank]) / penalty, Double(ours[rank]) / penalty))
            }
            // stop heuristic, once the set holds num_beams real hypotheses: the best candidate that
            // keeps running, over the same penalty, against the set's worst
            if finished.count >= generation.beams, let best = (0..<keep).first(where: { !isStop($0) }) {
                let worst = finished.sorted { $0.reference > $1.reference }[generation.beams - 1]
                let top = Double(clip.topScores[step * recorded + best]) / penalty
                margin.finishedUsed = max(margin.finishedUsed, ratio(top - worst.reference, Double(ours[best]) / penalty - worst.ours))
            }
        }
        let ranked = finished.sorted { $0.reference > $1.reference }
        for pair in zip(ranked, ranked.dropFirst()) {
            margin.finishedUsed = max(margin.finishedUsed, ratio(pair.0.reference - pair.1.reference, pair.0.ours - pair.1.ours))
        }
        return margin
    }

    static func sameScores(_ a: Run, _ b: Run) -> Bool {
        guard a.tokens == b.tokens, a.indices == b.indices, a.scores.count == b.scores.count else { return false }
        for (x, y) in zip(a.scores, b.scores) where x.map(\.bitPattern) != y.map(\.bitPattern) { return false }
        return true
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return 0 }
        let middle = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
    }

    static func peakMemory() -> UInt64 {
        var info = rusage_info_current()
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT, $0) }
        }
        return status == 0 ? info.ri_lifetime_max_phys_footprint : 0
    }

    static func appendToLog(_ log: [String: Any]) throws {
        let line = try JSONSerialization.data(withJSONObject: log, options: [.sortedKeys]) + Data("\n".utf8)
        let logURL = root.appendingPathComponent("docs/performance.jsonl")
        if let handle = try? FileHandle(forWritingTo: logURL) {
            try handle.seekToEnd()
            try handle.write(contentsOf: line)
            try handle.close()
        } else {
            try line.write(to: logURL)
        }
        print(String(data: try JSONSerialization.data(withJSONObject: log, options: [.sortedKeys, .prettyPrinted]), encoding: .utf8)!)
    }

    // MARK: batch widths

    /// batches of every width from 1 to 12 that production can make, every beam count from 1 to 4
    /// (a greedy draft beside a beam-3 final makes 4 rows, a width that is no multiple of 3). each
    /// is a list of beam counts, in the order the pieces enter the batch; the first and last
    /// piece of every list are the ones at the edges. the medium clip is long enough for all of
    /// them to overlap.
    static let mixes: [[Int]] = [
        [1, 1], [1, 2], [1, 3], [3, 1], [2, 3], [3, 4], [4, 3],
        [1, 3, 4], [3, 3, 4], [4, 4, 3], [3, 3, 3, 3], [4, 3, 2, 1, 2],
    ]
    static let requiredWidths: Set<Int> = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12]

    /// each piece's tokens and recorded candidates, bit for bit, alone against sharing a batch of
    /// each width, and with each beam count at the first and last place. `baseline` is the clip run
    /// alone at the checkpoint's own beam count.
    static func batchWidths(_ clip: Clip, baseline: Run, defaultBeams: Int, checks: Checks) async throws -> [String: Any] {
        func expect(_ passed: Bool, _ message: @autoclosure () -> String, line: Int = #line) { checks.expect(passed, message(), line: line) }
        await GoldenProbe.setMaxRows(12)
        var alone: [Int: Run] = [defaultBeams: baseline]
        for beams in [1, 2, 3, 4] where alone[beams] == nil {
            let first = try await run(clip, beams: beams)
            expect(first.reachedStop && first.widestBatch == beams, "beams \(beams) alone: no stop, or it did not run alone")
            alone[beams] = first
        }
        var widths: Set<Int> = [1, 2, 3, 4].reduce(into: []) { $0.insert($1) }
        for mix in mixes {
            let total = mix.reduce(0, +)
            widths.insert(total)
            let results = try await withThrowingTaskGroup(of: (Int, Run).self) { group in
                for (index, beams) in mix.enumerated() {
                    group.addTask { (index, try await GoldenProbe.transcribe(samples: clip.samples, beams: beams, candidates: recorded)) }
                    // the next piece is submitted once this one is in the host, which fixes the batch order
                    guard await GoldenProbe.waitForPieces(index + 1) else {
                        throw GoldenFailure("mix \(mix): piece \(index) never reached the engine host")
                    }
                }
                var out = [Run?](repeating: nil, count: mix.count)
                for try await (index, result) in group { out[index] = result }
                return out.map { $0! }
            }
            for (index, result) in results.enumerated() {
                let place = index == 0 ? "first" : index == mix.count - 1 ? "last" : "middle"
                expect(sameScores(alone[mix[index]]!, result), "mix \(mix): the \(place) piece (beams \(mix[index])) differs from its run alone in tokens or candidate scores")
                expect(result.widestBatch == total, "mix \(mix): the \(place) piece saw a batch of \(result.widestBatch) rows, not \(total)")
            }
        }
        expect(requiredWidths.isSubset(of: widths), "batch widths \(requiredWidths.subtracting(widths).sorted()) were never made")
        return ["mixes": mixes, "widths": widths.sorted()]
    }

    // MARK: stages

    /// max abs difference of the engine's mel, encoder output and first logits to the stored tensors
    static func stageErrors(_ clips: [Clip], directory: URL, checks: Checks) async throws -> [String: (mel: Double, encoder: Double, logits: Double)] {
        func expect(_ passed: Bool, _ message: @autoclosure () -> String, line: Int = #line) { checks.expect(passed, message(), line: line) }
        let stages = try await GoldenProbe.stages(samples: clips.map(\.samples), modelDirectory: directory)
        var out: [String: (mel: Double, encoder: Double, logits: Double)] = [:]
        func maxDifference(_ a: [Float], _ b: [Float]) -> Double {
            var worst = 0.0
            for (x, y) in zip(a, b) { worst = max(worst, Double(abs(x - y))) }
            return worst
        }
        for (clip, stage) in zip(clips, stages) {
            let mel = try GoldenProbe.floats(in: clip.tensorURL, named: "mel")
            let mask = try GoldenProbe.ints(in: clip.tensorURL, named: "mel_mask")
            let audio = try GoldenProbe.floats(in: clip.tensorURL, named: "audio")
            let logits = try GoldenProbe.floats(in: clip.tensorURL, named: "first_logits")
            expect(stage.melFrames == Int(mask.reduce(0, +)) && stage.melFrames == clip.melFrames, "\(clip.name): \(stage.melFrames) mel frames, the reference has \(clip.melFrames)")
            expect(stage.audioTokens == clip.audioTokens, "\(clip.name): \(stage.audioTokens) audio tokens, the reference has \(clip.audioTokens)")
            expect(mel.count == stage.melBins * mask.count && audio.count == stage.audio.count && logits.count == stage.firstLogits.count,
                   "\(clip.name): a stage has a different shape than the stored tensor")
            // the stored mel is [bin][frame] over the mask's frames, ours is padded to a whole chunk
            var melWorst = 0.0
            if mel.count == stage.melBins * mask.count {
                for bin in 0..<stage.melBins {
                    for frame in 0..<stage.melFrames {
                        melWorst = max(melWorst, Double(abs(mel[bin * mask.count + frame] - stage.mel[bin * stage.melPaddedFrames + frame])))
                    }
                }
            }
            let result = (mel: melWorst, encoder: maxDifference(audio, stage.audio), logits: maxDifference(logits, stage.firstLogits))
            // sanity bounds, far above what a working stage shows: they catch a broken stage, they do not measure exactness
            expect(result.mel < melBound, "\(clip.name): mel differs from the reference by \(result.mel)")
            expect(result.encoder < encoderBound, "\(clip.name): encoder output differs from the reference by \(result.encoder)")
            expect(result.logits < logitsBound, "\(clip.name): first logits differ from the reference by \(result.logits)")
            out[clip.name] = result
        }
        return out
    }

    static let melBound = 0.01
    static let encoderBound = 0.05
    static let logitsBound = 0.5

    // MARK: the tests

    @Test(.timeLimit(.minutes(60)))
    func golden() async throws {
        let checks = Checks()
        func expect(_ passed: Bool, _ message: @autoclosure () -> String = "a golden check failed", line: Int = #line) {
            checks.expect(passed, message(), line: line)
        }
        var log: [String: Any] = [:]
        log["date"] = ISO8601DateFormatter().string(from: Date())
        log["label"] = ProcessInfo.processInfo.environment["FOURIER_RUN_LABEL"] ?? "golden run"
        var thrown: Error?
        do {
            try await goldenBody(checks, &log)
        } catch {
            thrown = error
            checks.failures.append("the run stopped early: \(error)")
        }
        log["passed"] = checks.failures.isEmpty
        log["failures"] = checks.failures
        try Self.appendToLog(log)
        if let thrown { throw thrown }
    }

    func goldenBody(_ checks: Checks, _ log: inout [String: Any]) async throws {
        func expect(_ passed: Bool, _ message: @autoclosure () -> String = "a golden check failed", line: Int = #line) {
            checks.expect(passed, message(), line: line)
        }
        let (clips, manifest) = try Self.loadClips()
        let directory = Self.modelDirectory()
        try Self.requireModel(directory)
        let now = GoldenProbe.now
        let reference = Generation(manifest)

        var clock = now()
        let loaded = try await GoldenProbe.load(modelDirectory: directory)
        let firstLoad = loaded.seconds
        let beams = loaded.beams
        let generation = manifest["generation"] as! [String: Any]
        expect(beams == generation["num_beams"] as! Int)
        expect(loaded.bans == generation["banned_tokens"] as! Int)

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
            let margin = Self.scoreError(clip, first, reference)
            expect(margin.sameCandidates, "\(clip.name): the ranked candidates (all \(Self.recorded) of each step) differ from the reference")
            expect(margin.used < 1, "\(clip.name): a beam comparison uses \(margin.used) of its safety margin")
            expect(margin.finishedUsed < 1, "\(clip.name): a finished-set or stop comparison uses \(margin.finishedUsed) of its safety margin")
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
                "score_error": margin.error,
                "smallest_gap": clip.smallestGap,
                "margin_used": margin.used,
                "finished_margin_used": margin.finishedUsed,
            ]
        }
        log["clips"] = perClip

        // any batch or load: all three in one wide batch, then five pieces through a batch that
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

        // sustained throughput: the three clips, three times over, at the widest batch. the
        // results are kept and held to the same bit for bit standard.
        await GoldenProbe.setMaxRows(12)
        clock = now()
        let sustained = try await withThrowingTaskGroup(of: (String, Run).self) { group in
            for _ in 0..<3 {
                for clip in clips { group.addTask { (clip.name, try await Self.run(clip)) } }
            }
            var out: [(String, Run)] = []
            for try await pair in group { out.append(pair) }
            return out
        }
        let sustainedWall = now() - clock
        for (name, result) in sustained {
            expect(Self.sameScores(alone[name]!, result), "\(name): scores change in the sustained 12-row run")
        }
        expect(sustained.map(\.1.widestBatch).max() == 12, "the sustained run never filled 12 rows")

        // every width production makes, beam counts 1 to 4, at both edges of the batch
        clock = now()
        log["batch_widths"] = try await Self.batchWidths(clips[1], baseline: alone["medium"]!, defaultBeams: beams, checks: checks)
        log["batch_widths_seconds"] = now() - clock
        await GoldenProbe.setMaxRows(12)

        // warm load: drop the engine and map it again
        await GoldenProbe.unload()
        let warmLoad = try await GoldenProbe.load(modelDirectory: directory).seconds
        let reloaded = try await Self.run(clips[0])
        expect(Self.sameScores(alone[clips[0].name]!, reloaded), "short: scores change after a reload")

        let chip = await GoldenProbe.encoderChip()
        log["batch_throughput"] = audioSeconds / batchWall
        log["sustained_throughput"] = audioSeconds * 3 / sustainedWall
        // the weights are mapped from a warm page cache here, so this is not a cold disk read
        log["cold_load"] = firstLoad
        log["first_load_warm_cache"] = firstLoad
        log["warm_load"] = warmLoad
        log["peak_memory_bytes"] = Self.peakMemory()
        log["thermal_state"] = ProcessInfo.processInfo.thermalState.rawValue
        // the host reports the encoder's chip. the mel (accelerate), the decoder (metal) and the search (cpu) are fixed by design.
        log["placement"] = ["mel": "cpu", "encoder": chip, "decoder": "gpu", "search": "cpu"]
        if let share = await GoldenProbe.neuralShare() { log["neural_share"] = share }

        // diagnostics last, so the engine they build does not count toward peak memory
        let errors = try await Self.stageErrors(clips, directory: directory, checks: checks)
        for clip in clips {
            guard let found = errors[clip.name] else { continue }
            perClip[clip.name]?["mel_max_abs_diff"] = found.mel
            perClip[clip.name]?["encoder_max_abs_diff"] = found.encoder
            perClip[clip.name]?["first_logits_max_abs_diff"] = found.logits
        }
        log["clips"] = perClip
    }

    /// every mode, through the public api: a file, a recording, and the clip played in at real
    /// speed as if spoken. live numbers go to the same measurement log.
    @Test(.timeLimit(.minutes(30)))
    func modes() async throws {
        let checks = Checks()
        var log: [String: Any] = [
            "date": ISO8601DateFormatter().string(from: Date()),
            "label": ProcessInfo.processInfo.environment["FOURIER_RUN_LABEL"] ?? "golden run",
            "kind": "modes",
        ]
        var thrown: Error?
        do {
            log["live"] = try await modesBody(checks)
        } catch {
            thrown = error
            checks.failures.append("the run stopped early: \(error)")
        }
        log["passed"] = checks.failures.isEmpty
        log["failures"] = checks.failures
        try Self.appendToLog(log)
        if let thrown { throw thrown }
    }

    func modesBody(_ checks: Checks) async throws -> [String: [String: Any]] {
        func expect(_ passed: Bool, _ message: @autoclosure () -> String, line: Int = #line) { checks.expect(passed, message(), line: line) }
        let (clips, _) = try Self.loadClips()
        var options = FourierOptions()
        if ProcessInfo.processInfo.environment["FOURIER_MODEL_DIR"] != nil { options.model.directory = Self.modelDirectory() }
        // offline: a machine without the model must fail here, not start a download of several gigabytes
        options.model.offline = true
        let speech = Fourier(options)
        do {
            try await speech.prepare()
        } catch {
            throw GoldenFailure("the model is not ready (\(error)). prepare it first by running prepare() once with network access and offline mode off, or set FOURIER_MODEL_DIR to a prepared copy")
        }
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
        return live
    }

    /// clip shapes the reference never saw: the three golden clips are whole multiples of 160
    /// samples and longer than 0.5 s, so the frame count rounding and the 8,000 sample minimum
    /// padding of the mel are UNVERIFIED against the reference, and cannot be verified without new
    /// reference data. this only runs the short clip cut to such lengths through the engine and
    /// checks that each run ends on a stop token. it compares nothing to the reference.
    @Test(.timeLimit(.minutes(5)))
    func unverifiedClipShapesRunAndStop() async throws {
        let (clips, _) = try Self.loadClips()
        let directory = Self.modelDirectory()
        try Self.requireModel(directory)
        _ = try await GoldenProbe.load(modelDirectory: directory)
        let short = clips[0].samples
        // not multiples of 160, and 8,001 / 7,999 / 3,999 / 799 / 159 samples around and under 0.5 s
        for length in [short.count - 1, short.count - 79, 47_501, 8_001, 7_999, 3_999, 799, 159] {
            let result = try await GoldenProbe.transcribe(samples: Array(short[0..<length]), candidates: 0)
            #expect(result.reachedStop, "\(length) samples: the run did not end on a stop token (\(result.tokens.count) tokens)")
        }
    }
}
