# fourier, native on apple silicon

speech to text for `oruk/oruk-fourier-2b`, written in swift on apple frameworks only (accelerate, core ml, metal). no third-party packages.

this repo has two parts.

- `FourierKit` is the sdk. one import, three modes, every option exposed and defaulted. `FourierKitUI` is an optional second import with default swiftui views.
- `FourierPlayground` is a mac app that uses the sdk to try all three modes and every setting.

status: the engine, the sdk, the default ui, the app and the golden test all work on the mac they were built on. the app's real microphone flow and the first-run download of the full checkpoint have not been exercised end to end. see [CHANGELOG.md](CHANGELOG.md).

## what exact means here

on the three golden clips, `FourierKit` gives the same token ids and the same final text as hugging face transformers running the original weights. the reference is pinned to one transformers commit, runs on the cpu in 32-bit float with a batch of one, and is frozen in `golden/`.

what this does not claim:

- audio over 120 s is split at quiet points and the pieces are joined. that changes the model's input, so it is not claimed identical to one reference pass.
- custom audio processing, and any decoding option you change from its default, leave the exact path.
- file decoding and resampling are outside the contract. the contract starts at 16 khz mono float32 samples.

the golden test has two parts. one drives the engine and checks tokens, text, the stop token, repeat runs, that scores are bit-identical when clips share a batch or refill a slot, and that every beam comparison keeps its order with room to spare. the other goes through the sdk: a file, a recording, and each clip played in at real speed, where settled text must never be taken back and the final must equal `transcribe` on the same split points. see [CONTRIBUTING.md](CONTRIBUTING.md).

## requirements

- a mac with apple silicon
- macos 26 or later, xcode 27, swift 6
- disk and network for the first run. `prepare()` downloads the 4.08 gb checkpoint pinned to revision `8c2a8871f1304aa2f634da4fb95e24438c4945b9` and checks its sha256. the download resumes if it stops. nothing is bundled, and the weights are not in this repo.

## quick start

add the package by its url in xcode (file, add package dependencies), or in `Package.swift`:

```swift
.package(url: "https://github.com/karanganesan/FourierKit.git", branch: "main")
```

then add the `FourierKit` product to your target, and `FourierKitUI` if you want the default views.

```swift
import FourierKit

let speech = Fourier()
try await speech.prepare()          // optional: every mode prepares on first use

let transcript = try await speech.transcribe(contentsOf: url)
print(transcript.text)
```

`prepare()` reports one value from 0 to 1 that covers download to warm-up:

```swift
let preparing = Task { try await speech.prepare() }
for await progress in speech.prepareProgress {
    print(progress.stage.name, progress.fraction)
}
try await preparing.value
```

## the three modes

transcribe a file, samples or a recording. audio up to 120 s runs in one pass. longer audio is cut at quiet points and joined. `transcriptionJob(contentsOf:)` gives a handle with progress and cancel.

```swift
let job = speech.transcriptionJob(contentsOf: url)
Task { for await fraction in job.progress { print(fraction) } }
let transcript = try await job.result
```

record raw microphone audio with a live level, then transcribe it the same way.

```swift
let recorder = try await speech.record()
// ... later
let recording = await recorder.stop()
let transcript = try await speech.transcribe(recording)
```

listen for live text while you speak. `.update` events carry settled text, which only grows, and a draft, which can change. `finish()` returns the exact transcript.

```swift
let live = try await speech.listen()
let reading = Task {
    for try await event in live.events {
        if case .update(let update) = event { print(update.settled, update.draft) }
    }
}
let final = try await live.finish()
try await reading.value
```

the full api, with every option and its default, is in [docs/api.html](docs/api.html).

## the golden test

```sh
swift test -c release
```

the test needs the model prepared on this mac and the golden tensors in `golden/tensors/`. both steps are in [CONTRIBUTING.md](CONTRIBUTING.md). `scripts/golden.sh` runs the test and rebuilds the performance page.

## results

every golden run appends one line to `docs/performance.jsonl`. [docs/performance.html](docs/performance.html) is built from that log and holds the scoreboard, the baselines, the climb over runs and the method. the numbers live there, not here.

the newest passing line today is labelled "plateau: gpu encoder and decoder, quiet machine", measured warm in thermal state 1 (fair) on a fanless m4 macbook air with 32 gb.

| clip | audio | fourierkit, x realtime | transformers on the same mac, x realtime | time to first text |
|---|---|---|---|---|
| short | 3.7 s | 4.41 | 1.10 | 0.28 s |
| medium | 11.2 s | 6.29 | 1.18 | 0.44 s |
| long | 116.5 s | 6.05 | 0.81 | 4.15 s |

- three clips in one batch ran at 5.5x realtime in total, and nine pieces sustained ran at 6.1x.
- the decoder ran 19.5 to 21.7 steps per second. reading the weights alone caps it near 26.
- played in live at real speed, first text came after 0.7 to 1.1 s and the median draft delay was 0.4 to 0.8 s. the plan's goal of a first draft in about half a second is not met yet.
- the transformers column is the cpu reference run recorded in `golden/manifest.json` (`reference_seconds`).

what was tried, kept and dropped is in the tries section of the performance page. the encoder also runs on the neural engine, 3x faster than on the gpu, but its 16-bit math uses 35 to 44 times the safety margin, so it is not a default and is outside the exactness claim.

## docs

- [docs/plan.html](docs/plan.html): goals, rules, how we work, phases
- [docs/architecture.html](docs/architecture.html): which chip runs each step and how work is packed
- [docs/api.html](docs/api.html): the public api with examples
- [docs/performance.html](docs/performance.html): the measurement log, baselines and method

## license

the code is licensed under [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/). the full text is in [LICENSE](LICENSE). commercial use is not allowed under it.

the model is not part of this repo and has its own license, also cc by-nc 4.0, with terms of its own for hosting and commercial use. credits and details are in [NOTICE.md](NOTICE.md).
