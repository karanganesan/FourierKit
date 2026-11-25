# fourier, native on apple silicon

speech to text for `oruk/oruk-fourier-2b`, written in swift on apple frameworks only (accelerate, core ml, metal). no third-party packages.

website: [karanganesan.com/fourierkit](https://karanganesan.com/fourierkit/), with recordings of the app, the speed and accuracy figures and how it packs inference.

this repo has two parts.

- `FourierKit` is the sdk. one import, three modes, every option exposed and defaulted. `FourierKitUI` is an optional second import with default swiftui views.
- `FourierPlayground` is a swiftui app for mac, iphone and ipad that uses the sdk to try all three modes and every setting.

status: the engine, the sdk, the default ui, the app and the golden test work on the m4 mac they were built on. on an iphone 17 pro the three golden clips give the same token ids and candidate scores as the mac and the reference. what has not been checked is listed under [not verified](#not-verified). see [CHANGELOG.md](CHANGELOG.md).

## what exact means here

on the three golden clips, `FourierKit` gives the same token ids, the same final text and the same stop as hugging face transformers running the original weights. the reference is pinned to one transformers commit, runs on the cpu in 32-bit float with a batch of one, and is frozen in `golden/`.

the golden test compares, on each clip:

- token ids, final text, the stop token and the length.
- the ten ranked candidates of every step, including the gap between the 9th and 10th, which decides what beam search keeps.
- a safety margin: in every comparison that decides a ranking, the error in our gap must stay under half the reference gap. this covers neighbouring candidates, the ranking of finished hypotheses and the stop heuristic.
- bit identity: repeat runs, a piece alone against the same piece sharing a batch at total widths 1, 2, 3, 4, 5, 7, 8, 10, 11 and 12 rows with mixed beam counts from 1 to 4, a batch that refills its slots, a sustained 12-row run, and a reload of the engine.
- the hashes of the clip files, of the decoded samples and of the stored tensors.
- that the golden set was validated: each clip passed a 64-bit cross-check in the reference (same tokens, same candidates, under half the margin used).
- mel, encoder output and first logits, as diagnostics with loose bounds that catch a broken stage.

a second test runs the public calls: a file and a recording must give the reference text, and each clip played in at real speed must never take back settled text, and each live phrase must equal `transcribe` on the same audio.

where the bits are fixed: every kernel the decoder step uses computes each value in a fixed order that depends only on that row, so a row's scores are the same at any batch width. the prefill and the encoder use apple's matrix kernel, whose order is apple's. measured on the m4, its result for a row stops depending on the row count once the count passes a boundary (10 to 90 rows, by matrix shape), so those passes are a pure function of the piece. the encoder no longer depends on how many 1-second chunks share a command buffer. the full note is in the header of `Sources/FourierKit/Model/Kernels.swift`.

what this does not claim:

- the golden set is three english clips cut from one recording. all three are whole multiples of 160 samples and longer than 0.5 s. other lengths run and stop on a stop token, but that is not checked against the reference.
- batches wider than 12 rows are not run.
- audio over 120 s is split at quiet points and the pieces are joined. that changes the model's input, so it is not claimed identical to one reference pass.
- custom audio processing, a pinned chip and any decoding option you change from its default leave the exact path.
- file decoding and resampling are outside the contract. the contract starts at 16 khz mono float32 samples.

## requirements

- apple silicon: a mac on macos 26 or later, or an iphone or ipad on ios or ipados 26 or later. the engine reads the device when it loads and derives every limit from it: cache slots, batch rows, the encoder's conv group and the scratch memory. a device without room for the weights and one two-minute piece fails to load with an error that says so. tested on an m4 mac with 32 gb and an iphone 17 pro.
- xcode 27, swift 6.2
- disk and network for the first run. `prepare()` downloads the 4.08 gb checkpoint pinned to revision `8c2a8871f1304aa2f634da4fb95e24438c4945b9` and checks its sha256. the download resumes if it stops. nothing is bundled, and the weights are not in this repo.

## quick start

add the package by its url in xcode (file, add package dependencies), or in `Package.swift`:

```swift
.package(url: "https://github.com/karanganesan/FourierKit.git", from: "0.0.4")
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

all three show text while they work. settled text never changes, a draft can, and the transcript each one returns at the end is the one to keep.

listen for live text while you speak. `.update` events carry settled text, which only grows, and a draft, which can change. `finish()` returns the exact transcript. keep the session while it runs: a session that is let go is cancelled, even if a task is still reading its events.

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

record raw microphone audio with a live level and draft text, then transcribe it in one exact pass. a recorder that is let go stops capture.

```swift
let recorder = try await speech.record()
Task { for await update in recorder.drafts { print(update.text) } }
// ... later
let recording = await recorder.stop()
let transcript = try await speech.transcribe(recording)
```

transcribe a file, samples or a recording. audio up to 120 s runs in one pass. longer audio is cut at quiet points and joined. `transcriptionJob(contentsOf:)` gives a handle with progress, text so far and cancel.

```swift
let job = speech.transcriptionJob(contentsOf: url)
Task { for await fraction in job.progress { print(fraction) } }
Task { for await update in job.updates { print(update.settled, update.draft) } }
let transcript = try await job.result
```

the full api, with every option and its default, is in [the api page](https://karanganesan.github.io/FourierKit/api.html).

## the golden test

```sh
./scripts/golden.sh "what this run changed"
```

the test needs the model prepared and the golden tensors in `golden/tensors/`, which are not in git. `scripts/fetch-golden.sh` downloads them from the `golden-v1` release and checks their hashes. both steps are in [CONTRIBUTING.md](CONTRIBUTING.md).

## results

every golden run appends one line to `docs/performance.jsonl`, passed or failed. [the performance page](https://karanganesan.github.io/FourierKit/performance.html) is built from that log and holds the scoreboard, the baselines, the climb over runs, every try and the method.

the newest passing line is labelled "0.0.4 as released: check before the repo goes public" (2026-11-24, 23:50 utc). it ran on a fanless m4 macbook air with 32 gb, in thermal state 1 (fair). per-clip speed is the median of four warm runs after one cold run.

| clip | audio | fourierkit, x realtime | transformers, x realtime | time to first text |
|---|---|---|---|---|
| short | 3.7 s | 3.92 | 1.10 | 0.34 s |
| medium | 11.2 s | 5.66 | 1.18 | 0.50 s |
| long | 116.5 s | 6.53 | 0.81 | 3.12 s |

- in the same run: the three clips together in one batch ran at 6.88x realtime in total, and the three clips three times over (sustained) at 7.73x. peak memory was 3.74 gb.
- the decoder ran 19.4 to 20.4 steps per second. reading the weights alone caps it near 26.
- runs vary a lot. across the nine passing runs from 05:21 to 23:50 utc that day, all in thermal state 1, single-clip speed ranged 3.35x to 4.34x (short), 4.69x to 6.36x (medium) and 5.66x to 6.73x (long), and sustained throughput 4.31x to 11.33x. the code changed between those runs and some were labelled as sharing the machine, so the spread mixes both. the newest run is the same engine as the 09:10 run, which gave 4.34x, 6.36x and 6.73x and 11.33x sustained, so the gap between them is heat and what else the machine was doing. no controlled hot against cool comparison has been run.
- many files at once: twelve 11-second clips at 12 rows ran at 11.0x to 12.7x realtime in total over six runs in one try (see `docs/tries.jsonl`). clips of mixed length gain less.
- played in live at real speed (the newest modes line), first text came after 0.79 s (short), 0.78 s (medium) and 0.51 s (long). in a separate try, text then grew token by token, with a new update every 50 to 70 ms at the median.
- the transformers column is one instrumented cpu run per clip recorded in `golden/manifest.json` (`reference_seconds`), with extra top-k, scores and logits, and the first call's warm-up. it is not a plain warm run.
- an iphone 17 pro (a19 pro, 12 gb) ran the three clips at 2.45x, 3.37x and 3.51x realtime (short, medium, long) over six rounds back to back, on the 0.0.4 engine. the hashes of the token ids and of the ten candidate scores per step equal the mac's and the reference. this came from a benchmark build of the app, not from the golden test, which has not run on a phone. the phone reaches serious heat after about four minutes. see [HILLCLIMB.md](HILLCLIMB.md).

the encoder also has a neural engine path, about 3x faster than the gpu on two minutes of audio, but its 16-bit math uses 35 to 44 times the safety margin. it runs only if you pin it, and it is outside the exactness claim.

## not verified

- on an iphone or ipad: the microphone, listen and record, the first-run 4 gb download, and the golden test itself. its bit-identity checks across batch widths and its margin checks have not run there. token ids and candidate scores on the three clips have been compared, as above.
- anywhere: voice processing on a real input with noise, voiceover, energy use, and an instruments trace of where work ran.

## docs

the docs are single-file html in `docs/`, served by github pages.

- [website](https://karanganesan.com/fourierkit/): the kit's page, for a first look
- [plan](https://karanganesan.github.io/FourierKit/plan.html): goals, rules, how we work, phases, and what changed from the plan
- [architecture](https://karanganesan.github.io/FourierKit/architecture.html): which chip runs each step, where bits are fixed and how work is packed
- [api](https://karanganesan.github.io/FourierKit/api.html): the public api with examples
- [performance](https://karanganesan.github.io/FourierKit/performance.html): the measurement log, baselines, tries and method

## license

the code is licensed under [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/). the full text is in [LICENSE](LICENSE). commercial use is not allowed under it.

the model is not part of this repo and has its own license, also cc by-nc 4.0, with terms of its own for hosting and commercial use. credits for the model, the font, the golden audio and the code this project follows are in [NOTICE.md](NOTICE.md).
