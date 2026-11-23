# fourier, native on apple silicon: plan

a swift sdk and a swiftui mac app that run `oruk/oruk-fourier-2b` speech to text on this mac (apple m4, 32 gb), as fast as the machine allows, with the same accuracy as the original model.

**status:** plan only. nothing is built or measured yet.

**two things get built:** `FourierKit` (the sdk) and `FourierPlayground` (the mac app that uses it).

## goals, in order

1. **same accuracy as the reference.** the output must match hugging face transformers running the original weights, exactly or very nearly.
2. **fastest possible.** use the cpu, gpu and neural engine (npu) in whatever mix measures best.
3. **three modes:** `transcribe` (a file), `record` (capture, then transcribe), `listen` (live text while you speak).
4. **great defaults, everything exposed.** it works with zero config, and every knob is public.
5. **a mac app** in native swiftui with liquid glass, dark only.

## rules

- **weights:** only the native transformers checkpoint, bf16, 4.08 gb, unchanged, pinned to revision `8c2a8871f1304aa2f634da4fb95e24438c4945b9` and hash-checked.
- **native and low level:** swift and apple frameworks only (accelerate, core ml, metal). no third-party packages in the sdk or app.
- **external things are references only.** nothing is copied and nothing becomes a dependency.
- **python lives in one place:** a small harness that runs transformers to make the golden answers. it never ships.
- **nothing is set in stone.** no chip, engine or trick is chosen in advance. the numbers choose.
- **measure, don't assume.** only a trace proves where something ran.
- **license:** the weights are cc by-nc 4.0, so this is non-commercial.

## the model

confirmed from its source code.

| part | what it does |
|---|---|
| **mel** | whisper-style log-mel, 128 bins, 10 ms hop, any length |
| **conv front** | 3 convs. audio is cut into independent 1-second pieces, 13 tokens per second |
| **encoder** | 24 layers, 314m params. attention only sees its own 8-second block |
| **projector** | linear, gelu, linear |
| **decoder** | qwen3, 28 layers, 1.72b params, one token at a time with a key/value cache |
| **defaults** | beam search with 3 beams, up to 512 new tokens |

**why the blocks matter:** fixed 1-second and 8-second blocks can run in parallel, and a finished block never changes, so live mode can cache it.

## placement: an open search

every stage is built so it can run on more than one chip. the best mix is found by testing, and it can differ by task.

| stage | candidates |
|---|---|
| audio decode, resample, mel | cpu |
| conv front, encoder, projector | npu, gpu, cpu, or split between them |
| token table lookup | cpu |
| decoder | gpu, npu, cpu, or split by layers |
| output head | with the decoder, or on its own chip |
| search and text | cpu |

**what gets tried:**

- **each stage on each chip,** alone.
- **mixes:** for example npu encoder with gpu decoder, all gpu, all npu, or a decoder split across two chips.
- **overlap:** one chip encodes the next block while another decodes the current one.
- **different ways to run the same chip:** core ml, hand-written metal, accelerate on the cpu, and the private neural engine api as an experiment.

**the best mix may differ by:**

- **task:** a long file wants throughput, live mode wants the shortest delay.
- **clip length:** short clips are dominated by start-up cost, long ones by steady speed.
- **power:** plugged in, on battery, or hot.

**what ships:** a placement table built from the measurements. `automatic` reads it. `performance` and `efficiency` bias it. any stage can be pinned by hand.

**starting hunches, to be tested, not trusted:** other projects report decoders this size run faster on the gpu than the npu, and encoders run about equally on both. the npu usually wins on power.

## the engine

- **a swift converter** reads the safetensors file and builds what each chip needs, once, on this machine.
- **weights stay exact.** bf16 converts to 16-bit float without loss when values are in range. the converter counts any that are not.
- **precision is the main accuracy risk.** the fast paths compute in 16-bit. where that changes the output, that stage moves to 32-bit or to another chip.
- **no weight compression at first.** it is tried last, and kept only if accuracy holds.

## accuracy

**the bar: the exact weights, and the exact same result as transformers on all three golden clips.**

- **the reference:** transformers running the pinned checkpoint in 32-bit float with the model's own default settings (3 beams).
- **golden clips:** three fixed clips, short (about 3 s), medium (about 11 s) and long (a few minutes). harder ones are added later (noise, fast speech, silence) and never replace these.
- **the golden reference set,** saved once and then frozen:
  - the three audio files and their hashes.
  - the checkpoint revision and hash, the transformers version and every setting used.
  - per clip: mel, encoder output, first-step scores, token ids and final text.
- **every later run is compared with this set,** so a result can always be traced back to it.

| gate | passes when |
|---|---|
| weights | every converted value equals the checkpoint value |
| mel | matches the reference within a tiny tolerance |
| encoder | output matches closely |
| decoder | the same top token at every step when fed the reference tokens |
| end to end | **identical token ids and identical text on all three golden clips** |
| wider check | word error on a few hundred clips within noise of the reference |

- a change that fails a gate is not kept, however fast it is.
- **there is always one exact path.** if a fast 16-bit path cannot reproduce the reference, a slower 32-bit path that does is kept, and only exact paths can be defaults.

## speed: the hill-climb

- **numbers tracked:** audio seconds per wall second, time to first text, live draft and final delay, tokens per second, load time, memory, energy.
- **loop:** change one thing, run the golden clips, check the gates, keep it only if it is faster and still correct. log every try, kept or dropped.
- **things to try:** placement mixes, parallel encoder blocks, batched beams, greedy against 3 beams, decoder splits, cache layout, 16-bit against 32-bit per stage, shared memory between chips, weight compression last.
- **baselines to beat:** transformers on this mac, and the published 8-bit mlx build.

## folder structure

**one repo (a monorepo) with two parts:** `FourierKit` and `FourierPlayground`. nothing else, and no command line tool. it stays private while it is built and is laid out to go public later without a reshuffle.

**repo name:** `FourierKit`, the same as the package, so other apps can add it by its url. the package file sits at the repo root for the same reason.

```text
FourierKit/
├── README.md                what it is, a quick start, results
├── LICENSE                  the code license
├── NOTICE.md                the model's license and credits
├── CONTRIBUTING.md          how to build, test and send changes
├── CHANGELOG.md             what changed in each version
├── .gitignore               keeps weights, builds and private notes out
├── .github/workflows/       builds and tests on every push
├── Package.swift            the FourierKit swift package, one module
├── Sources/FourierKit/      the sdk
│   ├── Audio/               files, microphone, resampling, mel, voice activity
│   ├── Model/               converter, encoder and decoder engines, placement
│   ├── Text/                tokenizer, prompt, greedy and beam search
│   └── Speech/              the public api, the three modes, config, metrics
├── Tests/
│   ├── FourierKitTests/     unit tests and the golden gates
│   └── FourierBenchmarks/   speed and placement measurements for the hill-climb
├── FourierPlayground/       the mac app, an xcode project that depends on FourierKit
│   └── FourierPlayground/   views, state, settings, assets
├── golden/                  the frozen golden reference set
├── reference/python/        the transformers harness that makes the golden set
├── docs/
│   ├── api.md               the public api, with examples
│   ├── architecture.md      how the engine works
│   ├── performance.md       the measurement log and method
│   └── plan.md              this plan
└── scripts/                 build and run helpers
```

**ready to go public later:**

- **two licenses, kept apart.** the code gets its own license, chosen before release. the weights stay cc by-nc 4.0 and are never in the repo, and `NOTICE.md` says so clearly.
- **golden clips must be shareable.** use public-domain or openly licensed audio, or your own recordings, so the golden set can be published.
- **nothing private in the repo.** no local paths, no private repo names, no personal details. private working notes sit in an ignored folder.
- **no copied code.** everything read as a reference is credited in `NOTICE.md` as inspiration only.
- **the private neural engine api is fenced off.** it sits behind a build flag that is off by default, so the public kit builds and ships without it.
- **anyone can reproduce the numbers.** the readme gives the exact commands, and the golden gates run from a clean clone.
- **clean history from the start.** small commits, no large files, no secrets, so the history can be published as it is.

| part | what it is | who uses it |
|---|---|---|
| **`FourierKit`** | the sdk, one module with one import | any swift app |
| **`FourierPlayground`** | the swiftui mac app, for trying all three modes and every setting | people |

**what replaces a command line tool:**

- **checking:** the golden gates run with `swift test`.
- **measuring:** the benchmarks are a test target, run the same way, and they write to the measurement log.
- **converting:** the kit's `prepare()` downloads and converts the model, so no separate tool is needed.
- **trying it by hand:** `FourierPlayground`.

**why one repo:**

- the kit and the app change together while the engine is being tuned, so one commit keeps them in step.
- the golden set and the gates sit next to the code they check.
- the kit is a clean swift package at the repo root, so any app can depend on it directly.

**rules that keep it clean:**

- **the app uses only the public api.** if it needs something private, the api is missing a feature.
- **one module, four folders.** `FourierKit` is a single import. inside it, `Speech` uses `Audio`, `Model` and `Text`, and they never reach back up.
- **measuring code lives in the benchmark tests,** not in the kit.
- **the app is an xcode project,** because it needs a microphone permission, an icon and signing.
- **weights and converted models are never in the repo.** they are downloaded and built into the user's application support folder.
- **large golden files** (audio and saved tensors) are kept out of normal git history, with their hashes checked in.

## the sdk

```swift
let speech = Fourier()                                // great defaults
try await speech.prepare()                            // optional: download, convert, warm

let transcript = try await speech.transcribe(contentsOf: url)

let recorder = try await speech.record()
let recording = await recorder.stop()
let text = try await speech.transcribe(recording)

let live = try await speech.listen()                  // full schema in "listen, in detail"
for try await event in live.events { /* settled and draft text */ }
let final = try await live.finish()
```

`FourierKit` is one module. inside it:

| folder | job |
|---|---|
| `Audio` | file decoding, microphone, resampling, mel, voice activity |
| `Model` | converter, encoder and decoder engines, placement |
| `Text` | tokenizer, prompt, greedy and beam search |
| `Speech` | the three modes, configuration, progress, metrics |

**exposed config, all with defaults:**

- **compute:** placement policy, chip and engine per stage, precision per stage.
- **decoding:** greedy or beam, beam count, max tokens, repeat guard, language, context text.
- **audio:** input device, chunking, silence gate.
- **listen:** update rate, how soon text becomes final, pause length that ends a phrase.
- **model:** storage location, revision, offline mode.
- **results:** text, segments, language, and metrics (speed, time to first text, which chip ran each stage).

## the three modes

- **transcribe:** split long audio at quiet points, encode blocks in parallel, decode, join. progress and cancel.
- **record:** raw microphone capture with a live level, then the same path as transcribe.
- **listen:** live text while you speak. it has its own section below.

## listen, in detail

the model is not a streaming model, so live text is built on top of it.

**how it works:**

- finished blocks are encoded once and cached.
- every short step, only the newest audio is encoded and the current phrase is decoded again.
- text that agrees across two passes becomes **settled**. the rest is the **draft**.
- a pause ends the phrase, settles all of it and resets the window, so cost stays flat.
- stopping runs one full final pass over the audio.

**promises the api makes:**

- **settled text never changes.** it only grows.
- **draft text can change or vanish** on any update. it is always the newest guess for audio after the settled part.
- **drafts are optional.** with drafts off, or when nothing is uncertain, the draft is empty and only settled text arrives.
- **`settled + draft` is always the best full text so far.**
- **the final transcript** from `finish()` comes from the full final pass. it is the one to save.

**the schema:**

```swift
let live = try await speech.listen()            // or speech.listen(options)

for try await event in live.events {
    switch event {
    case .started:                    break     // microphone is live
    case .level(let level):           break     // input loudness, for a meter
    case .speechStarted:              break     // voice detected
    case .update(let update):         break     // new settled and draft text
    case .phraseEnded(let phrase):    break     // a pause closed a phrase
    case .warning(let warning):       break     // fell behind, input changed
    case .finished(let transcript):   break     // final pass is done
    }
}

let transcript = try await live.finish()        // stop and get the final text
live.cancel()                                   // stop and discard
```

```swift
struct LiveUpdate {
    var settled: String             // never changes, only grows
    var draft: String               // may change, may be empty
    var text: String                // settled + draft
    var newlySettled: String        // what this update added to settled
    var phrases: [LivePhrase]       // text with start and end times, each settled or draft
    var audioTime: Duration         // how much audio has been heard
    var delay: Duration             // how far the text is behind the voice
}
```

**defaults.** these are starting values, tuned in phase 7 and then frozen as the defaults.

| option | default | what it does |
|---|---|---|
| `drafts` | on | show the changing guess as well as settled text |
| `updateInterval` | about 300 ms | how often the text refreshes |
| `settleAfter` | 2 agreeing passes | how sure text must be before it settles |
| `pauseToEndPhrase` | about 600 ms | silence that closes a phrase |
| `maxPhraseLength` | about 24 s | longest phrase before a forced settle |
| `silenceGate` | automatic | ignores background noise, adapts to the room |
| `decoding` | fast for drafts, reference settings for the final pass | keeps drafts quick and finals exact |
| `language` | english | the model is english only |

**goals:** a first draft in about half a second, settled text soon after, no flicker in settled words, and no growing delay in a long session. if the machine falls behind, updates slow down before any audio is dropped, and a warning says so.

**how it is checked:**

- the three golden clips are played in as if spoken, at real speed.
- settled text must never be taken back.
- the final transcript must equal the file transcript of the same audio.
- draft delay, settle delay and how often drafts change are logged.

## the app: FourierPlayground

- **look:** native swiftui, liquid glass controls floating over a black window, dark only, quiet motion.
- **layout:** one window, three modes in a glass switcher.
- **transcribe:** drop a file, see progress, read, copy or export the text.
- **record:** one big button, a live waveform, then the transcript.
- **listen:** the centerpiece. large text, drafts in gray settling to white, a soft level meter, one key to start and stop.
- **first run:** a clear download and prepare screen, since the model is 4 gb.
- **settings:** every sdk option, grouped as above, with a reset to defaults.
- **readout:** an optional strip showing speed and which chip is running each stage, live.

## phases

each phase ends with a check that must pass before the next starts.

| # | phase | done when |
|---|---|---|
| 1 | **reference harness** | the golden reference set is saved and frozen for the three clips |
| 2 | **cpu parts:** mel, tokenizer, prompt | mel and token ids pass their gates |
| 3 | **converter and encoder** | encoder gate passes on at least two chips |
| 4 | **decoder** | decoder and end-to-end gates pass on at least two chips |
| 5 | **placement search and hill-climb** | placement table built, every try logged, gates green |
| 6 | **sdk: transcribe and record** | the api works end to end on files and the microphone |
| 7 | **sdk: listen** | settled text is never taken back, finals equal the file result, delay goals met, defaults frozen |
| 8 | **app** | all three modes usable, settings wired, first run smooth |
| 9 | **wider accuracy and energy** | word error and energy logged against the baselines |

## what could go wrong

- **16-bit math may change the text.** known trouble spots are the norm layers and long sums. fix by moving that stage to 32-bit, or by proven rescaling.
- **exact match is hard for the fast paths.** a tiny numeric difference can flip a close call between two tokens. the bar does not move: that path is fixed, or it loses its place to the exact 32-bit path.
- **cache bugs.** another project's port of this model family gave wrong output at certain cache lengths. every length is tested against the reference.
- **the npu has size and shape limits.** reports disagree on how large one piece can be, so the limit is found by testing.
- **more chips and engines mean more work.** build the simplest path per chip first, and write custom metal only where the numbers say it pays.
- **beam search costs about 3x.** it is the reference default, so it stays unless greedy proves equal on the wider check.
- **first run is slow.** a 4 gb download and a conversion. cache everything after.
- **silent fallback.** core ml can move work to another chip without saying so. trace every stage.

## not known yet

- which mix of chips wins each task on this m4.
- whether 16-bit computation reproduces the reference text exactly.
- real speed, energy and accuracy.

## references, read only

- model: https://huggingface.co/oruk/oruk-fourier-2b
- model source: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_asr and https://github.com/QwenLM/Qwen3-ASR
- vocule: `/Users/mac/code/vocule` and https://karanganesan.com/vocule/ for the three modes and the feel
- voculekit: `karanganesan/VoculeKit` on github for the swift api shape, placement and energy ideas
- compute-atlas: `/Users/mac/Desktop/untitled folder` for writing core ml from swift, tracing and measuring
- fluidaudio and fluidvoice: https://github.com/FluidInference/FluidAudio and https://github.com/altic-dev/FluidVoice
- apple: https://machinelearning.apple.com/research/neural-engine-transformers and https://machinelearning.apple.com/research/core-ml-on-device-llama
- live mode: https://arxiv.org/html/2507.10860v1 and https://arxiv.org/html/2307.14743
