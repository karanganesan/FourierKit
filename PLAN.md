# fourier, native on apple silicon: plan

**the goal: the best inference engine for `oruk/oruk-fourier-2b` on this mac (apple m4, 32 gb).**

| target | what it means | how it is measured |
|---|---|---|
| **same accuracy** | the same answer as the reference. this is never traded away | identical token ids and text on the three golden clips |
| **highest throughput** | the most audio finished under sustained load | audio seconds per wall second across a batch |
| **highest speed** | the fastest single run | audio seconds per wall second and tokens per second on one clip |
| **lowest latency** | text appears as soon as possible | time to first text, live draft delay, settle delay |

it ships as a swift sdk and a swiftui mac app.

**status:** plan only. nothing is built or measured yet.

**two things get built:** `FourierKit` (the sdk) and `FourierPlayground` (the mac app that uses it).

## goals, in order

1. **same accuracy as the reference.** the output must match hugging face transformers running the original weights, exactly, on the three golden clips.
2. **highest throughput, highest speed, lowest latency.** only exact paths compete. the cpu, gpu and neural engine (npu) are only means to that.
3. **three modes:** `transcribe` (a file), `record` (capture, then transcribe), `listen` (live text while you speak).
4. **great defaults, everything exposed.** it works with zero config, and every knob is public.
5. **a mac app** in native swiftui with liquid glass, dark only, for macos 26 or later.

## rules

- **weights:** only the native transformers checkpoint, bf16, 4.08 gb, unchanged, pinned to revision `8c2a8871f1304aa2f634da4fb95e24438c4945b9` and hash-checked.
- **native and low level:** swift and apple frameworks only (accelerate, core ml, metal). no third-party packages in the sdk or app.
- **external things are references only.** nothing is copied and nothing becomes a dependency.
- **python lives in one place:** a small harness that runs transformers to make the golden answers. it never ships.
- **nothing is set in stone.** no chip, engine or trick is chosen in advance. the numbers choose.
- **npu first.** anywhere the npu can run a stage, it gets built and tried there first. it stays wherever it passes the golden test and is fastest or ties. accuracy is never traded for it.
- **measure, don't assume.** only a trace proves where something ran.
- **license:** the engine, sdk and app are [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/), the same as the weights. all of it is non-commercial and requires attribution.
- **all lowercase.** every piece of text the app renders is lowercase, including labels, menus, errors and transcripts, and so is all markdown prose. code identifiers, paths, urls and license text keep their case.
- **lowercase is display only.** the sdk returns the model's exact text, so the golden test still compares true output. copy and export also keep the exact text.

## how we work

**the mindset: hill-climb everything.** nobody needs to know the final design up front. we need a direction, a score and a fast loop. attempts are cheap, a failed attempt is information, and only a measured gain moves us uphill.

**the loop:**

1. **pick one number** with a concrete target, such as "make encoder blocks 2x faster", never just "optimize".
2. **start from the current best,** with its score on record.
3. **make one bet,** with a stated reason it should help.
4. **try it in a fresh agent** in its own git worktree.
5. **run the golden test.** accuracy first, speed second.
6. **keep it or drop it.** keep only if it is exact and beats noise. log the result either way.
7. **climb again** from the new best.

**getting off a local peak:** when small bets stop paying, run a few big jumps in parallel, such as another engine, another chip split or a clean rewrite of one stage. a jump can look worse at first, so it gets a short budget to climb before it is judged.

**the orchestrator is fable.** it owns the goal, the scoreboard and the next bets. it spawns subagents, hands each a short brief with one target, and lands only work that passes the golden test. it stays out of bulk implementation so its context stays clean.

**subagents are spawned as needed.** use the cheapest model likely to get the job done, and move up when it fails.

| model | use it for |
|---|---|
| **haiku** | searching, reading logs, mechanical edits, running and summarizing golden runs |
| **sonnet** | routine implementation, harness work, app ui |
| **opus** | hard implementation: metal, core ml conversion, concurrency, the decoder |
| **fable** | orchestration, architecture, numeric mismatches and problems others could not solve |

**swarm rules:**

- **one agent, one question, one worktree.** parallel work never shares files.
- **timing runs go one at a time.** agents can build and explore in parallel, but a speed or energy number only counts from a quiet machine. golden runs go through one queue.
- **fresh context beats a long thread.** after about five rounds of back and forth, or two failed fixes, stop and restart with the lessons and the errors.
- **state lives in the repo, not in chat.** the scoreboard and every try go in `docs/performance.md`, so any new agent can pick up from there.
- **agents cannot move the goalposts.** no agent may edit the golden set, loosen the golden test or change how it measures to get a win. those changes need the owner.
- **show the climb.** speed and accuracy are plotted over time like a loss curve, with each stage measured end to end.
- **one test, no others.** agents do not write unit tests or new test suites. time goes into the engine and the golden test.
- **the owner stays on decisions.** goals, license, going public, and anything hard to undo come back as one clear question at a time.

**what we take from the two posts:**

- **[effective swarm riding for engineers](https://www.paradigm.xyz/writing/swarms)** by georgios konstantopoulos: short prompts with concrete targets, tune the environment, differential testing against a reference, no unit tests, no ci, measure every component, a monorepo with few dependencies, and let the swarm run while the human gets out of the way.
- **[you're vibe coding wrong](https://edwardahn.me/archive/2026/04/21/VibeCodingWrong)** by edward ahn: building is search, failed attempts are information, restart often to avoid context rot, carry lessons forward, and treat a clean one-shot as a sign the problem is understood.

**what we deliberately do not take:** deleting `docs/`, removing all comments, or skipping orchestration. this project runs long and is meant to go public, so it keeps its measurement log and a light orchestrator.

## the model

confirmed from its source code.

| part | what it does |
|---|---|
| **mel** | whisper-style log-mel, 128 bins, 10 ms hop, any length |
| **conv front** | 3 convs. audio is cut into independent 1-second pieces, 13 tokens per second |
| **encoder** | 24 layers, 314m params. attention only sees its own 8-second block |
| **projector** | linear, gelu, linear |
| **decoder** | qwen3, 28 layers, 1.72b params, one token at a time with a key/value cache |
| **defaults** | beam search with 3 beams, up to 512 new tokens, no repeated 8-grams, 32,253 banned tokens, two stop tokens |
| **input details** | the prompt prefill and short-clip padding are matched to the transformers processor |

**why the blocks matter:** fixed 1-second and 8-second blocks can run in parallel. the mel floor comes from the loudest point in the clip, so a cached block stays valid only until a louder frame raises that floor.

## placement: an open search

every stage is built so it can run on more than one chip. the best mix is found by testing, and it can differ by task.

**how a placement wins, in order:**

1. **it is exact.** it passes the golden test, or it is out.
2. **it wins the number that matters for the task.** highest throughput for batches, highest speed for a single file, lowest latency for live text.
3. **on a tie, the npu wins.** a tie means within measurement noise, and the npu usually uses less power and leaves the gpu free.

**npu first:** every stage that can run on the npu is built for it and tried there first.

| stage | npu? | try in this order | why |
|---|---|---|---|
| audio file decode | no | cpu | file parsing and branching |
| resample | try | npu, cpu | a fixed filter, but very small work |
| mel | try | npu, cpu | fixed-size math, but the log and 16-bit precision may break exactness |
| conv front | **yes** | npu, gpu, cpu | the best fit for the npu |
| encoder | **yes** | npu, gpu, cpu | fixed 8-second blocks suit it well |
| projector | **yes** | npu, gpu, cpu | runs with the encoder |
| token table lookup | **yes** | npu, gpu, cpu | runs with the decoder |
| decoder | **yes** | npu, gpu, cpu | the hardest fit: 16-bit only, size limits, and a key/value cache |
| output head | **yes** | npu, gpu, cpu | runs with the decoder |
| top-k of the scores | try | npu, gpu, cpu | can cut how much data comes back each step |
| beam search, repeat and token bans | no | cpu | branching logic over changing state |
| tokenizer and text | no | cpu | strings |

on the npu, the decoder can be cut into several sequential pieces to fit its size limits.

**what gets tried:**

- **each stage on each chip,** alone.
- **mixes:** for example npu encoder with gpu decoder, all gpu, or all npu.
- **overlap:** one chip encodes the next block while another decodes the current one.
- **different ways to run the same chip:** mpsgraph and hand-written metal on the gpu, accelerate on the cpu, and core ml on the npu. the private neural engine api waits until after phase 9.

**the best mix may differ by:**

- **task:** a long file wants throughput, live mode wants the shortest delay.
- **clip length:** short clips are dominated by start-up cost, long ones by steady speed.
- **power:** plugged in, on battery, or hot.

**what ships:** a placement table built from the measurements. `automatic` is the default and picks the fastest exact path for the task. `npu` is opt-in and uses the npu at every stage where it passes the golden test, even if another chip is faster. `efficiency` is opt-in and favors lower power. any stage can be pinned by hand.

**starting hunches, to be tested, not trusted:** other projects report decoders this size run faster on the gpu than the npu, and encoders run about equally on both. the npu usually wins on power.

**the decoder is limited by memory bandwidth.** each step reads about 3.4 gb in 16-bit or 6.9 gb in 32-bit. at the m4's 120 gb/s, that caps a single stream near 35 or 17 steps per second. this is an estimate until measured, and it shapes the design:

- **batched beams are the base design,** sharing one audio prefix cache, because a batch reads the weights once per step.
- **long files decode pieces in batches** for throughput.
- **the decoder is not split across chips,** since they share the same memory.
- **before writing converters,** a one-layer synthetic run on each chip measures the real ceiling.

## the engine

- **a swift converter** reads the safetensors file and builds what each chip needs, once, on this machine.
- **weights stay exact.** bf16 converts to 16-bit float without loss when values are in range. the converter counts any that are not.
- **the first path is exact and simple.** one 32-bit path on mpsgraph or accelerate runs end to end before anything branches from it.
- **core ml is the npu path for every npu stage.** the encoder comes first, with fixed 104-token windows and a mask, then the decoder with a stateful cache. with no packages, its model format has to be written by hand once, and every stage reuses that writer.
- **precision is the main accuracy risk.** the fast paths compute in 16-bit, and the npu is 16-bit only. where that changes the output, that stage moves to a 32-bit path on the gpu or cpu.
- **no weight compression at first.** the source checkpoint is never changed. a compressed copy is tried last, and kept only if it passes the golden test.

## accuracy

**the holy grail: one golden test, three audio clips, many dimensions.** there are no unit tests, no ci and no other test suites. if the golden test is right, the engine is right.

- **the reference:** transformers running the pinned checkpoint in 32-bit float on one recorded device and thread setup, with the model's own default settings.
- **golden clips:** three fixed clips, short (about 3 s), medium (about 11 s) and long (about 2 minutes). each must finish in one pass under the model's 512-token limit, so the reference is complete and never chunked.
- **what exact covers:** the claim is identical output on these clips. audio too long for one pass is split, which changes the model's input, so it is not claimed identical to a single reference pass.
- **the golden reference set,** saved once and then frozen:
  - the three audio files and their hashes.
  - the checkpoint revision and hash, the transformers version and every setting used.
  - per clip: mel, encoder output, scores and top-2 margin at every step, beam scores, token ids and final text.

**must pass on all three clips.** one miss and the change is out:

| dimension | passes when |
|---|---|
| **tokens** | token ids are identical to the reference |
| **text** | final text is identical to the reference |
| **stop** | stops at the same token and the same length |
| **repeatable** | repeat runs, cold and warm, give the same result |
| **every placement** | each chip mix that can ship passes on its own |
| **every mode** | file and recorded input match the reference |
| **live replay** | played in at real speed, settled text is never taken back and the final equals the file result |

**scored on all three clips.** these drive the hill-climb:

| dimension | what it records |
|---|---|
| **throughput** | audio seconds per wall second across a sustained batch of the three clips |
| **speed** | audio seconds per wall second and tokens per second on each clip |
| **latency** | time to first text, live draft delay, settle delay |
| **startup** | cold load, warm load |
| **resources** | peak memory, energy, which chip ran each stage |
| **live quality** | how often drafts change |
| **safety margin** | the closest top-2 call, which shows how near a path is to flipping a token |

**when it fails, it says where.** the saved mel, encoder output and step scores show the first stage that drifted. they are diagnostics, so only the final result decides pass or fail.

- a change that fails the golden test is not kept, however fast it is.
- **there is always one exact path.** if a fast 16-bit path cannot reproduce the reference, a slower 32-bit path that does is kept, and only exact paths can be defaults.

## speed: the hill-climb

- **numbers tracked:** audio seconds per wall second, time to first text, live draft and final delay, tokens per second, load time, memory, energy.
- **loop:** change one thing, run the golden test, keep it only if it is faster and still correct. log every try, kept or dropped.
- **things to try:** placement mixes, parallel encoder blocks, batched beams, greedy against 3 beams, cache layout, 16-bit against 32-bit per stage, shared memory between chips, weight compression last.
- **baselines to beat:** transformers on this mac. the published 8-bit mlx build and q8 gguf are tracked as a separate, lower-precision class, because they are not exact.
- **how numbers are taken:** release builds, a quiet machine in a nominal thermal state, cold and warm runs recorded apart, and the median of 5 repeats. a tie means within 3%.

## folder structure

**one repo (a monorepo) with two parts:** `FourierKit` and `FourierPlayground`. nothing else, and no command line tool. it stays private while it is built and is laid out to go public later without a reshuffle.

**repo name:** `FourierKit`, the same as the package, so other apps can add it by its url. the package file sits at the repo root for the same reason.

```text
FourierKit/
├── README.md                what it is, a quick start, results
├── LICENSE                  cc by-nc 4.0, for the code
├── NOTICE.md                the model's license and credits
├── CONTRIBUTING.md          how to build, run the golden test and send changes
├── CHANGELOG.md             what changed in each version
├── .gitignore               keeps weights, builds and private notes out
├── Package.swift            the FourierKit swift package, one module
├── Sources/FourierKit/      the sdk
│   ├── Audio/               files, microphone, resampling, mel, voice activity
│   ├── Model/               converter, encoder and decoder engines, placement
│   ├── Text/                tokenizer, prompt, greedy and beam search
│   └── Speech/              the public api, the three modes, config, metrics
├── Tests/
│   └── FourierKitTests/     the golden test, and nothing else
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

- **one license.** the code and the weights are both cc by-nc 4.0. the weights are still never in the repo, and `NOTICE.md` carries the model's license and credit.
- **golden clips must be shareable.** use public-domain or openly licensed audio, or your own recordings, so the golden set can be published.
- **nothing private in the repo.** no local paths, no private repo names, no personal details. private working notes sit in an ignored folder.
- **no copied code.** everything read as a reference is credited in `NOTICE.md` as inspiration only.
- **the private neural engine api is fenced off.** it sits behind a build flag that is off by default, so the public kit builds and ships without it.
- **anyone can reproduce the numbers.** the readme gives the exact commands, and the golden test runs from a clean clone.
- **clean history from the start.** small commits, no large files, no secrets, so the history can be published as it is.

| part | what it is | who uses it |
|---|---|---|
| **`FourierKit`** | the sdk, one module with one import | any swift app |
| **`FourierPlayground`** | the swiftui mac app, for trying all three modes and every setting | people |

**what replaces a command line tool:**

- **checking and measuring:** one command, `swift test -c release`, runs the golden test and writes every dimension to the measurement log.
- **converting:** the kit's `prepare()` downloads and converts the model, so no separate tool is needed.
- **trying it by hand:** `FourierPlayground`.

**why one repo:**

- the kit and the app change together while the engine is being tuned, so one commit keeps them in step.
- the golden set and its test sit next to the code they check.
- the kit is a clean swift package at the repo root, so any app can depend on it directly.

**rules that keep it clean:**

- **the app uses only the public api.** if it needs something private, the api is missing a feature.
- **one module, four folders.** `FourierKit` is a single import. inside it, `Speech` uses `Audio`, `Model` and `Text`, and they never reach back up.
- **measuring code lives in the golden test,** not in the kit.
- **the app is an xcode project,** because it needs a microphone permission, an icon and signing.
- **weights and converted models are never in the repo.** they are downloaded and built into the user's application support folder.
- **golden files:** the three clips live in git. the large saved tensors live on a private github release, with their hashes in git.

## the sdk

```swift
let speech = Fourier()                                // great defaults
try await speech.prepare()                            // optional: download, convert, warm

let transcript = try await speech.transcribe(contentsOf: url)

let recorder = try await speech.record()
let recording = await recorder.stop()
let text = try await speech.transcribe(recording)

let live = try await speech.listen()                  // full schema in "listen, in detail"
let ui = Task { for try await event in live.events { /* settled and draft text */ } }
let final = try await live.finish()                   // ends the event stream
try await ui.value
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

- **transcribe:** audio that fits in one pass is never split, so it matches the reference path. longer audio is cut at quiet points by one deterministic splitter, each piece runs the exact path, and the pieces are joined. progress and cancel.
- **record:** raw microphone capture with a live level, then the same path as transcribe.
- **listen:** live text while you speak. it has its own section below.

## listen, in detail

the model is not a streaming model, so live text is built on top of it.

**how it works:**

- finished blocks are encoded once and cached. they are encoded again only if a louder frame raises the phrase's mel floor.
- every short step, only the newest audio is encoded. settled tokens are prefilled, so only the unsettled tail is generated.
- text that agrees across two passes becomes **settled**. the rest is the **draft**.
- a pause ends the phrase, settles all of it and resets the window, so cost stays flat.
- **the final:** a session that fits in one pass gets one exact pass, the same as the file path. a longer session uses the same splitter as files, with exact passes running in the background so `finish()` stays short.

**promises the api makes:**

- **settled text never changes.** it only grows.
- **draft text can change or vanish** on any update. it is always the newest guess for audio after the settled part.
- **drafts are optional.** with drafts off, or when nothing is uncertain, the draft is empty and only settled text arrives.
- **`settled + draft` is always the best full text so far.**
- **the final transcript** from `finish()` comes from the exact passes. its wording can differ from the live text, and it is the one to save.

**the schema:**

```swift
let live = try await speech.listen()            // or speech.listen(options)

let ui = Task {
    for try await event in live.events {
        switch event {
        case .started:                    break // microphone is live
        case .level(let level):           break // input loudness, for a meter
        case .speechStarted:              break // voice detected
        case .update(let update):         break // new settled and draft text
        case .phraseEnded(let phrase):    break // a pause closed a phrase
        case .warning(let warning):       break // fell behind, input changed
        }
    }
}

// when the user stops:
let transcript = try await live.finish()        // stop, get the final text, end the stream
try await ui.value

// or, instead of finish():
live.cancel()                                   // stop, discard, end the stream
```

**stream rules:**

- `.level` events are coalesced to the newest value. `.update` events are never dropped.
- cancelling the task that reads events does not stop the session. only `finish()` or `cancel()` does.
- the model runs one job at a time. a `transcribe` during `listen` waits, and live work goes first.

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
- **text:** everything on screen is lowercase, transcripts included.
- **layout:** one window, three modes in a glass switcher.
- **transcribe:** drop a file, see progress, read, copy or export the text.
- **record:** one big button, a live waveform, then the transcript.
- **listen:** the centerpiece. large text, drafts in gray settling to white, a soft level meter, one key to start and stop.
- **first run:** a clear download and prepare screen, since the model is 4 gb.
- **settings:** every sdk option, grouped as above, with a reset to defaults.
- **readout:** an optional strip showing speed and which chip is running each stage, live.

## phases

the golden test is the only automated check. once it passes in phase 4, it must keep passing.

| # | phase | done when |
|---|---|---|
| 1 | **reference harness** | the golden reference set is saved and frozen for the three clips |
| 2 | **cpu parts:** mel, tokenizer, prompt | output lines up with the saved reference |
| 3 | **converter and encoder** | encoder output lines up with the saved reference |
| 4 | **decoder** | **the golden test passes end to end** on one exact path |
| 5 | **placement search and hill-climb** | fastest exact path found for each task, placement table built, every try logged, golden test passes |
| 6 | **sdk: transcribe and record** | the api works end to end on files and the microphone |
| 7 | **sdk: listen** | settled text is never taken back, finals equal the file result, delay goals met, defaults frozen |
| 8 | **app** | all three modes usable, settings wired, first run smooth |
| 9 | **final scorecard** | every golden dimension logged against the baselines |

## what could go wrong

- **16-bit math may change the text.** known trouble spots are the norm layers and long sums. fix by moving that stage to 32-bit, or by proven rescaling.
- **exact match is hard for the fast paths.** a tiny numeric difference can flip a close call between two tokens. the bar does not move: that path is fixed, or it loses its place to the exact 32-bit path.
- **cache bugs.** another project's port of this model family gave wrong output at certain cache lengths. the golden clips run through every cache length up to their full token count.
- **the npu has size and shape limits.** reports disagree on how large one piece can be, so the limit is found by testing.
- **more chips and engines mean more work.** build the simplest path per chip first, and write custom metal only where the numbers say it pays.
- **beam search costs more than greedy.** batching should keep it well under 3x. it is the reference default, so it stays unless greedy passes the golden test.
- **first run is slow and large.** a 4 gb download and a conversion, and all variants can pass 16 gb. `prepare()` checks disk space first, resumes downloads and verifies the sha256. cache everything after.
- **silent fallback.** core ml can move work to another chip without saying so. trace every stage.

## not known yet

- which mix of chips wins each task on this m4.
- whether 16-bit computation reproduces the reference text exactly.
- real speed, energy and accuracy.

## references, read only

- model: https://huggingface.co/oruk/oruk-fourier-2b
- model source: https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_asr and https://github.com/QwenLM/Qwen3-ASR
- private working references: `.private/references.md`, which stays out of git
- fluidaudio and fluidvoice: https://github.com/FluidInference/FluidAudio and https://github.com/altic-dev/FluidVoice
- apple: https://machinelearning.apple.com/research/neural-engine-transformers and https://machinelearning.apple.com/research/core-ml-on-device-llama
- live mode: https://arxiv.org/html/2507.10860v1 and https://arxiv.org/html/2307.14743
- how we work: https://www.paradigm.xyz/writing/swarms and https://edwardahn.me/archive/2026/04/21/VibeCodingWrong
