# changelog

## 0.0.4

- the playground's toolbar icons on iphone sit closer together.
- `scripts/archive.sh` no longer fails at the upload step.

## 0.0.3

### fixes

- the host no longer hangs when the model is unloaded while a piece is being read in. pieces being read in count as busy, so an unload waits for them.
- `keepWarm = .never` now lets go of the model after listen and between back-to-back jobs, and the next job loads it again.
- switching to another model while work is running is refused with an error, instead of swapping the engine under that work.
- a `Recorder` or `LiveSession` that is let go without `stop()`, `finish()` or `cancel()` now stops capture and ends its job. a `LiveSession` held only through its `events` stream is cancelled when the session is let go.
- the model store hashes a weights file that is already present before it can delete it, so a good file is never removed unchecked.
- a `maxTokens` the caller sets is now a cap: the piece stops there and is not split. the default, the checkpoint's 512, still splits and decodes again.
- the playground never writes over recents or settings it could not read. an unreadable file is moved aside (the last five kept), a rolling backup is written before each overwrite, one bad entry does not cost the others, and nothing is written before the first read ends.
- allocation failures in the engine throw instead of switching to another kernel, so they can no longer change bits silently. every tensor's shape and size is checked against `config.json` at load.

### additions

- `TranscriptionJob.updates` and `latestUpdate` stream the text of a file or a recording while it is transcribed: finished pieces settled, the rest a draft. `TranscriptionUpdate` holds `settled`, `draft` and `text`.
- `TranscriptionJob.progress` moves inside a piece too: the encoder, the prefill, then every decoder step against an estimate of its length.
- `Recorder.drafts` and `latestDraft` show draft text while recording, from quick greedy passes over the audio so far. `FourierOptions.recordDrafts` turns them off. the transcript still comes from one exact pass after stop.
- live drafts grow token by token: an update every 50 to 70 ms at the median in one try, where one used to arrive only when a pass ended.
- `TranscriptPiece.isExact`: false when a live phrase's exact pass failed twice and its settled live text stands in. a failed exact pass for a live phrase is now retried once.
- `ComputeOptions.resolvedPlacement` shows the chip each stage runs on for a set of options. every `compute.policy` resolves to the gpu for the encoder and decoder. only a pin in `chips` moves a stage.
- `TranscriptMetrics.realtimeSpeed` and `timeToFirstText` are measured from when the engine was ready and the audio read, so a cold first call's prepare is not counted. `elapsed` is the whole call.
- `FourierMetrics.chipBusy` has no neural engine key, rather than 0. gpu busy is the wall time from commit to completion of its work. `placement` in the state and the metrics is the configured placement, not a measurement.
- `FourierMetrics.device` reports what the engine read about the device and the limits it derived.
- `fourierFollowsNewest()` in `FourierKitUI` keeps the end of a scroll view in view while text streams in, and lets go when the reader scrolls up. live text in the playground follows the newest words.
- the sdk and the playground build for ios and ipados as well as macos. every limit is derived from the device at load: cache slots, batch rows, the encoder's conv group, scratch memory and the piece length for long audio.
- the playground: deleted transcripts go to recently deleted for 30 days and can be restored. a stop bar follows a running capture on every other screen. after a failed transcription of a recording, the audio is kept and can be retried or saved. automated launches (any `-fourier` argument) use their own recents file. in full screen the toolbar steps aside. the modes are listed as listen, record, transcribe.
- the playground is named fourierkit on screen and its bundle id is `com.karanganesan.fourierkit`. `scripts/archive.sh ios|macos [upload]` archives it for testflight and the app store. the mac archive is sandboxed, which the mac app store requires, so it keeps its model and transcripts in its own container. the app carries a privacy manifest.
- a device build reads the team id from `FourierPlayground/Local.xcconfig`, which git ignores. `Local.example.xcconfig` shows what to fill in.

### engine

- a second bandwidth kernel computes three rows per thread, bit-identical to the first. twelve medium clips at once went from about 7x to 12x realtime in one try.
- the encoder's result no longer depends on the conv group size: the projection out of the conv front runs once over every chunk of the clip. the medium and long clips' encoder bits changed with this. tokens and candidates did not.
- a decoder step never uses apple's matrix kernel, whatever the batch width allowed.
- `prefill(pieces:)` in the engine reads several pieces in one prefill, bit-identical to reading each alone. it cut gpu time by 27 to 39 percent for groups of short pieces in one try. the scheduler does not use it: a try that read waiting pieces together gave no gain for twelve clips at once.

### golden test

- the ten ranked candidates of every step are compared, so the cut between the 9th and 10th is covered, and the margin now also covers the finished-set ranking and the stop heuristic.
- every total batch width of 1, 2, 3, 4, 5, 7, 8, 10, 11 and 12 rows, with beam counts from 1 to 4 at both edges of the batch, is checked bit for bit against the clip alone.
- the hashes of the clip files, the decoded samples and the tensor files are checked, and the golden set must be validated: each clip passed the 64-bit cross-check. `make_golden.py` writes into a staging folder and only moves the set into place when every clip passed, and takes `--output`, `--skip-float64` and `--unvalidated`.
- a run that fails or stops early is still logged.
- clip lengths the reference never saw (not a multiple of 160 samples, or under 0.5 s) are run and must end on a stop token. they are not compared to the reference.

### measured elsewhere

- on an iphone 17 pro the three golden clips gave the reference text at 2.9x, 3.9x and 4.3x realtime from fresh launches. this was before the engine changes above, and checked by text only, not by the golden test.

### docs

- every doc brought in line with the code: the exactness claim says what the golden test compares and what it does not cover, the api page matches the public declarations, and the plan says what changed from it.

## 0.0.2

- the app is one glass surface that follows light and dark, with a floating glass bar for live controls, glass panels for loading and progress, three type sizes and three text levels.
- a code pane shows the swift for the current screen: adding the package, options that differ from the defaults, and the calls.
- the sidebar draws one quiet selection, selects every recent with command a or control a, and asks before deleting several.
- what scrolls fades out at the top and bottom of every pane.
- live text no longer flashes on updates, and a live transcript's speed is the engine's, not the session's.
- prefill attention runs on the matrix kernel: the long clip's prefill went from 3.1 s to 2.3 s.

## 0.0.1

nothing is released yet. this is what exists in the repo.

### engine

- audio: file reading and resampling to 16 khz mono, 128-bin log-mel on the cpu in 32-bit, a lock-free ring buffer, microphone capture with voice processing, an energy speech detector and a splitter that cuts long audio at quiet points.
- model: a safetensors reader, the encoder and the qwen3 decoder on the gpu with metal, beam search with batched beams, a shared audio prefix cache and slot refill, and a core ml encoder path whose model files are written from swift. the gpu encoder is the default.
- text: tokenizer, prompt, beam and greedy search, token bans, the repeat guard and the transcript text fix.
- the exact path is the default. placement in the first logged run: mel and search on the cpu, encoder and decoder on the gpu.

### sdk

- `Fourier` with `prepare()`, `warm()` and `cool()`, and `prepareProgress` from download to warm-up.
- `transcribe` for a file, samples and a recording, with `TranscriptionJob` for progress and cancel. audio over 120 s is split at quiet points and joined.
- `record()` with a live level, and `listen()` with settled and draft text, phrases, warnings and a final pass per phrase.
- `FourierOptions` with a default for every option, and replaceable parts: `AudioSource`, `NoiseProcessor`, `SpeechDetecting` and `ModelStore`.
- `FourierState`, `FourierMetrics` and microphone state, permission, device and level streams.
- `FourierError` names the cause and says what to do.
- the model is downloaded from the pinned revision with resume and a sha256 check.

### ui and app

- `FourierKitUI` has the bundled allo variable font and its license. its views cover prepare, permission, the mic button, the level meter, live text, live stats, and one drop-in live view.
- the ui is one glass surface that follows light and dark, with four type sizes, three text levels and a floating glass bar for the live controls. `FourierControlBar` and `fourierGlassPanel()` are public.
- `FourierPlayground` is a swiftui mac app with transcribe, record and listen, recent transcripts, a live stats inspector and a settings window for every sdk option. it uses only the public api.

### golden test

- one test, three clips, checked against frozen transformers output: token ids, text, stop, ranked candidates, the safety margin, bit-identical repeats, batch and slot refill, and reload.
- each run appends a line to `docs/performance.jsonl`.
- `golden/` holds the clips, the manifest and the hashes of the tensors. `reference/python/` remakes the set.

### docs

- `docs/plan.html`, `docs/architecture.html`, `docs/api.html` and `docs/performance.html`.
- `scripts/` has helpers for the golden test, the golden tensors, the performance page and the app build.
- `README.md`, `CONTRIBUTING.md` and `NOTICE.md`.
