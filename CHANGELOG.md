# changelog

## unreleased

- the sdk and the playground build for ios and ipados as well as macos. every limit is derived from the device at load: memory headroom, core counts, the gpu's buffer and working-set sizes, the neural engine.
- a second bandwidth kernel computes three rows per thread, bit-identical to the first. twelve medium clips at once go from about 7x to 12x realtime.
- live drafts grow token by token: an update every 50 to 70 ms at the median, where one used to arrive only when a pass ended.
- in full screen the toolbar steps aside, so no bar is drawn behind it.

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
