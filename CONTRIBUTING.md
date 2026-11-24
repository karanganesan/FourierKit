# contributing

this project is built by hill-climbing: one change, one score, keep it or drop it. the golden test is the only automated check. read [the plan](https://karanganesan.github.io/FourierKit/plan.html) (`rules`, `how we work`, `accuracy`) before you change the engine.

## build

```sh
swift build -c release
```

needs apple silicon, macos 26 or later and xcode 27. the package has no third-party dependencies, and none may be added.

to build the mac app:

```sh
scripts/build-app.sh
```

the mac app builds without signing settings and is signed ad hoc. a build for an iphone or ipad needs your team id:

```sh
cp FourierPlayground/Local.example.xcconfig FourierPlayground/Local.xcconfig
```

then set `DEVELOPMENT_TEAM` in `Local.xcconfig`, and `FOURIER_BUNDLE_ID` if the default bundle id is not yours to sign. git ignores `Local.xcconfig`, so the team id never lands in the repo.

## 1. prepare the model

the tests run offline and never download. without the model they fail at once with a message that names the folder and the missing files.

the engine reads the pinned checkpoint from `~/Library/Application Support/FourierKit/models/oruk--oruk-fourier-2b/<revision>/`. it needs four files there: `config.json`, `generation_config.json`, `tokenizer.json` and `model.safetensors`. the weights are 4.08 gb and their sha256 is checked.

the kit has no command line tool. prepare the model from code, once, with network access:

```swift
import FourierKit

try await Fourier().prepare()
```

run that from a scratch package, a playground or `FourierPlayground`. a folder you filled by hand works too. point the tests at it with `FOURIER_MODEL_DIR=/path/to/folder`.

## 2. get the golden tensors

the golden test compares against frozen tensors in `golden/tensors/`. they are large, so git ignores them. the clips (`golden/clips/`) and `golden/manifest.json` are in git, and the manifest holds the sha256 of each clip file (`file_sha256`), of its decoded samples (`samples_sha256`) and of each tensor file (`tensors_sha256`).

fetch them from the github release named `golden-v1`. the script needs the `gh` cli, and checks each file's hash against the manifest:

```sh
scripts/fetch-golden.sh
```

to remake them from the reference instead:

```sh
scripts/make-golden.sh
```

that runs `uv run --project reference/python python reference/python/make_golden.py`. it needs the model prepared, `uv`, and time: it runs transformers on the cpu, once in 32-bit and once in 64-bit. the options:

- `--output <dir>` writes `manifest.json` and `tensors/` to another folder. the default is the repo's `golden/`.
- the 64-bit run is a noise check. a clip is rejected if it gives other tokens or candidates in 64-bit, or if 32-bit noise uses half or more of a comparison's margin. nothing is written unless every clip passes.
- `--skip-float64` skips that check. it must come with `--unvalidated`, and then the manifest says `"validated": false`. the golden test refuses such a set, so use it only to try the harness.

remaking the set rewrites `golden/manifest.json`. the set is frozen: do not commit a changed manifest or changed clips. changes to the set need the owner.

## 3. run the golden test

```sh
./scripts/golden.sh "what this run changed"
```

it runs `swift test -c release` with your label, then rebuilds `docs/performance.html`. a full run took about eight minutes on the m4 mac. it needs the gpu to itself: close other apps, do not build or run anything else on the gpu meanwhile, and check the thermal state in the log line.

the test appends one json line to `docs/performance.jsonl` even when it fails or stops early, with `passed` and the failures. keep failed runs in the log: a try that failed is information.

what the golden test checks, on all three clips (short, medium and long):

- the clip files, their decoded samples and the tensor files match the manifest's hashes, and the set is validated: each clip passed the 64-bit check.
- token ids, final text, the stop token and the length are identical to the reference.
- the ten ranked candidates of every step are identical, including the 9th to 10th cut.
- the safety margin: in every comparison that decides an order (neighbouring candidates, the finished-set ranking, the stop heuristic), our gap is off by less than half the reference gap. the share of the margin used is logged and must be under 1.
- five runs of each clip are bit-identical.
- batching is invisible: the three clips in one batch, five pieces through a batch that fits two so slots refill, the clips three times over at 12 rows, every total width of 1, 2, 3, 4, 5, 7, 8, 10, 11 and 12 rows with beam counts from 1 to 4 at both edges of the batch, and the short clip after a reload of the engine all give bit-identical scores to the clip run alone.
- mel, encoder output and first logits stay inside loose bounds. these are diagnostics that catch a broken stage, not a measure of exactness.

a second test goes through the public api: each clip as a file and as a recording must give the reference text, and each clip played in at real speed must never take back settled text, and each live phrase must equal `transcribe` on the same audio. a third test cuts the short clip to lengths the reference never saw (not a multiple of 160 samples, or under 0.5 s) and only checks that each run ends on a stop token. those lengths are not verified against the reference.

it also records speed, tokens per second, decoder steps per second, time to first text, the time in mel, encode, prefill and decode, batch and sustained throughput, load time, peak memory, the thermal state and the configured placement. these are scores, not pass or fail.

`FOURIER_RUN_LABEL="..." swift test -c release` does the same as the script without rebuilding the page. `swift scripts/performance.swift` rebuilds the page from the log, `docs/tries.jsonl` and `docs/standing.jsonl`.

## the hill-climb

the method is the inference-optimisation skill's, credited in [NOTICE.md](NOTICE.md).

1. find the ceiling first: what the hardware allows for this step (memory bandwidth, arithmetic rate), measured, not taken from a spec sheet. a lever is worth only the gap between now and the ceiling.
2. one change per experiment, with one number it should move and a reason it should help.
3. measure control, candidate, control, interleaved, so drift on a fanless machine shows up instead of looking like a win.
4. take medians of at least six runs. a tie is within 3%.
5. keep a change only if the golden test passes and it beats noise. accuracy first, speed second.
6. log every try in `docs/tries.jsonl`, kept or dropped, one line each:

```json
{"date": "2026-11-24", "bet": "why it should help", "change": "what was changed", "result": "the numbers", "verdict": "kept", "why": "what we learned"}
```

`verdict` is `kept` or `dropped`. a dropped try saves the next person from running it again. where the climb stands, the ceilings and the next levers go in `docs/standing.jsonl`.

## the rules for changes

these come from the plan.

- the golden test must pass. a change that fails it is not kept, however fast it is.
- nobody edits the golden set, loosens the golden test or changes how it measures to get a win. those changes need the owner.
- timing runs go one at a time on a quiet machine. builds and exploring can run in parallel, but a speed number only counts from a quiet run.
- no unit tests and no other test suites. time goes into the engine and the golden test.
- no third-party packages in the sdk or the app, and no private apis.
- no dependencies on external projects. where code follows another project's behaviour step for step (for example torch's top-k order), credit it in [NOTICE.md](NOTICE.md) with its license.
- the main thread only draws the ui. all model, audio and file work runs in the background.
- the app uses only the public api. if it needs something private, the api is missing a feature.
- weights, converted models and large tensors never go in the repo.
- nothing private in the repo: no local paths, no usernames, no team ids, no private repo names.
- all lowercase. docs, comments, ui text and markdown prose are lowercase. code identifiers, paths, urls and license text keep their case.

docs under `docs/` are single-file html in the style of `docs/architecture.html`. the files in the repo root stay markdown.

## commits

small commits, one idea each. the subject is short and lowercase, starts with a verb and has no full stop, like the history:

```
keep the newest live text in view
never overwrite recents or settings after a failed read
test every batch width, the tenth candidate, and the file hashes
```

add a short lowercase body when the reason is not obvious. no large files and no secrets, because the history is meant to be published as it is.
