# contributing

this project is built by hill-climbing: one change, one score, keep it or drop it. the golden test is the only automated check. read [docs/plan.html](docs/plan.html) (`rules`, `how we work`, `accuracy`) before you change the engine.

## build

```sh
swift build -c release
```

needs a mac with apple silicon, macos 26 or later and xcode 27. the package has no third-party dependencies, and none may be added.

to build the mac app (once `FourierPlayground/` exists):

```sh
scripts/build-app.sh
```

## prepare the model

the engine reads the pinned checkpoint from `~/Library/Application Support/FourierKit/models/oruk--oruk-fourier-2b/<revision>/`. it needs four files there: `config.json`, `generation_config.json`, `tokenizer.json` and `model.safetensors`. the weights are 4.08 gb and their sha256 is checked.

the kit has no command line tool. prepare the model from code, once:

```swift
import FourierKit

try await Fourier().prepare()
```

run that from a scratch package, a playground or `FourierPlayground`. a folder you filled by hand works too. point the test at it with `FOURIER_MODEL_DIR=/path/to/folder`.

## the golden tensors

the golden test compares against frozen tensors in `golden/tensors/`. they are large, so they are not in git. the clips (`golden/clips/`) and `golden/manifest.json` are in git, and the manifest holds each tensor file's sha256 as `tensors_sha256`. two ways to get them:

1. fetch them from the github release named `golden-v1`, once the owner has published it:

   ```sh
   scripts/fetch-golden.sh
   ```

   it needs the `gh` cli, downloads the three `.safetensors` files and fails if any hash differs from the manifest. if the release does not exist yet, use the second way.

2. remake them with the python reference. this needs the model prepared first, `uv`, and several minutes:

   ```sh
   scripts/make-golden.sh
   ```

   that runs `uv run --project reference/python python reference/python/make_golden.py`. it also rewrites `golden/manifest.json`. the golden set is frozen, so do not commit a changed manifest or changed clips. changes to the set need the owner.

## run the golden test

```sh
swift test -c release
```

or, to name the run and rebuild the performance page in one step:

```sh
scripts/golden.sh "what this run changed"
```

use a release build, a quiet machine and one timing run at a time. close other apps first, and check the thermal state in the log line. the test takes a while: each clip runs five times, then the batch and reload checks.

what the test checks, on all three clips (short, medium and long):

- token ids are identical to the reference.
- the final text is identical to the reference.
- it stops at the same token and the same length.
- repeat runs give bit-identical scores.
- the ranked candidates at every step match the reference.
- the safety margin: the error in every beam comparison stays under half the reference gap. the test records how much of that margin is used, and it must be under 1.
- batching is invisible: the three clips run together in one wide batch, then through a batch that only fits two so slots are refilled, and every score stays bit-identical to the run alone.
- scores are bit-identical after the engine is unloaded and loaded again.

it also records speed, tokens per second, decoder steps per second, time to first text, the time in mel, encode, prefill and decode, batch and sustained throughput, cold and warm load, peak memory and where each stage ran. these are scores, not pass or fail.

the test needs the model and the tensors. if the tensors are missing it fails with a message that names the file.

## the measurement log

each run of the golden test appends one json line to `docs/performance.jsonl`. the line holds the date, a label, every score above, whether the run passed and the failures if it did not. keep failed runs in the log: a try that failed is information.

name the run with the `FOURIER_RUN_LABEL` environment variable, with one short line about what changed:

```sh
FOURIER_RUN_LABEL="wide multiplies on apple's kernel" swift test -c release
```

`scripts/golden.sh` does the same with its first argument.

`swift scripts/performance.swift` rebuilds `docs/performance.html` from the log: the scoreboard for the newest passing run, the baselines, the climb over runs and the method. it also renders `docs/tries.jsonl` if that file exists. each line is one experiment:

```json
{"date": "2026-11-24", "bet": "why it should help", "change": "what was changed", "result": "the numbers", "verdict": "kept", "why": "what we learned"}
```

`verdict` is `kept` or `dropped`. log every try, kept or dropped.

## the rules for changes

these come from the plan.

- one change per experiment, with one number it should move and a reason it should help.
- the golden test must pass. a change that fails it is not kept, however fast it is. accuracy first, speed second.
- keep a change only if it is exact and beats noise. a tie is within 3%.
- nobody edits the golden set, loosens the golden test or changes how it measures to get a win. those changes need the owner.
- log every try, kept or dropped.
- timing runs go one at a time on a quiet machine in a nominal thermal state. builds and exploring can run in parallel, but a speed number only counts from a quiet run.
- no unit tests and no other test suites. time goes into the engine and the golden test.
- no third-party packages in the sdk or the app, and no private apis.
- external projects are references only. nothing is copied and nothing becomes a dependency. credit what you read in [NOTICE.md](NOTICE.md).
- the main thread only draws the ui. all model, audio and file work runs in the background.
- the app uses only the public api. if it needs something private, the api is missing a feature.
- weights, converted models and large tensors never go in the repo.
- nothing private in the repo: no local paths, no usernames, no private repo names.
- all lowercase. docs, comments, ui text and markdown prose are lowercase. code identifiers, paths, urls and license text keep their case.

docs under `docs/` are single-file html in the style of `docs/architecture.html`. the files in the repo root stay markdown.

## commits

small commits, one idea each. the subject is short and lowercase, starts with a verb and has no full stop, like the history:

```
bundle allo variable for the default ui
specify native app design and live stats
tighten exactness contract and sdk design
```

add a short lowercase body when the reason is not obvious. no large files and no secrets, because the history is meant to be published as it is.
