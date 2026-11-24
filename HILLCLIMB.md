# hill-climb efforts

where the speed work on the engine stands, written when the second climb was stopped on 2026-11-24. the rule throughout: same weights, same outputs, bit for bit. a change that moves one token id or one candidate score is dropped, whatever it gains.

the full ledgers are `docs/tries.jsonl` (every try, kept or dropped), `docs/performance.jsonl` (every golden run), `docs/standing.jsonl` (ceilings and where things stand) and the page built from them, `docs/performance.html`.

## the short version

- nothing from the second climb is merged into `main`. the engine in 0.0.4 is the engine that passed the last golden run.
- one real gain was found and is not shippable yet: reading waiting pieces in as a group lifts twelve files at once from 13.3x to 14.5x realtime, but it delays live text by about 2 s when someone listens during a batch.
- single-clip decode has little left. it runs at about 90 percent of the memory-bandwidth ceiling, and every cpu-side lever measured under half a percent of a step.
- three things were learned that are not speed-ups: the iphone is bit-identical to the mac, heat costs 12 to 16 percent and the system does not report it, and there is now a protocol that can tell a 1.5 percent change from noise.

## where the engine stands

measured on an apple m4 with 32 gb, cool, each run after 20 s of idle:

| clip | length | speed | steps per second | first text |
|---|---|---|---|---|
| short | 3.7 s | 4.37x | 21.4 | 0.32 s |
| medium | 11.2 s | 6.34x | 21.5 | 0.45 s |
| long | 116.5 s | 6.73x | 19.8 | 3.0 s |

- twelve 11-second clips at once: 13.3x. the three golden clips together: 7.2x. three clips three times over: 11.9x.
- a decoder step at a 1,533-token context is about 52 ms: 36.7 ms reading the layer matrices, 7.6 ms the output head, 6.8 ms attention.
- reading the weights caps a single clip near 26 steps per second with a short context and near 20 with a long one.

on an iphone 17 pro (a19 pro, 12 gb), six rounds back to back:

| clip | speed | steps per second | first text |
|---|---|---|---|
| short | 2.45x | 12.2 | 0.52 s |
| medium | 3.37x | 13.0 | 1.02 s |
| long | 3.51x | 10.4 | 4.1 s cool |

- token ids, all ten candidates per step and the score hashes equal the mac's and the reference.
- the phone derives 6 batch rows, 4,864 cache slots, conv group 8 and 30 s pieces. peak memory 2.0 gb.
- it reaches serious heat after about 4 minutes. decode holds within 3 percent when hot, but the encoder plus prefill doubles. 4.5 minutes of idle restores it.

## how to measure

heat is the main source of error, so the protocol is built around it.

- heat costs 12 to 16 percent of speed after 7 minutes of load, and up to 26 percent on a machine that was already warm. throttling starts after about 3 minutes.
- `ProcessInfo.thermalState` read "fair" from a 2 percent loss to a 26 percent loss, so it cannot be used to detect throttling.
- 45 s of idle restores full speed. 20 s of idle before each run holds it.

the protocol:

1. idle 60 s, then 6 runs with 20 s of idle before each.
2. control, candidate, control. the two controls must agree within 1 percent, or the session is thrown away.
3. a difference is real above 1.5 percent on speed and steps per second for the long and medium clips, 3 percent for the short clip and for first text on long and medium, 8 percent for first text on the short clip.
4. one gpu job at a time on a machine. builds count, because they load the memory bus.
5. screen first: two runs against two controls. only a candidate that beats the threshold gets the full protocol.
6. prove identity with hashes of the token ids and of the ten candidate scores per step, on every clip, before timing anything.

the phone can screen a change and prove identity on a second chip. it is a different gpu with six rows, so a phone number never stands in for a mac number.

## second climb, 2026-11-24: what was tried

six agents ran in parallel, each in its own worktree, sharing one lock for the mac and one for the phone. it was stopped before every lever had been measured.

### finished

| lever | result | verdict |
|---|---|---|
| log-softmax and top-k on the gpu | not built. every score subtracts the log of the sum of exp over all 151,936 logits, so top-k alone cannot be exact. the logits buffer is shared memory, so there is no copy to drop. search is 0.47% of a step. | dropped |
| widen the next matrix on a second queue | prefill on the long clip 5.1% slower, short and medium inside control spread. costs 64 mb. | dropped |
| object churn in the step loop | everything outside gpu wait and search is 0.45% of a step. inside noise. | dropped |
| shared cache reads across beams in attention | long +2.2%, medium -2.5%, measured warm with a drifting control. | dropped, worth a cool retry |
| hold free rows so groups refill as four | +6.7% against +7.2% without the hold. the groups stayed one plus three. | dropped |
| the engine on a real iphone | bit-identical to the mac and the reference. | kept as a fact |
| an instruments trace of a long clip | gpu busy 95.1% of decode. one idle gap per step, median 2.6 ms under tracing. | kept as a fact |
| hot against cool | see "how to measure". | kept as the protocol |

### found a gain, not mergeable

**grouped refill.** with pieces running, every waiting file piece that fits is read in with one prefill, so each matrix is widened once per group and not once per piece.

- twelve medium clips, full protocol: 13.30x off, 14.52x on, 13.37x off. the controls agree within 0.54 percent. the gain is 8.6 to 9.2 percent.
- no change for the three golden clips together or for three clips three times over.
- first text moves: the second piece 0.90 s to 1.52 s, the sixth 4.21 to 4.59 s, the last 8.54 s to 7.67 s.
- token and score hashes equal the clip run alone, on the mac and the phone.
- **the blocker:** a live piece that arrives beside the batch gets its first text at 3.40 s with groups on, against 1.24 s with groups off and 0.76 s alone. grouping does switch off once the live piece is in the host, but a group that formed just before holds it back about 2.1 s. the cause was being traced when the climb stopped. it is not explained.

### stopped before a result

- **the wait between live draft passes.** per-pass timing on the medium clip: encoder 0.05 to 0.13 s, prefill 0.10 to 0.40 s and growing with the phrase, decode 0.05 to 0.24 s. prefill is the largest part. one idea was built: tokens guessed from the last pass ride along in a greedy step. in replay over 22 passes it was bit-identical (203 comparisons) and made no difference: pass time 0.467 s plain against 0.462 s with the guess. the live comparison was not finished. keeping the prefill cache for the settled prefix, the obvious next idea given those timings, was not tried.
- **the in-step gap (77 against 86 gb/s).** diagnostics only. layers alone took 40.3 ms median and 36.4 ms at best; a warm full step 58.6 ms median and 50.4 ms at best; attention alone 11.6 ms median and 6.6 ms at best; the small kernels 2.3 ms. one candidate was screened twice and its hashes matched, but the machine was throttling (controls of 6.64x and 4.77x in the same session), so the screen says nothing either way.
- **the hand-over between steps.** the trace shows the gpu idle 2.6 ms per step while the cpu work measured in the loop is about 0.5 ms. the instrumentation to measure the true gap without a tracer was written and never ran.
- **the cool retry of shared beam reads.** never ran.

## what is where

nothing below is on the remote. these are local branches.

| branch | holds |
|---|---|
| `worktree-agent-a16b36ac2a601fa4e` | grouped refill (`af63d25`), then a wip commit with trace prints for the live regression |
| `worktree-agent-a59211c5094f003ef` | a device benchmark mode in the app, compiled only with `FOURIER_BENCH` (`7a248c8`) |
| `worktree-agent-a6ff897456a2c2376` | wip: guessed tokens in a greedy step, unfinished |
| `worktree-agent-ad437c9ce15eb9fa9` | wip: gap diagnostics and one candidate behind an environment switch, unfinished |
| `a7-exp2`, `a7-exp3`, `a7-exp4` | the dropped second-queue, object-churn and shared-beam-read changes |

the benchmark harnesses for the mac and the phone were kept outside the repo, in a scratch folder that a reboot can clear. they are not saved anywhere durable.

## what to try next, in order

1. **fix grouped refill for live.** find where the 2.1 s goes, then either switch groups off while a live session is open or bound what a group may delay. the bar: live first text beside a batch within noise of 1.24 s, with the batch-only gain kept.
2. **keep the prefill cache for the settled prefix of a live phrase.** prefill is the largest and fastest-growing part of a draft pass. this is the change a person would feel most.
3. **measure the true gap between steps** from the gpu's own timestamps. if it is near 2.6 ms, encode the next step's commands while the current step runs.
4. **retry shared beam reads cool.** it may win only at long contexts, in which case it could be used above a context length.
5. **the in-step gap,** on a cool machine, one candidate at a time.

not measured at all: energy, the microphone path and record on the phone, batched throughput noise, and whether a candidate that draws more power throttles sooner.

## off the table

- a 32-bit copy of the weights removes the 88 ms of widening per prefill, at the cost of 5.6 gb.
- 16-bit math, quantisation or a smaller model would be faster and would change the outputs.
