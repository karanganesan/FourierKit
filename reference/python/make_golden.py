"""makes the golden reference set with hugging face transformers.

the reference: transformers at one pinned commit, the pinned checkpoint, cpu, 32-bit float,
a batch of one, apply_transcription_request(language="en"), no system prompt, and the model's
generation defaults. a second 64-bit run measures 32-bit noise, and a clip is rejected if any of
its top-candidate comparisons sits inside that noise (see `gaps`).

nothing is written into the output directory (default golden/) unless every clip passed that
check: the set is built in a temporary directory beside it and moved into place at the end.
--skip-float64 skips the check, and then needs --unvalidated and writes "validated": false, which
the golden test refuses.

run from the repo root:  uv run --project reference/python python reference/python/make_golden.py
this never ships and is not a dependency of the kit.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from safetensors.numpy import save_file
from transformers import AutoModelForMultimodalLM, AutoProcessor

REPO = "oruk/oruk-fourier-2b"
REVISION = "8c2a8871f1304aa2f634da4fb95e24438c4945b9"
WEIGHTS_SHA256 = "e64765ffdd90c83704c0951e0d88b7f2ed84499af5c208f7ddf8d922cf08441c"
CLIPS = ["short", "medium", "long"]
RECORDED = 10  # top candidates saved per step: the 9 that beam search keeps, plus the first one it drops
MARGIN_LIMIT = 0.5  # a clip is rejected when a comparison uses this much of its margin; the test reads it from the manifest
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    Path.home() / "Library/Application Support/FourierKit/models/oruk--oruk-fourier-2b" / REVISION
)


def sha256_file(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class TopKRecorder:
    """records the accumulated candidates beam search ranks at each step.

    beam search calls torch.topk on the flattened num_beams * vocab scores once per step. this
    wraps that one call and reads it without changing what it returns.
    """

    def __init__(self, width):
        self.width = width
        self.scores = []
        self.indices = []
        self.original = torch.topk

    def __enter__(self):
        def topk(input, k, *args, **kwargs):
            if input.dim() == 2 and input.shape[-1] == self.width:
                values, indices = self.original(input, RECORDED)
                self.scores.append(values[0].to(torch.float64).numpy().copy())
                self.indices.append(indices[0].numpy().copy())
            return self.original(input, k, *args, **kwargs)

        torch.topk = topk
        return self

    def __exit__(self, *exc):
        torch.topk = self.original


def plain_seconds(model, processor, samples):
    """seconds of one plain warm generate: no recording hooks, no scores or logits kept.

    the caller has already run generate once, so one-time costs are paid.
    """
    inputs = processor.apply_transcription_request(
        language="en", audio=samples, sampling_rate=16000, return_tensors="pt"
    ).to("cpu", dtype=torch.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inputs)
    return time.perf_counter() - started


def run(model, processor, samples, dtype):
    inputs = processor.apply_transcription_request(
        language="en", audio=samples, sampling_rate=16000, return_tensors="pt"
    )
    features = inputs["input_features"]
    inputs = inputs.to("cpu", dtype=dtype)
    captured = {}
    original = model.model.get_audio_features

    def get_audio_features(*args, **kwargs):
        output = original(*args, **kwargs)
        # generate repeats the clip once per beam before this call, so keep one copy
        rows = output.pooler_output.shape[0] // model.generation_config.num_beams
        captured["audio"] = output.pooler_output[:rows].detach().to(torch.float64).numpy().copy()
        return output

    model.model.get_audio_features = get_audio_features
    vocab = model.config.text_config.vocab_size
    beams = model.generation_config.num_beams
    started = time.perf_counter()
    try:
        with torch.inference_mode(), TopKRecorder(beams * vocab) as recorder:
            output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, output_logits=True)
    finally:
        model.model.get_audio_features = original
    seconds = time.perf_counter() - started
    prompt = inputs["input_ids"][0]
    sequence = output.sequences[0]
    generated = sequence[prompt.shape[0] :]
    return {
        "prompt_ids": prompt.numpy().copy(),
        "mel": features[0].numpy().copy(),
        "mel_mask": inputs["input_features_mask"][0].numpy().copy(),
        "audio": captured["audio"],
        "token_ids": generated.numpy().copy(),
        "top_scores": np.stack(recorder.scores),
        "top_indices": np.stack(recorder.indices),
        "first_logits": output.logits[0][0].to(torch.float64).numpy().copy(),
        "sequence_score": float(output.sequences_scores[0]),
        "seconds": seconds,
    }


def gaps(scores):
    """the gap between neighbours among the ranked candidates of every step.

    these are the comparisons that rank the candidates of one step: the order of the 9 that beam
    search keeps, and the cut between the 9th and the 10th. they are not every comparison beam
    search makes: the finished-set ranking (length-penalised scores of different steps) and the
    stop heuristic are not in these gaps; the golden test checks them from the same recorded scores.
    candidates near -1e9 are masked beams or banned tokens and are skipped.
    """
    return scores[:, :-1] - scores[:, 1:], scores[:, 1:] > -1e8


def smallest_gap(scores):
    values, live = gaps(scores)
    return float(values[live].min())


def margin_used(reference, other):
    """how much of the safety margin a second run uses, over every gap in `gaps`.

    a comparison keeps its order when the other run's gap is off by less than half the reference
    gap. this returns the worst ratio of gap error to half the gap, so under 1 means every
    comparison is safe. it compares gaps, not raw scores, because an error shared by both sides of
    a comparison cannot change its outcome.
    """
    wanted, live = gaps(reference)
    got, _ = gaps(other)
    return float((np.abs(got - wanted)[live] / (wanted[live] / 2)).max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--skip-float64", action="store_true", help="skip the 64-bit noise check; needs --unvalidated")
    parser.add_argument("--unvalidated", action="store_true", help="allow writing a set without the noise check, marked validated: false")
    parser.add_argument("--output", default=str(ROOT / "golden"), help="directory for manifest.json and tensors/ (default: the repo's golden/)")
    parser.add_argument("--clips", nargs="*", default=CLIPS)
    args = parser.parse_args()
    if args.skip_float64 and not args.unvalidated:
        sys.exit("--skip-float64 writes a golden set with no noise check. pass --unvalidated as well to write it, marked validated: false")
    if args.unvalidated and not args.skip_float64:
        sys.exit("--unvalidated only goes with --skip-float64")
    output = Path(args.output).resolve()

    model_dir = Path(args.model)
    weights = model_dir / "model.safetensors"
    print("hashing weights", flush=True)
    weights_hash = sha256_file(weights)
    if weights_hash != WEIGHTS_SHA256:
        sys.exit(f"weights hash {weights_hash} does not match the pinned {WEIGHTS_SHA256}")

    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    generation = model.generation_config
    eos = list(generation.eos_token_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        build(args, model, processor, generation, eos, weights_hash, staging, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def build(args, model, processor, generation, eos, weights_hash, staging, output):
    tensors_dir = staging / "tensors"
    tensors_dir.mkdir()
    clips = {}
    results = {}
    for name in args.clips:
        path = ROOT / "golden" / "clips" / f"{name}.wav"
        samples, rate = sf.read(path, dtype="float32", always_2d=False)
        assert rate == 16000 and samples.ndim == 1 and samples.dtype == np.float32
        print(f"{name}: {len(samples) / rate:.2f} s, 32-bit run", flush=True)
        result = run(model, processor, samples, torch.float32)
        results[name] = result
        ids = result["token_ids"].tolist()
        stop = ids[-1] if ids and ids[-1] in eos else None
        raw_text = processor.decode(result["token_ids"], skip_special_tokens=True)
        final_text = processor.decode(
            np.concatenate([result["prompt_ids"], result["token_ids"]]), return_format="transcription_only"
        )
        if stop is None or len(ids) >= generation.max_new_tokens:
            sys.exit(f"{name} did not reach a stop token under the token limit, so it cannot be a golden clip")
        clips[name] = {
            "file": f"golden/clips/{name}.wav",
            "file_sha256": sha256_file(path),
            "samples": int(len(samples)),
            "samples_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            "seconds": len(samples) / rate,
            "mel_frames": int(result["mel_mask"].sum()),
            "audio_tokens": int(result["audio"].shape[0]),
            "prompt_ids_head": result["prompt_ids"][:9].tolist(),
            "prompt_ids_tail": result["prompt_ids"][-9:].tolist(),
            "prompt_length": int(result["prompt_ids"].shape[0]),
            "token_ids": ids,
            "stop_token": stop,
            "steps": int(result["top_scores"].shape[0]),
            "raw_text": raw_text,
            "final_text": final_text,
            "sequence_score": result["sequence_score"],
            "smallest_gap": smallest_gap(result["top_scores"]),
            # one instrumented run: extra top-k, output scores and logits, and the first call's
            # warm-up on the first clip. not a plain warm generate; see reference_seconds_plain.
            "reference_seconds": result["seconds"],
        }
        print(f"  {len(ids)} tokens in {result['seconds']:.1f} s: {final_text[:96]}", flush=True)

    # a plain warm generate per clip, timed apart from the instrumented runs above (which record
    # top-k, scores and logits, and pay the first call's warm-up on the first clip). one warm-up
    # call first, on the shortest clip.
    warm_up = sf.read(ROOT / "golden" / "clips" / f"{args.clips[0]}.wav", dtype="float32", always_2d=False)[0]
    plain_seconds(model, processor, warm_up)
    for name in args.clips:
        samples, _ = sf.read(ROOT / "golden" / "clips" / f"{name}.wav", dtype="float32", always_2d=False)
        clips[name]["reference_seconds_plain"] = plain_seconds(model, processor, samples)
        print(f"{name}: plain warm generate {clips[name]['reference_seconds_plain']:.1f} s", flush=True)

    rejected = []
    if not args.skip_float64:
        model = model.to(torch.float64)
        for name in args.clips:
            path = ROOT / "golden" / "clips" / f"{name}.wav"
            samples, _ = sf.read(path, dtype="float32", always_2d=False)
            print(f"{name}: 64-bit run", flush=True)
            wide = run(model, processor, samples, torch.float64)
            narrow = results[name]
            same = wide["token_ids"].tolist() == narrow["token_ids"].tolist()
            same_candidates = (
                wide["top_indices"].shape == narrow["top_indices"].shape
                and bool((wide["top_indices"][:, :9] == narrow["top_indices"][:, :9]).all())
            )
            noise = None
            used = None
            if same_candidates:
                live = narrow["top_scores"] > -1e8
                noise = float(np.abs(wide["top_scores"] - narrow["top_scores"])[live].max())
                used = margin_used(narrow["top_scores"], wide["top_scores"])
            clips[name]["float64"] = {
                "same_tokens": same,
                "same_candidates": same_candidates,
                "score_noise": noise,
                "margin_used": used,
                "audio_max_abs_diff": float(np.abs(wide["audio"] - narrow["audio"]).max()),
            }
            narrow["top_scores_float64"] = wide["top_scores"] if same_candidates else None
            print(f"  same tokens {same}, same candidates {same_candidates}, score noise {noise}, "
                  f"margin used {used}, smallest gap {clips[name]['smallest_gap']}", flush=True)
            # at 1 the reference itself would be one rounding away from another answer. half of
            # that keeps a clip well clear of 32-bit noise.
            if not same or not same_candidates or used >= MARGIN_LIMIT:
                rejected.append(name)

    if rejected:
        sys.exit(f"{rejected} have a top-candidate comparison inside 32-bit noise, so they are rejected as golden clips. nothing was written to {output}")

    tensor_hashes = {}
    for name in args.clips:
        result = results[name]
        tensors = {
            "prompt_ids": result["prompt_ids"].astype(np.int32),
            "mel": result["mel"].astype(np.float32),
            "mel_mask": result["mel_mask"].astype(np.int32),
            "audio": result["audio"].astype(np.float32),
            "token_ids": result["token_ids"].astype(np.int32),
            "top_scores": result["top_scores"].astype(np.float32),
            "top_indices": result["top_indices"].astype(np.int32),
            "first_logits": result["first_logits"].astype(np.float32),
        }
        if result.get("top_scores_float64") is not None:
            tensors["top_scores_float64"] = result["top_scores_float64"].astype(np.float64)
        target = tensors_dir / f"{name}.safetensors"
        save_file(tensors, str(target))
        tensor_hashes[name] = sha256_file(target)
        clips[name]["tensors"] = f"golden/tensors/{name}.safetensors"
        clips[name]["tensors_sha256"] = tensor_hashes[name]

    manifest = {
        "model": {"repo": REPO, "revision": REVISION, "weights_sha256": weights_hash},
        "reference": {
            "transformers": transformers.__version__,
            "transformers_commit": "d8f10a82213a3a9db48385aba3a4b57ab535de50",
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "batch": 1,
            "request": 'apply_transcription_request(language="en")',
            "system_prompt": None,
            "return_format": "transcription_only",
            "threads": torch.get_num_threads(),
        },
        "generation": {
            "num_beams": generation.num_beams,
            "max_new_tokens": generation.max_new_tokens,
            "no_repeat_ngram_size": generation.no_repeat_ngram_size,
            "length_penalty": generation.length_penalty,
            "repetition_penalty": generation.repetition_penalty,
            "early_stopping": generation.early_stopping,
            "do_sample": generation.do_sample,
            "eos_token_id": eos,
            "pad_token_id": generation.pad_token_id,
            "banned_tokens": len(generation.bad_words_ids),
        },
        "validated": not args.skip_float64,
        "margin_threshold": MARGIN_LIMIT,
        "recorded_candidates": RECORDED,
        "source": {
            "title": "the gettysburg address, read for librivox",
            "url": "https://archive.org/details/gettysburg_johng_librivox",
            "license": "public domain",
            "cuts_seconds": {"short": [54.75, 58.45], "medium": [38.45, 49.65], "long": [35.0, 151.5]},
        },
        "clips": clips,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    # every clip passed: move the set into place, tensors first so a manifest never points at a missing file
    output.mkdir(parents=True, exist_ok=True)
    (output / "tensors").mkdir(exist_ok=True)
    for name in args.clips:
        os.replace(tensors_dir / f"{name}.safetensors", output / "tensors" / f"{name}.safetensors")
    os.replace(staging / "manifest.json", output / "manifest.json")
    print(f"wrote {output / 'manifest.json'}" + ("" if manifest["validated"] else " (unvalidated)"), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
