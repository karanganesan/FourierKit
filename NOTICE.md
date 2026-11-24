# notice

## this project

`FourierKit` and `FourierPlayground` are copyright (c) 2026 karan ganesan and licensed under [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/). the full text is in [LICENSE](LICENSE).

you can share and adapt this work for non-commercial purposes if you give credit, link to the license and say what you changed. commercial use is not allowed under this license.

## the model

this project runs a model it does not own or include.

| | |
|---|---|
| **model** | [`oruk/oruk-fourier-2b`](https://huggingface.co/oruk/oruk-fourier-2b) |
| **creator** | [oruk](https://oruk.ai) |
| **revision** | `8c2a8871f1304aa2f634da4fb95e24438c4945b9` |
| **license** | [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |

the model is a derivative of these apache-2.0 models, and their credit is kept here:

| base model | creator | license |
|---|---|---|
| [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | qwen | [apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [`Musci-research/Musci-ASR-qwen-1019-hf`](https://huggingface.co/Musci-research/Musci-ASR-qwen-1019-hf), a fine-tune of qwen3-asr | musci-research | [apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |

- **commercial use and hosting:** hosted or api serving, and other commercial use, need a separate commercial license from [oruk](https://oruk.ai). the license of this project does not grant that.
- **app-specific permissions:** oruk's permissions for named apps on the model page do not extend to this project.
- the weights are not in this repo. they are downloaded from the model's page.
- **changes to the weights:** the gpu path reads the bf16 weights as stored. the neural engine encoder, used only when pinned, writes a core ml copy of the encoder weights in 16-bit float on your device. that copy is an adaptation of the model and stays under its license.
- the model's creators do not endorse this project.

## code this project follows

the exact path has to reproduce the reference bit for bit, so parts of the kit follow other projects' code step for step. that behaviour is derived from theirs, so their credit and licenses are kept here. none of them is a dependency of the kit.

| what the kit follows | where in the kit | project | license |
|---|---|---|---|
| beam search (`_beam_search`) and greedy search, the logits processors (no-repeat n-gram, banned tokens), the length penalty and the finished set | `Sources/FourierKit/Text/Search.swift` | [hugging face transformers](https://github.com/huggingface/transformers) | [apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| the qwen3-asr processor: the prompt of `apply_transcription_request`, and the `transcription_only` output parsing with its repetition fix | `Sources/FourierKit/Text/Prompt.swift`, `Sources/FourierKit/Text/TranscriptText.swift` | hugging face transformers (`models/qwen3_asr`) | apache-2.0 |
| the feature extractor: slaney mel filters (`mel_filter_bank`), the hann window, log10, clamp and floor, the minimum length and mask | `Sources/FourierKit/Audio/LogMel.swift` | hugging face transformers | apache-2.0 |
| the model's forward pass (conv front, encoder, projector, qwen3 decoder, rotary positions, sinusoidal positions) as written in the qwen3-asr modeling code | `Sources/FourierKit/Model/`, `Sources/FourierKit/Model/CoreML/EncoderGraph.swift` | hugging face transformers, and [qwen3-asr](https://github.com/QwenLM/Qwen3-ASR) | apache-2.0 |
| the byte-level bpe tokenizer and its lossy decode, and the qwen2 pre-tokenizer pattern with oniguruma semantics | `Sources/FourierKit/Text/Tokenizer.swift` | [hugging face tokenizers](https://github.com/huggingface/tokenizers), and the pattern from the checkpoint's `tokenizer.json` | apache-2.0 |
| `torch.topk(largest=True, sorted=True)` order including exact ties, `torch.stft(center=True)` reflect padding, and float32 rounding that matches torch's arm64 cpu kernels | `Sources/FourierKit/Text/TopK.swift`, `Sources/FourierKit/Audio/LogMel.swift`, `Sources/FourierKit/Text/TorchFloat.swift` | [pytorch](https://github.com/pytorch/pytorch) | [bsd-3-clause](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| the steps of `std::partial_sort` and `std::nth_element`, which torch's top-k uses | `Sources/FourierKit/Text/TopK.swift` | [llvm libc++](https://github.com/llvm/llvm-project/tree/main/libcxx) | [apache-2.0 with llvm exceptions](https://llvm.org/LICENSE.txt) |
| the `expf4_u10` and `logf4_u10` recipes, which torch's vectorised `exp` and `log` use on arm64 | `Sources/FourierKit/Text/TorchFloat.swift` | [sleef](https://github.com/shibatch/sleef) | [boost software license 1.0](https://www.boost.org/LICENSE_1_0.txt) |
| `numpy.linspace` spacing, inside the mel filters | `Sources/FourierKit/Audio/LogMel.swift` | [numpy](https://github.com/numpy/numpy) | [bsd-3-clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| the core ml model format (protobuf messages and the mil program), written from swift | `Sources/FourierKit/Model/CoreML/` | [apple coremltools](https://github.com/apple/coremltools), whose public format definitions were read | [bsd-3-clause](https://github.com/apple/coremltools/blob/main/LICENSE.txt) |

the gpu's `erf` in `Sources/FourierKit/Model/Kernels.swift` uses chebyshev fits made for this project. it is not taken from another library.

`reference/python/` is the test harness that makes the golden set. it uses transformers, torch, numpy, soundfile and safetensors as dependencies through `uv`. it never ships.

## the font

`FourierKitUI` and `FourierPlayground` use [allo](https://allo.joshpuckett.me) by josh puckett, based on [inter](https://github.com/rsms/inter) by the inter project authors.

- the font file is bundled unchanged at `Sources/FourierKitUI/Resources/Fonts/AlloVariable.woff2`.
- it is licensed under the sil open font license 1.1, included beside it as `Sources/FourierKitUI/Resources/Fonts/OFL.txt`.
- the cc by-nc 4.0 license of this project does not apply to the font.

## golden clips

the three clips in `golden/clips/` are cut from "the gettysburg address", read for librivox ([archive.org/details/gettysburg_johng_librivox](https://archive.org/details/gettysburg_johng_librivox)). the recording is in the public domain. the cut points are in `golden/manifest.json`.

## references

these were read for ideas. no code from them is in this repo, and none of them is a dependency.

- [fluidaudio](https://github.com/FluidInference/FluidAudio)
- [fluidvoice](https://github.com/altic-dev/FluidVoice)
- [apple: deploying transformers on the neural engine](https://machinelearning.apple.com/research/neural-engine-transformers), whose `[1][feature][1][token]` layout the neural engine encoder uses
- [apple: on-device llama with core ml](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill), the source of the hill-climb method
