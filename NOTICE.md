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
- **changes to the weights:** the kit converts them on your machine into formats for the cpu, gpu and neural engine. this can include converting bf16 values to 16-bit float. the converted files are adaptations of the model and stay under its license.
- the model's creators do not endorse this project.

## the font

`FourierKitUI` and `FourierPlayground` use [allo](https://allo.joshpuckett.me) by josh puckett, based on [inter](https://github.com/rsms/inter) by the inter project authors.

- the font file is bundled unchanged at `Sources/FourierKitUI/Resources/Fonts/AlloVariable.woff2`.
- it is licensed under the sil open font license 1.1, included beside it as `OFL.txt`.
- the cc by-nc 4.0 license of this project does not apply to the font.

## references

these projects were read for ideas only. no code from them is copied here, and none of them is a dependency.

- [hugging face transformers](https://github.com/huggingface/transformers), also used by a separate test harness that never ships
- [qwen3-asr](https://github.com/QwenLM/Qwen3-ASR)
- [fluidaudio](https://github.com/FluidInference/FluidAudio)
- [fluidvoice](https://github.com/altic-dev/FluidVoice)
- [apple: deploying transformers on the neural engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [apple: on-device llama with core ml](https://machinelearning.apple.com/research/core-ml-on-device-llama)
