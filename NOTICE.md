# notice

## this project

`FourierKit` and `FourierPlayground` are copyright (c) 2026 karan ganesan and licensed under [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/). the full text is in [LICENSE](LICENSE).

you can share and adapt this work for non-commercial purposes if you give credit, link to the license and say what you changed. commercial use is not allowed under this license.

## the model

this project runs a model it does not own or include.

| | |
|---|---|
| **model** | [`oruk/oruk-fourier-2b`](https://huggingface.co/oruk/oruk-fourier-2b) |
| **creator** | oruk |
| **revision** | `8c2a8871f1304aa2f634da4fb95e24438c4945b9` |
| **license** | [cc by-nc 4.0](https://creativecommons.org/licenses/by-nc/4.0/) |
| **base models listed on its model card** | `Qwen/Qwen3-ASR-1.7B`, `Musci-research/Musci-ASR-qwen-1019-hf` |

- the weights are not in this repo. they are downloaded from the model's page.
- **changes to the weights:** the kit converts them on your machine into formats for the cpu, gpu and neural engine. this can include converting bf16 values to 16-bit float. the converted files are adaptations of the model and stay under its license.
- the model's creators do not endorse this project.

## references

these projects were read for ideas only. no code from them is copied here, and none of them is a dependency.

- [hugging face transformers](https://github.com/huggingface/transformers), also used by a separate test harness that never ships
- [qwen3-asr](https://github.com/QwenLM/Qwen3-ASR)
- [fluidaudio](https://github.com/FluidInference/FluidAudio)
- [fluidvoice](https://github.com/altic-dev/FluidVoice)
- [apple: deploying transformers on the neural engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [apple: on-device llama with core ml](https://machinelearning.apple.com/research/core-ml-on-device-llama)
