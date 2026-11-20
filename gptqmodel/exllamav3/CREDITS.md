This directory vendors the EXL3 kernel and quantizer pieces adapted from `turboderp-org/exllamav3`.

Primary upstream source:
- https://github.com/turboderp-org/exllamav3

Ported components in this repo:
- `gptqmodel/exllamav3/ext.py`
- `gptqmodel/exllamav3/modules/quant/exl3.py`
- `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py`
- `gptqmodel/exllamav3/util/*`
- `gptqmodel_ext/exllamav3/*`

AlexBrandes (Alex Brandes) developed the Q4NX codebook and beam Viterbi
encoder. GPT-QModel adapted the encoder from his ExLlamaV3 fork (MIT) and
the tensor-core fragment decoder from his vLLM fork (Apache-2.0).
[Encoder source](https://github.com/AlexBrandes/exllamav3/commit/0674697c45e2450ffd4538eb5ade0cf8e6a78f6d) · [Decoder source](https://github.com/AlexBrandes/vllm/blob/ff413b2df19718342e02d9763f161d69afce7b4a/csrc/exl3/codec_q4n.cuh)

The code remains self-contained inside GPT-QModel and does not depend on the external `exllamav3` Python package.
