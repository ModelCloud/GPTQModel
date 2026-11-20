# EXL3 credits

GPT-QModel adapts the EXL3 kernels and quantizer from ExLlamaV3 by TurboDerp
and contributors. The EXL3 port includes the Q4NX codebook, beam Viterbi
encoder, and tensor-core fragment decoder developed by AlexBrandes
(Alex Brandes). The encoder comes from his MIT-licensed ExLlamaV3 fork and
the decoder from his Apache-2.0 vLLM fork. ModelCloud extended Q4NX from 2.5
and 3.0 bits per weight to support 3.5 bits per weight. [ExLlamaV3](https://github.com/turboderp-org/exllamav3) · [Q4NX encoder](https://github.com/AlexBrandes/exllamav3/commit/0674697c45e2450ffd4538eb5ade0cf8e6a78f6d) · [Q4NX decoder](https://github.com/AlexBrandes/vllm/blob/ff413b2df19718342e02d9763f161d69afce7b4a/csrc/exl3/codec_q4n.cuh)

Ported components in this repo:
- `gptqmodel/exllamav3/ext.py`
- `gptqmodel/exllamav3/modules/quant/exl3.py`
- `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py`
- `gptqmodel/exllamav3/util/*`
- `gptqmodel_ext/exllamav3/*`

The code remains self-contained inside GPT-QModel and does not depend on the external `exllamav3` Python package.
