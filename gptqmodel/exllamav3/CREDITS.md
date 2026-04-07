This directory vendors the EXL3 kernel and quantizer pieces adapted from `turboderp-org/exllamav3`.

Primary upstream source:
- https://github.com/turboderp-org/exllamav3

Ported components in this repo:
- `gptqmodel/exllamav3/ext.py`
- `gptqmodel/exllamav3/modules/quant/exl3.py`
- `gptqmodel/exllamav3/modules/quant/exl3_lib/quantize.py`
- `gptqmodel/exllamav3/util/*`
- `gptqmodel_ext/exllamav3/*`

The code remains self-contained inside GPT-QModel and does not depend on the external `exllamav3` Python package.
