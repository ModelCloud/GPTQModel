<!-- SPDX-FileCopyrightText: 2026 ModelCloud.ai
SPDX-License-Identifier: Apache-2.0 -->

# Prism Bonsai GGUF compatibility

GPT-QModel loads Prism/Bonsai GGUF checkpoints through its native GGUF loading
path and internal GGUF runtime. No external `gguf` PyPI package is required.
Support is for inference only; Prism model quantization is not included.

Pass a local model path or Hugging Face repository to `GPTQModel.load`:

```python
from gptqmodel import GPTQModel

model = GPTQModel.load("prism-ml/Bonsai-1.7B-gguf")
# or: model = GPTQModel.load("prism-ml/Bonsai-1.7B-gguf", profile="low_memory")

tokens = model.generate(
    "Who wrote Romeo and Juliet?",
    max_new_tokens=128,
)[0]

print(model.tokenizer.decode(tokens, skip_special_tokens=True))
```

Prism Bonsai `Q1_0_g128` remains accepted as a compatibility alias for the
official 128-element `Q1_0` layout.
