#!/usr/bin/env bash
set -euo pipefail

cd /root/qvq-gsq-validation

exec /root/venv-py3.14t/bin/python -m gpu_allocator.cli run -n 1 --style uuid -- \
  /root/venv-py3.14t/bin/python - <<'PY'
import json

import torch

from gptqmodel import BACKEND, GPTQModel


model_path = "/root/qvq-results/w3-p32-gsq-ab-20260915/gsq"
prompt = "What is 2 + 2? Answer with a number."

print(json.dumps({
    "phase": "load",
    "model": model_path,
    "backend": "qvq",
    "device": "cuda",
    "dtype": "float16",
    "prompt": prompt,
}), flush=True)

model = GPTQModel.load(
    model_path,
    backend=BACKEND.QVQ,
    dtype=torch.float16,
)
model.model.to("cuda").eval()
encoded = model.tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    logits = model.model(**encoded).logits[:, -1, :].float()
    top = torch.topk(logits, 5, dim=-1)
    generated = model.generate(
        **encoded,
        max_new_tokens=32,
        do_sample=False,
    )

new_tokens = generated[0, encoded.input_ids.shape[1]:]
print(json.dumps({
    "finite_logits": bool(torch.isfinite(logits).all()),
    "logit_min": float(logits.min()),
    "logit_max": float(logits.max()),
    "top_ids": top.indices[0].tolist(),
    "top_values": top.values[0].tolist(),
    "generated_ids": new_tokens.tolist(),
    "text": model.tokenizer.decode(new_tokens),
}, indent=2), flush=True)
PY
