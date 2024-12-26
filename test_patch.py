import time

import torch
from transformers import AutoTokenizer, LlamaModel
# from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from gptqmodel import GPTQModel, BACKEND
from gptqmodel.integration.src.transformers.models.llama.modeling_llama import LlamaModel as patched_LlamaModel

LlamaModel.forward = patched_LlamaModel.forward

# model_id = "/monster/data/lrl/qwq_quant/vortex_v2"
model_id = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"

model = GPTQModel.load(model_id, backend = BACKEND.TORCH, device = "cpu")
# model.to("cpu")

# for name, module in model.named_modules():
#     if isinstance(module, LlamaDecoderLayer):
#       print(f"eeeeeeeeee {name} {module}".replace("\n", "\\n"))
#       module.to("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer("Uncovering deep insights begins with", return_tensors="pt").to(model.device)


now = time.time()

result = model.generate(
    **input_ids, min_new_tokens=100, max_new_tokens=100
)[0]

print(f"patched ime={time.time() - now}")

if __name__ == '__main__':
    print()
