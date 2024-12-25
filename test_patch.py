import time

from transformers import AutoTokenizer, LlamaModel

from gptqmodel import GPTQModel
from gptqmodel.integration.src.transformers.models.llama.modeling_llama import LlamaModel as patched_LlamaModel

LlamaModel.forward = patched_LlamaModel.forward

model_id = "/monster/data/lrl/qwq_quant/vortex_v2"

model = GPTQModel.load(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer("Uncovering deep insights begins with", return_tensors="pt").to(model.device)


patched_model = GPTQModel.load(model_id)

now = time.time()

result = patched_model.generate(
    **input_ids, min_new_tokens=100, max_new_tokens=100
)[0]

print(f"patched ime={time.time() - now}")

if __name__ == '__main__':
    print()
