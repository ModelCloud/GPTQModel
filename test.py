import time

from transformers import AutoTokenizer, LlamaModel

from gptqmodel import GPTQModel
from gptqmodel.integration.src.transformers.models.llama.modeling_llama import LlamaModel as patched_LlamaModel


model_id = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"

model = GPTQModel.load(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer("Uncovering deep insights begins with", return_tensors="pt").to(model.device)

#### non patched ####
now = time.time()

result = model.generate(
    **input_ids, min_new_tokens=100, max_new_tokens=100
)[0]

print(f"non patched time={time.time() - now}")

#### patched ####
LlamaModel.forward = patched_LlamaModel.forward

now = time.time()

result = model.generate(
    **input_ids, min_new_tokens=100, max_new_tokens=100
)[0]

print(f"patchedt ime={time.time() - now}")

if __name__ == '__main__':
    print()
