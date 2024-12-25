import time

from transformers import AutoTokenizer, LlamaModel

from gptqmodel import GPTQModel


model_id = "/monster/data/lrl/qwq_quant/vortex_v2"

model = GPTQModel.load(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_ids = tokenizer("Uncovering deep insights begins with", return_tensors="pt").to(model.device)

now = time.time()

result = model.generate(
    **input_ids, min_new_tokens=100, max_new_tokens=100
)[0]

print(f"non patched time={time.time() - now}")

# 20.77G 5.4s
if __name__ == '__main__':
    print()
