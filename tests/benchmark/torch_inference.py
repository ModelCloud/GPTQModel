import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


model_id = "/monster/data/lrl/llama31/best-8b-it/gptq_4bit_07-29_06-46-12_maxlen2048_ns1024_descFalse_damp0.005/"
device = 'cuda'
max_new_token = 100

prompts = [
            "I am in Paris and I",
            "The capital of the United Kingdom is",
            "The largest ocean on Earth is",
            "The world’s longest river is",
            "The tallest mountain in the world is",
            "The currency used in Japan is",
            "How to consult a dictionary?",
            "What is the boiling point of water in degrees Celsius?"
            "Which is the most widely used Internet search engine in the world?",
            "What is the official language of France?"
        ]


model = GPTQModel.from_quantized(
            model_id,
            device=device,
            backend=BACKEND.TORCH,
        )

tokenizer = AutoTokenizer.from_pretrained(model_id)
inp = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
res = model.generate(**inp, num_beams=1, min_new_tokens=max_new_token, max_new_tokens=max_new_token)



if __name__ == '__main__':
    pass