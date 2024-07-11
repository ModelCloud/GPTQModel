import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, BACKEND


quantized_path = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

backend = BACKEND.QBITS

device = torch.device('cpu')
def main():
    prompt = "gptqmodel is an easy-to-use model and"

    model = GPTQModel.from_quantized(
        quantized_path,
        backend=backend,
    )

    tokenizer = AutoTokenizer.from_pretrained(quantized_path)

    input = tokenizer(prompt, return_tensors='pt').to(device)

    result = model.generate(**input, num_beams=1, max_new_tokens=5)

    output = tokenizer.decode(result[0])

    print(f"Prompt: {prompt}, Generated text: {output}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()