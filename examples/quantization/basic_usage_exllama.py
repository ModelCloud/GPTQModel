import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, BACKEND, get_backend

quantized_model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"


def main():
    parser = ArgumentParser()
    parser.add_argument("--backend", choices=['EXLLAMA', 'EXLLAMA_V2'])
    args = parser.parse_args()

    prompt = "gptqmodel is an easy-to-use model and"

    model = GPTQModel.from_quantized(
        quantized_model_id,
        device="cuda:0",
        backend=get_backend(args.backend),
    )

    tokenizer = AutoTokenizer.from_pretrained(quantized_model_id)

    input = tokenizer(prompt, return_tensors='pt').to("cuda:0")

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