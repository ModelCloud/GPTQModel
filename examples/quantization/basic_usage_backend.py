import subprocess

import os
import torch
from argparse import ArgumentParser
from future.moves import sys
from gptqmodel import get_backend, QuantizeConfig, BACKEND, GPTQModel
from transformers import AutoTokenizer, TextGenerationPipeline

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
pretrained_model_id = "facebook/opt-125m"
quantized_model_id = "./facebook/opt-125m-4bit-128g"

def main():

    parser = ArgumentParser()
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA', 'EXLLAMA_V2', 'MARLIN', 'BITBLAS', 'QBITS'])
    args = parser.parse_args()

    backend = get_backend(args.backend)

    if backend == BACKEND.SGLANG:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang>=0.1.19"])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
    examples = [
        tokenizer(
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False if backend == BACKEND.BITBLAS or backend == BACKEND.MARLIN else True,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = GPTQModel.from_pretrained(pretrained_model_id, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples)

    # save quantized model
    model.save_quantized(quantized_model_id)
    tokenizer.save_pretrained(quantized_model_id)

    prompt = "gptqmodel is"

    device = 'cpu' if backend == BACKEND.QBITS else 'cuda:0'

    model = GPTQModel.from_quantized(
        quantized_model_id,
        device=device,
        backend=backend,
    )

    inp = tokenizer(prompt, return_tensors="pt").to(device)

    res = model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

    # -- simple token evaluate --

    output = tokenizer.decode(res[0])
    print(f"Prompt: {prompt}, Generated text: {output}")

    # or you can also use pipeline
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    print(pipeline(prompt)[0]["generated_text"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()