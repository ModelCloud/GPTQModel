import os
import subprocess
import sys
import torch
from argparse import ArgumentParser

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig, get_backend, get_best_device
from transformers import AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
pretrained_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quantized_model_id = "./TinyLlama/TinyLlama-1.1B-Chat-v1.0-4bit-128g"

def main():
    global quantized_model_id

    parser = ArgumentParser()
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA_V2', 'MARLIN', 'CUDA', 'BITBLAS', 'IPEX', 'SGLANG', 'VLLM'])
    args = parser.parse_args()

    backend = get_backend(args.backend)
    device = get_best_device(backend)

    if backend == BACKEND.SGLANG:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.6.2"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang[srt]>=0.3.2"])
    elif backend == BACKEND.VLLM:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.6.2"])

    prompt = "I am in Paris and"

    if backend == BACKEND.SGLANG or backend == BACKEND.VLLM:
        quantized_model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

        if backend == BACKEND.SGLANG:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.6.2"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang[srt]>=0.3.2"])
            model = GPTQModel.load(
                quantized_model_id,
                device=device,
                backend=backend,
                disable_flashinfer=True
            )

            output = model.generate(prompts=prompt, temperature=0.8, top_p=0.95)
            model.shutdown()
            del model
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.6.2"])
            model = GPTQModel.load(
                quantized_model_id,
                device=device,
                backend=backend,
            )
            output = model.generate(prompts=prompt, temperature=0.8, top_p=0.95)[0].outputs[0].text
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
        examples = [
            tokenizer(
                "I am in Paris and I can't wait to explore its beautiful"
            )
        ]

        quantize_config = QuantizeConfig(
            bits=4,  # quantize model to 4-bit
            group_size=128,  # it is recommended to set the value to 128
            # set to False can significantly speed up inference but the perplexity may slightly bad
            desc_act=False if backend == BACKEND.BITBLAS or backend == BACKEND.MARLIN else True,
        )

        # load un-quantized model, by default, the model will always be loaded into CPU memory
        model = GPTQModel.load(pretrained_model_id, quantize_config)

        # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
        model.quantize(examples)

        # save quantized model
        model.save(quantized_model_id)
        tokenizer.save_pretrained(quantized_model_id)

        model = GPTQModel.load(
            quantized_model_id,
            device=device,
            backend=backend,
        )

        inp = tokenizer(prompt, return_tensors="pt").to(device)

        res = model.generate(**inp, num_beams=1, max_new_tokens=10)
        output = tokenizer.decode(res[0])

    print(f"Prompt: {prompt}, Generated text: {output}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
