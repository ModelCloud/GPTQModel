import subprocess
import sys

from gptqmodel import GPTQModel, BACKEND
from vllm import SamplingParams

subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.5.1"])
quantized_model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
backend = BACKEND.VLLM

def main():
    prompt = [
        "gptqmodel is an easy-to-use model and",
        "Hello, my name is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    model = GPTQModel.from_quantized(quantized_model_id, backend=backend, device='cuda:0')

    outputs = model.generate(prompts=prompt, sampling_params=sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}, Generated text: {generated_text}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()