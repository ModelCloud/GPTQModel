from gptqmodel import GPTQModel, BACKEND
import subprocess
import sys

quantized_model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

backend = BACKEND.SGLANG

def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sglang>=0.1.19"])

    prompt = "gptqmodel is an easy-to-use model and"

    model = GPTQModel.from_quantized(
        quantized_model_id,
        device='cuda:0',
        backend=backend,
        disable_flashinfer=True
    )

    output = model.generate(prompts=prompt, temperature=0.8, top_p=0.95)

    print(f"Prompt: {prompt}, Generated text: {output}")
    model.shutdown()
    del model


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()