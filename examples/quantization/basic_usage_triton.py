from transformers import AutoTokenizer

from gptqmodel import GPTQModel, BACKEND

quantized_path = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

backend = BACKEND.MARLIN

def main():
    prompt = "gptqmodel is an easy-to-use model and"

    model = GPTQModel.from_quantized(
        quantized_path,
        device='cuda:0',
        backend=backend,
    )

    tokenizer = AutoTokenizer.from_pretrained(quantized_path)

    inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    res = model.generate(**inp, num_beams=1, max_new_tokens=60)

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