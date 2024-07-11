from transformers import AutoTokenizer

from gptqmodel import GPTQModel, BACKEND

quantized_model_id = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

backend = BACKEND.MARLIN

def main():
    prompt = "gptqmodel is an easy-to-use model and"

    model = GPTQModel.from_quantized(
        quantized_model_id,
        device='cuda:0',
        backend=backend,
    )

    tokenizer = AutoTokenizer.from_pretrained(quantized_model_id)

    inp = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    res = model.generate(**inp, num_beams=1, min_new_tokens=60, max_new_tokens=60)

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