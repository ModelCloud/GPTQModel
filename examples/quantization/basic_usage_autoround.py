import torch
from gptqmodel import GPTQModel
from gptqmodel.quantization.config import AutoRoundQuantizeConfig  # noqa: E402
from transformers import AutoTokenizer

pretrained_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quantized_model_id = "./autoround/TinyLlama-1.1B-Chat-v1.0-4bit-128g"

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)
    examples = [
        tokenizer(
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = AutoRoundQuantizeConfig(
        bits=4,
        group_size=128
    )

    model = GPTQModel.load(
        pretrained_model_id,
        quantize_config=quantize_config,
    )

    model.quantize(examples)

    model.save(quantized_model_id)

    tokenizer.save_pretrained(quantized_model_id)

    del model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.from_quantized(
        quantized_model_id,
        device=device,
    )

    input_ids = torch.ones((1, 1), dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids)
    print(f"output logits {outputs.logits.shape}: \n", outputs.logits)
    # inference with model.generate
    print(
        tokenizer.decode(
            model.generate(
                **tokenizer("gptqmodel is", return_tensors="pt").to(model.device)
            )[0]
        )
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
