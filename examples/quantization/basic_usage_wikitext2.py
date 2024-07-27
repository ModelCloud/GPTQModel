import torch
import torch.nn as nn
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

pretrained_model_id = "facebook/opt-125m"
quantized_model_id = "opt-125m-4bit-128g"


# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


@torch.no_grad()
def calculate_avg_ppl(model, tokenizer):
    from gptqmodel.utils import Perplexity

    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path="wikitext",
        dataset_name="wikitext-2-raw-v1",
        split="train",
        text_column="text",
    )

    all = ppl.calculate(n_ctx=512, n_batch=512)

    # average ppl
    avg = sum(all) / len(all)

    return avg

def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id, use_fast=True)

    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=1024)

    quantize_config = QuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = GPTQModel.from_pretrained(pretrained_model_id, quantize_config)

    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset)

    # save quantized model
    model.save_quantized(quantized_model_id)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_id, use_safetensors=True)

    # load quantized model, currently only support cpu or single gpu
    model = GPTQModel.from_quantized(quantized_model_id, device="cuda:0")

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("test is", return_tensors="pt").to("cuda:0"))[0]))

    print(f"Quantized Model {quantized_model_id} avg PPL is {calculate_avg_ppl(model, tokenizer)}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
