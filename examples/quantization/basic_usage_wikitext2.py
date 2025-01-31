# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from transformers import AutoTokenizer

pretrained_model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0" # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
quantized_model_id = "TinyLlama-1.1B-Chat-v1.0-4bit-128g"


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
    model = GPTQModel.load(pretrained_model_id, quantize_config)

    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset)

    # save quantized model
    model.save(quantized_model_id)

    # save quantized model using safetensors
    model.save(quantized_model_id)

    # load quantized model, currently only support cpu or single gpu
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = GPTQModel.load(quantized_model_id, device=device)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("test is", return_tensors="pt").to(device))[0]))

    print(f"Quantized Model {quantized_model_id} avg PPL is {calculate_avg_ppl(model, tokenizer)}")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
