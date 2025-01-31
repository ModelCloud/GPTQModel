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

from argparse import ArgumentParser
from functools import partial

import datasets
import torch
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.eval_tasks import SequenceClassificationTask
from gptqmodel.utils.torch import torch_empty_cache
from transformers import AutoTokenizer

DATASET = "cardiffnlp/tweet_sentiment_multilingual"
TEMPLATE = "Question:What's the sentiment of the given text? Choices are {labels}.\nText: {text}\nAnswer:"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABELS = list(ID2LABEL.values())


def ds_refactor_fn(samples):
    text_data = samples["text"]
    label_data = samples["label"]

    new_samples = {"prompt": [], "label": []}
    for text, label in zip(text_data, label_data):
        prompt = TEMPLATE.format(labels=LABELS, text=text)
        new_samples["prompt"].append(prompt)
        new_samples["label"].append(ID2LABEL[label])

    return new_samples


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="how many samples will be sampled to evaluation",
    )
    parser.add_argument("--sample_max_len", type=int, default=1024, help="max tokens for each sample")
    parser.add_argument("--block_max_len", type=int, default=2048, help="max tokens for each data block")
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA_V2', 'MARLIN', 'CUDA', 'BITBLAS', 'IPEX'])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)

    model = GPTQModel.load(args.base_model_dir, QuantizeConfig())
    device = "cpu" if not torch.cuda.is_available() or args.backend == "IPEX" else "cuda:0"
    model.to(device)

    task = SequenceClassificationTask(
        model=model,
        tokenizer=tokenizer,
        classes=LABELS,
        data_name_or_path=DATASET,
        prompt_col_name="prompt",
        label_col_name="label",
        **{
            "num_samples": args.num_samples,  # how many samples will be sampled to evaluation
            "sample_max_len": args.sample_max_len,  # max tokens for each sample
            "block_max_len": args.block_max_len,  # max tokens for each data block
            "load_fn": partial(datasets.load_dataset, name="english"),  # function to load dataset
            "preprocess_fn": ds_refactor_fn,  # function to preprocess dataset
            "truncate_prompt": False,  # truncate label when sample's length exceed sample_max_len
        },
    )

    print(f"eval result for base model: {task.run()}")
    task.model = None
    model.cpu()
    del model
    torch_empty_cache()

    model = GPTQModel.from_quantized(args.quantized_model_dir, device=device, backend=BACKEND(args.backend.lower()))
    task.model = model
    task.device = model.device
    print(f"eval result for quantized model: {task.run()}")


if __name__ == "__main__":
    main()
