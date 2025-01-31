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

import logging
import random
import time
from argparse import ArgumentParser
from itertools import chain
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.utils.progress import ProgressBar
from transformers import AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import LogitsProcessor

logger = logging.getLogger(__name__)

random.seed(0)

class CustomizedMinNewTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        min_new_tokens: int = None,
        eos_token_id: int = None,
    ):
        self.eos_token_id = eos_token_id
        self.min_new_tokens = min_new_tokens or 0
        self.current_step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.current_step += 1

        if self._skip_process():
            return scores

        if any(each is not None for each in [self.eos_token_id]):
            banned_mask = torch.zeros_like(scores).to(scores.device)
            if self.eos_token_id and self.current_step <= self.min_new_tokens:
                banned_mask = self._fill_banned_mask(input_ids, banned_mask, {1: [[self.eos_token_id]]})
            scores = scores.masked_fill(banned_mask.bool(), -float("inf"))

        return scores

    def _skip_process(self):
        if self.current_step > self.min_new_tokens:
            return True
        return False

    @staticmethod
    def _fill_banned_mask(
        input_ids: torch.LongTensor,
        banned_mask: torch.Tensor,
        len2words_ids: Dict[int, List[List[int]]],
    ):
        for token_len, token_ids in len2words_ids.items():
            if token_len == 1:
                banned_mask[..., list(chain(*token_ids))] = 1
            elif input_ids.shape[-1] < token_len - 1:
                continue
            else:
                token_ids = torch.LongTensor(token_ids).to(input_ids.device)
                hit_masks = torch.all(
                    token_ids[..., :-1].unsqueeze(0).repeat(input_ids.shape[0], 1, 1)
                    == input_ids[..., -(token_ids.shape[-1] - 1) :].unsqueeze(1),
                    dim=-1,
                )
                for idx in range(hit_masks.shape[0]):
                    selected_token_ids = torch.masked_select(token_ids[..., -1], hit_masks[idx])
                    if len(selected_token_ids):
                        banned_mask[idx, selected_token_ids] = 1
        return banned_mask


def load_data(tokenizer, n_samples, max_new_tokens):
    data_dict = load_dataset("ModelCloud/alpaca-data-cleaned", data_files="alpaca_data_cleaned.json", split="train")

    datas = [
        {
            'input': item['input'],
            'output': item['output'],
            'instruction': item['instruction']
        }
        for item in data_dict
    ]

    raw_data = random.sample(datas, k=min(n_samples, len(datas)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length - max_new_tokens:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def load_model_tokenizer(
    model_id_or_path: str,
    backend: BACKEND,
    tokenizer_name_or_path: Optional[str] = None,
    from_pretrained: bool = False,
    model_basename: Optional[str] = None,
    quantize_config: Optional[str] = None,
    trust_remote_code: bool = False,
    use_fast_tokenizer: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name_or_path or model_id_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if from_pretrained:
        model = GPTQModel.load(
            pretrained_model_id_or_path=model_id_or_path,
            quantize_config=QuantizeConfig(),
            trust_remote_code=trust_remote_code,
        )
    else:
        model = GPTQModel.load(
            model_id_or_path,
            quantize_config=quantize_config,
            model_basename=model_basename,
            trust_remote_code=trust_remote_code,
            backend=backend,
        )

    return model, tokenizer


def benchmark_generation_speed(model, tokenizer, examples, generation_config):
    generation_time_list = []
    num_generated_tokens_list = []
    progress_bar = ProgressBar(examples)
    for example in progress_bar:
        input_ids = example["input_ids"].to(model.device)

        start = time.time()
        outputs_ids = model.generate(
            input_ids=input_ids.unsqueeze(0),
            generation_config=generation_config,
            logits_processor=[
                CustomizedMinNewTokensLogitsProcessor(generation_config.max_new_tokens, tokenizer.eos_token_id)
            ],
        )
        end = time.time()

        generation_time_list.append(end - start)
        num_generated_tokens = 0
        for output_ids in outputs_ids:
            num_generated_tokens += len(
                [token_id for token_id in output_ids[len(input_ids) :] if token_id != tokenizer.pad_token_id]
            )
        num_generated_tokens_list.append(num_generated_tokens)

        progress_bar.set_postfix(
            num_tokens=num_generated_tokens_list[-1],
            time=generation_time_list[-1],
            speed=f"{num_generated_tokens_list[-1] / generation_time_list[-1]:.3f} tokens/s",
        )

    total_tokens = sum(num_generated_tokens_list)
    total_seconds = sum(generation_time_list)
    logger.info(
        f"generated {total_tokens} tokens using {total_seconds:.3f} seconds, "
        f"generation speed: {total_tokens / total_seconds:.3f} tokens/s"
    )

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_id_or_path", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--from_pretrained", action="store_true")
    parser.add_argument("--model_basename", type=str, default=None)
    parser.add_argument("--quantize_config_save_dir", type=str, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA_V2', 'MARLIN', 'CUDA', 'BITBLAS', 'IPEX'])
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    quantize_config = None
    if args.quantize_config_save_dir:
        quantize_config = QuantizeConfig.from_pretrained(args.quantize_config_save_dir)

    logger.info("loading model and tokenizer")
    start = time.time()
    model, tokenizer = load_model_tokenizer(
        model_id_or_path=args.model_id_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        from_pretrained=args.from_pretrained,
        model_basename=args.model_basename,
        quantize_config=quantize_config,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=args.use_fast_tokenizer,
        backend=BACKEND(args.backend.lower()),
    )
    end = time.time()
    logger.info(f"model and tokenizer loading time: {end - start:.4f}s")
    logger.info(f"model quantized: {model.quantized}")
    logger.info(f"quantize config: {model.quantize_config.to_dict()}")
    logger.info(f"model device map: {model.hf_device_map}")
    logger.info("loading data")
    examples = load_data(
        tokenizer,
        args.num_samples,
        args.max_new_tokens,
    )

    device = "cpu" if not torch.cuda.is_available() or args.backend == "IPEX" else "cuda:0"
    model.to(device)

    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        do_sample=args.do_sample,
        min_new_tokens=args.max_new_tokens,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    logger.info(f"generation config: {generation_config.to_dict()}")

    logger.info("benchmark generation speed")
    benchmark_generation_speed(model, tokenizer, examples, generation_config)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
