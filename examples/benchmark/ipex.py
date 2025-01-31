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

import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from optimum.intel.utils.modeling_utils import bind_cores_for_best_perf
    bind_cores_for_best_perf()
except ImportError as e:
    print(f"Error: {e}\nCannot Bind process to Single NUMA Region/Socket. Please check permissions as this may reduce performance by ~10%")
    pass

import argparse

parser = argparse.ArgumentParser(description="Benchmark IPEX vs HF on a pre-trained model.")
parser.add_argument("--model", type=str, required=True, help="Path or name of the pre-trained model.")
parser.add_argument("--cores", type=int, default=8, help="Number of CPU cores to use.")
parser.add_argument("--batch", type=int, default=8, help="Batch size for processing messages.")
parser.add_argument("--backend", type=str, choices=["ipex", "hf"], default="ipex", help="Backend to optimize the model. Choose between 'ipex' and 'hf'.")
ars = parser.parse_args()

print("use model: ", ars.model)
print("use cores: ", ars.cores)
print("use batch: ", ars.batch)
print("use backend: ", ars.backend)

# Set the "OMP_NUM_THREADS" environment variable to the specified number of cores to control the number of threads used by OpenMP
os.environ["OMP_NUM_THREADS"] = str(ars.cores)


# Set the number of threads used by PyTorch
torch.set_num_threads(ars.cores)


def prepare_dataset_for_bench(tokenizer, batch_size=8):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files="prompts.jsonl", split="train")[:batch_size]
    prompts = [[{"role": "user", "content": data}] for data in dataset['input']]
    input_tensors = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, return_tensors="pt", padding=True)
    print(input_tensors.shape)
    return input_tensors



# load model, check model backend
start_load = time.time()
config = AutoConfig.from_pretrained(ars.model)
is_quantized_model = hasattr(config, "quantization_config")
if is_quantized_model:
    from gptqmodel import BACKEND, GPTQModel

    print("load quantized model, will use BACKEND.IPEX")
    model = GPTQModel.load(ars.model, backend=BACKEND.IPEX)
else:
    model = AutoModelForCausalLM.from_pretrained(ars.model, device_map="cpu", torch_dtype=torch.bfloat16)
print(f"load model use: {time.time() - start_load}")

# set model to eval mode
model.eval()

if ars.backend == "ipex" and not is_quantized_model:
    start_opt = time.time()
    import intel_extension_for_pytorch as ipex
    model = ipex.llm.optimize(model, dtype=torch.bfloat16)
    model.forward = torch.compile(model.forward, dynamic=True, backend="ipex")
    print(f"use ipex optimize model use: {time.time() - start_opt}")


# load tokenizer, and normalize the pad_token_id
tokenizer = AutoTokenizer.from_pretrained(ars.model)
tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>
model.pad_token_id = tokenizer.pad_token_id


# prepare dataset for benchmark
input_ids = prepare_dataset_for_bench(tokenizer, ars.batch)


# benchmark
with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
    start = time.time()
    new_token_len = 0
    outputs = model.generate(input_ids=input_ids.to(model.device), max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
    for i, output in enumerate(outputs):
        new_token_len += len(output) - len(input_ids[i])
        # debug print
        # result = tokenizer.decode(output, skip_special_tokens=False)
        # print(result)
        # print("="*50)

    total_time = time.time() - start


# display benchmark result
print(f"generate use :{total_time}")
print(f"total new token: {new_token_len}")
print(f"token/sec: {(new_token_len/total_time):.4f}")
