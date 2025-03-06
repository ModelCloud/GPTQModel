# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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
from gptqmodel import BACKEND
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


from gptqmodel import GPTQModel  # noqa: E40
from gptqmodel.adapter.adapter import Adapter, AdapterCache, Lora

def benchmark_full(full_precision_model):

    MODEL_id = full_precision_model
    MIN_NEW_TOKENS = 10
    MAX_NEW_TOKENS = 20
    NUM_RUNS = 50
    PROMPTS = [
        "I am in Paris and I",
        "The capital of the United Kingdom is",
        "The largest ocean on Earth is",
        "The world’s longest river is",
        "The tallest mountain in the world is",
        "The currency used in Japan is",
        "How to consult a dictionary?",
        "What is the boiling point of water in degrees Celsius?",
        "Which is the most widely used Internet search engine in the world?",
        "What is the official language of France?",
    ]
    MAX_DELTA_FLOOR_PERCENT = 0.25
    MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0

    model = AutoModelForCausalLM.from_pretrained(
        full_precision_model,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(full_precision_model)

    tokenizer.pad_token = tokenizer.eos_token

    inp = tokenizer(PROMPTS, padding=True, padding_side="left", pad_to_multiple_of=16, truncation=True, return_tensors="pt",).to("cuda")
    
    warmup_iter = 100
    print(f"Warming up: warmup_iter = `{warmup_iter}`")
    for i in range(warmup_iter):
        _ = model.generate(**inp, min_new_tokens=MIN_NEW_TOKENS,
                            max_new_tokens=MAX_NEW_TOKENS)

    times = []

    for _ in range(NUM_RUNS):
        start_time = time.time()
        _ = model.generate(**inp,min_new_tokens=MIN_NEW_TOKENS,
                                max_new_tokens=MAX_NEW_TOKENS)
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    sum_time = sum(times)
    sum_tokens = len(PROMPTS) * MIN_NEW_TOKENS * NUM_RUNS
    avg_tokens_per_second = sum_tokens / sum_time

    print("**************** Benchmark Result Info****************")
    print(f"Times: {times}")
    print(f"Sum Times: {sum_time}")
    print(f"Sum New Tokens: {sum_tokens}")
    print(f"Benchmark Result: {avg_tokens_per_second} token/s")
    print("**************** Benchmark Result Info End****************")

def benchmark_GPTQModel(quantized_model):

    MODEL_id = quantized_model
    MIN_NEW_TOKENS = 10
    MAX_NEW_TOKENS = 20
    NUM_RUNS = 50
    PROMPTS = [
        "I am in Paris and I",
        "The capital of the United Kingdom is",
        "The largest ocean on Earth is",
        "The world’s longest river is",
        "The tallest mountain in the world is",
        "The currency used in Japan is",
        "How to consult a dictionary?",
        "What is the boiling point of water in degrees Celsius?",
        "Which is the most widely used Internet search engine in the world?",
        "What is the official language of France?",
    ]
    MAX_DELTA_FLOOR_PERCENT = 0.25
    MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0

    model = GPTQModel.load(
            MODEL_id,
            device="cuda",
            backend=BACKEND.EXLLAMA_EORA,
            use_cache=False,
        )

    ## what is this for??/
    model.optimize()

    tokenizer = model.tokenizer
    inp = tokenizer(PROMPTS, padding=True, padding_side="left", pad_to_multiple_of=16, truncation=True, return_tensors="pt",).to("cuda")
    
    warmup_iter = 100
    print(f"Warming up: warmup_iter = `{warmup_iter}`")
    for i in range(warmup_iter):
        _ = model.generate(**inp, min_new_tokens=MIN_NEW_TOKENS,
                            max_new_tokens=MAX_NEW_TOKENS)

    times = []

    for _ in range(NUM_RUNS):
        start_time = time.time()
        _ = model.generate(**inp,min_new_tokens=MIN_NEW_TOKENS,
                                max_new_tokens=MAX_NEW_TOKENS)
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    sum_time = sum(times)
    sum_tokens = len(PROMPTS) * MIN_NEW_TOKENS * NUM_RUNS
    avg_tokens_per_second = sum_tokens / sum_time

    print("**************** Benchmark Result Info****************")
    print(f"Times: {times}")
    print(f"Sum Times: {sum_time}")
    print(f"Sum New Tokens: {sum_tokens}")
    print(f"Benchmark Result: {avg_tokens_per_second} token/s")
    print("**************** Benchmark Result Info End****************")

def benchmark_EORA(quantized_model, eora_path, eora_rank):

    MODEL_id =  quantized_model
    EORA_PATH = eora_path

    MIN_NEW_TOKENS = 10
    MAX_NEW_TOKENS = 20
    NUM_RUNS = 50
    PROMPTS = [
        "I am in Paris and I",
        "The capital of the United Kingdom is",
        "The largest ocean on Earth is",
        "The world’s longest river is",
        "The tallest mountain in the world is",
        "The currency used in Japan is",
        "How to consult a dictionary?",
        "What is the boiling point of water in degrees Celsius?",
        "Which is the most widely used Internet search engine in the world?",
        "What is the official language of France?",
    ]
    MAX_DELTA_FLOOR_PERCENT = 0.25
    MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0

    ## Load EoRA Here:
    eora = Lora(
            rank=eora_rank,
            path=EORA_PATH)

    model = GPTQModel.load(MODEL_id, backend=BACKEND.EXLLAMA_EORA, adapter=eora)
    ## what is this for??/
    model.optimize()

    tokenizer = model.tokenizer
    inp = tokenizer(PROMPTS, padding=True, padding_side="left", pad_to_multiple_of=16, truncation=True, return_tensors="pt",).to("cuda")
    
    warmup_iter = 100
    print(f"Warming up: warmup_iter = `{warmup_iter}`")
    for i in range(warmup_iter):
        _ = model.generate(**inp, min_new_tokens=MIN_NEW_TOKENS,
                            max_new_tokens=MAX_NEW_TOKENS)

    times = []

    for _ in range(NUM_RUNS):
        start_time = time.time()
        _ = model.generate(**inp,min_new_tokens=MIN_NEW_TOKENS,
                                max_new_tokens=MAX_NEW_TOKENS)
        end_time = time.time()

        elapsed_time = end_time - start_time
        times.append(elapsed_time)

    sum_time = sum(times)
    sum_tokens = len(PROMPTS) * MIN_NEW_TOKENS * NUM_RUNS
    avg_tokens_per_second = sum_tokens / sum_time

    print("**************** Benchmark Result Info****************")
    print(f"Times: {times}")
    print(f"Sum Times: {sum_time}")
    print(f"Sum New Tokens: {sum_tokens}")
    print(f"Benchmark Result: {avg_tokens_per_second} token/s")
    print("**************** Benchmark Result Info End****************")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--quantized_model', type=str,
        help='Quantized model to load; pass'
    )
    parser.add_argument(
        '--full_precision_model', type=str, default=None
    )
    parser.add_argument(
        '--eora_save_path',type=str, default=None
    )
    parser.add_argument(
        '--eora_rank', type=int
    )

    args = parser.parse_args()

    if args.eora_save_path:
        benchmark_EORA(quantized_model=args.quantized_model, eora_path=args.eora_save_path, eora_rank=args.eora_rank)
    elif args.full_precision_model:
        benchmark_full(full_precision_model=args.full_precision_model)
    else:
        benchmark_GPTQModel(quantized_model=args.quantized_model)

