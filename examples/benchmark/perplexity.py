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

import argparse
import os

from gptqmodel.utils import Perplexity
from transformers import AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if __name__ == "__main__":
    """
    Example usage:

    python examples/benchmark/perplexity.py \
        --model ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5 \
        --is_quantized

    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model", type=str, default="ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5", help="Model name.")
    parser.add_argument("--n_ctx", type=int, default=1024, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=1024, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column in the dataset containing the text.",
    )
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument("--backend", choices=['auto', 'marlin', 'exllama_v1', 'exllama_v2', 'triton', 'cuda', 'torch', 'ipex', 'bitblas'], default='auto', help="Whether to use BACKEND format")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=args.use_fast_tokenizer)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.is_quantized:
        from gptqmodel import BACKEND, GPTQModel

        model = GPTQModel.load(
            args.model,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            backend=BACKEND(args.backend.lower()),
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=args.trust_remote_code,
        )

    ppl = Perplexity(
        model,
        tokenizer,
        args.dataset_path,
        args.dataset_name,
        args.split,
        args.text_column,
    )
    ppl.calculate(args.n_ctx, args.n_batch)
