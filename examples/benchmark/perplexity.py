import argparse
import os

import torch
from gptqmodel.utils import Perplexity
from transformers import AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":
    """
    Example usage.

    Default usage with GPT2 model:
    python examples/benchmark/perplexity.py

    Specify GPTQ quantized model:
    python examples/benchmark/perplexity.py \
        --model_name LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit \
        --is_quantized \
        --backend AUTO

    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name.")
    parser.add_argument("--model_basename", type=str, default=None, help="Model file's basename.")
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
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
    parser.add_argument("--backend", choices=['AUTO', 'TRITON', 'EXLLAMA_V2', 'MARLIN', 'CUDA', 'BITBLAS', 'IPEX'], help="Whether to use BACKEND format")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.is_quantized:
        from gptqmodel import GPTQModel, BACKEND

        model = GPTQModel.load(
            args.model_name,
            device_map="auto",
            model_basename=args.model_basename,
            trust_remote_code=args.trust_remote_code,
            backend=BACKEND(args.backend.lower()),
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
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
