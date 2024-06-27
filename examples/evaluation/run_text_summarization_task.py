import os
from argparse import ArgumentParser

import datasets
import torch
from gptqmodel import GPTQModel, QuantizeConfig, get_backend
from gptqmodel.eval_tasks import TextSummarizationTask
from transformers import AutoTokenizer, GenerationConfig

os.system("pip install py7zr")


DATASET = "samsum"
TEMPLATE = "Instruction: Summarize the conversation into one sentence.\n\nInput:\n{diag}\n\nOutput:\n"


def ds_refactor_fn(samples):
    dialogues = samples["dialogue"]

    new_samples = {"prompt": [], "summary": samples["summary"]}
    for diag in dialogues:
        prompt = TEMPLATE.format(diag=diag)
        new_samples["prompt"].append(prompt)

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
    parser.add_argument("--backend", choices=['AUTO', 'CUDA_OLD', 'CUDA', 'TRITON', 'EXLLAMA', 'EXLLAMA_V2', 'MARLIN', 'BITBLAS'])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)

    model = GPTQModel.from_pretrained(args.base_model_dir, QuantizeConfig())
    model.to("cuda:0")

    task = TextSummarizationTask(
        model=model,
        tokenizer=tokenizer,
        data_name_or_path=DATASET,
        prompt_col_name="prompt",
        label_col_name="summary",
        **{
            "num_samples": args.num_samples,  # how many samples will be sampled to evaluation
            "sample_max_len": args.sample_max_len,  # max tokens for each sample
            "block_max_len": args.block_max_len,  # max tokens for each data block
            "load_fn": datasets.load_dataset,  # function to load dataset
            "preprocess_fn": ds_refactor_fn,  # function to preprocess dataset
            "truncate_prompt": False,  # truncate label when sample's length exceed sample_max_len
        },
    )

    print(f"eval result for base model: {task.run(generation_config=GenerationConfig(max_new_tokens=32))}")
    task.model = None
    model.cpu()
    del model
    torch.cuda.empty_cache()

    model = GPTQModel.from_quantized(args.quantized_model_dir, device="cuda:0", backend=get_backend(args.backend))
    task.model = model
    task.device = model.device
    print(f"eval result for quantized model: {task.run(generation_config=GenerationConfig(max_new_tokens=32))}")


if __name__ == "__main__":
    main()
