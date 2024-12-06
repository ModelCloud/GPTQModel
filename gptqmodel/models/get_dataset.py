import itertools
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_all_combinations():
    step = 0.25
    split_points = [i * step for i in range(int(1 / step) + 1)]  # [0.0, 0.25, 0.5, 0.75, 1.0]

    all_combinations = [
        combo for combo in itertools.product(split_points, repeat=3) if combo != (0, 0, 0)
    ]
    return all_combinations #[native, code, nm]


MODEL_PATH="/monster/data/cl/cl-QwQ-32B-Preview-qwq-32b-preview-pack1-lr2.2e-05-cosine-bs4--mtl1088-seq2048-thrusher-04_11-53-30/checkpoint-36"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_lm_compression_data(tokenizer, nsamples, seqlen):
    traindata = load_dataset("neuralmagic/LLM_compression_calibration", split="train")

    ds = traindata
    traindataset = []
    for example in ds:
        if len(traindataset) == nsamples:
            break

        if example["text"] and len(example["text"].strip()) > 0:
            data = tokenizer(
                example["text"], max_length=seqlen, truncation=True, return_tensors='pt'
            )

            inp = data.input_ids
            attention_mask = torch.ones_like(inp)

            traindataset.append({'input_ids': inp.squeeze(), 'attention_mask': attention_mask.squeeze()})

    if len(traindataset) != nsamples:
        raise ValueError(f"Requested {nsamples} wikitext2 samples, but only {len(traindataset)} samples are available.")

    return traindataset

def _print_first_prompt(type, prompt):
    print(f"\n--- First {type} Prompt Start---")
    print(prompt)
    print(f"--- First {type} Prompt End---\n")

def _gen_input_ids_and_attention_mask(tokenizer, format_data):
    trainenc = tokenizer(format_data, return_tensors='pt')
    inp = trainenc.input_ids.squeeze()
    attention_mask = torch.ones_like(inp).squeeze()
    return inp, attention_mask

def get_native_outputs(model_path, native_outputs_name, tokenizer, nsamples):
    if nsamples == 0:
        return []
    from datasets import load_dataset

    native_data_path = os.path.join(model_path, native_outputs_name)

    traindata = load_dataset("json", data_files=native_data_path, split='train')

    traindataset = []
    for i in range(0, nsamples):
        if i == 0:
            _print_first_prompt("Native", traindata[i]['output'])

        inp, attention_mask = _gen_input_ids_and_attention_mask(tokenizer, traindata[i]['output'])
        traindataset.append({'input_ids': inp, 'attention_mask': attention_mask})

    if len(traindataset) != nsamples:
        raise ValueError(f"Requested {nsamples} rate samples, but only {len(traindataset)} samples are available.")

    return traindataset

def count_jsonl_items(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)

def get_data_set(native_percent, code_percent, nm_percent):
    total_native_data_size = count_jsonl_items(os.path.join(MODEL_PATH, "native_outputs.jsonl"))
    total_code_data_size = count_jsonl_items(os.path.join(MODEL_PATH, "native_code_outputs.jsonl"))
    print("total_native_data_size",total_native_data_size,total_code_data_size)
    total_nm_data_size = 1000

    native_data = get_native_outputs(MODEL_PATH, "native_outputs.jsonl", tokenizer, int(total_native_data_size * native_percent))
    code_data = get_native_outputs(MODEL_PATH, "native_code_outputs.jsonl", tokenizer, int(total_code_data_size * code_percent))
    nm_data = get_lm_compression_data(tokenizer, int(total_nm_data_size * nm_percent), 2048)
    print("get_data_set", len(native_data), len(code_data), len(nm_data))
    return native_data + code_data + nm_data
