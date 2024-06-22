import threading
import time
from argparse import ArgumentParser

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel

model_name_or_path = "/monster/data/model/Meta-Llama-3-8B-Instruct/quant/2024-06-18_05-58-43_anaerobation_A100_auto_autogptq_version_pr640_bit4_group128_seq2048_batch1/damp0.005_descFalse_gptq_symTrue_pack_dataFalse/wikitext21419_gr0_dic0_sen0_det0_rate0"


def load_model(disable_exllama, disable_exllamav2, use_marlin):
    model = GPTQModel.from_quantized(
        model_name_or_path,
        use_cuda_fp16=True,
        quantize_config={
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": False,
        },
        trust_remote_code=True,
        device="cuda:0",
        disable_exllama=disable_exllama,
        disable_exllamav2=disable_exllamav2,
        use_marlin=use_marlin,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )

    return model, tokenizer


def load_datasets(tokenizer, prompt_length):
    data_dict = load_dataset("ModelCloud/alpaca-data-cleaned", data_files="alpaca_data_cleaned.json", split="train")

    datas = [
        {
            'input': item['input'],
            'output': item['output'],
            'instruction': item['instruction']
        }
        for item in data_dict
    ]

    def dummy_gen():
        return datas

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]

        prompts = []

        for istr, inp in zip(instructions, inputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
            t = tokenizer(prompt)
            if len(t["input_ids"]) < prompt_length:
                continue

            prompts.append(tokenizer.decode(t["input_ids"][:prompt_length]))

        return prompts

    dataset = Dataset.from_generator(dummy_gen)

    dataset = tokenize(dataset)

    return dataset


lock = threading.Lock()
sum_token = 0

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--disable_exllama", action="store_true", default=False)
    parser.add_argument("--disable_exllamav2", action="store_true", default=False)
    parser.add_argument("--use_marlin", action="store_true", default=False)
    parser.add_argument("--prompt_length", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    model, tokenizer = load_model(args.disable_exllama, args.disable_exllamav2, args.use_marlin)
    data = load_datasets(tokenizer, args.prompt_length)[3:4]  # only 4 items have prompt size > 512 in this dataset
    # data.append(data)  # double it

    start_time = time.time()


    def generate(d, batch):
        global sum_token
        generated_tokens = model.generate(**tokenizer(d, return_tensors="pt").to("cuda:0"), max_new_tokens=1024)[0]

        generated_size = len(generated_tokens)

        # out = tokenizer.decode(generated_tokens)

        with lock:
            sum_token += generated_size

        # time_usage = time.time() - start_time
        # print(f"time: {time_usage:.3f}, size: {generated_size}, speed: {sum_token / time_usage:.3f}")
        return generated_size


    for d in data:
        if args.batch == 1:
            generated_size = generate(d, 1)
            sum_token += generated_size
        else:
            threads = []
            for _ in range(args.batch):
                thread = threading.Thread(target=generate, args=(d, args.batch))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

    total_time = time.time() - start_time
    print(f"total time: {total_time:.3f}, speed: {(sum_token / total_time):.3f} | {args}")
