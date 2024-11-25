import time
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from gptqmodel import GPTQModel, BACKEND

def prepare_dataset_for_bench(tokenizer, batch_size=8):
    dataset = load_dataset("json", data_files="prompts.jsonl", split="train")[:batch_size]
    prompts = [[{"role": "user", "content": data}] for data in dataset['input']]
    input_tensors = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, return_tensors="pt", padding=True)
    return input_tensors

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

# Set the number of threads to use
torch.set_num_threads(ars.cores)

# load model, check model backend
config = AutoConfig.from_pretrained(ars.model)
is_quantized_model = hasattr(config, "quantization_config")
if is_quantized_model:
    print("load quantized model, will use BACKEND.IPEX")
    model = GPTQModel.load(ars.model, backend=BACKEND.IPEX)
    model.to(torch.device("cpu"))
else:
    model = AutoModelForCausalLM.from_pretrained(ars.model, device_map="cpu", torch_dtype=torch.bfloat16)

# set model to eval mode
model.eval()

if ars.backend == "ipex" and not is_quantized_model:
    import intel_extension_for_pytorch as ipex
    model = ipex.llm.optimize(model, dtype=torch.bfloat16)
    model.forward = torch.compile(model.forward, dynamic=True, backend="ipex")

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