import time
import torch
import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Benchmark IPEX vs HF on a pre-trained model.")
parser.add_argument("--model", type=str, required=True, help="Path or name of the pre-trained model.")
parser.add_argument("--cores", type=int, default=8, help="Number of CPU cores to use.")
parser.add_argument("--batch", type=int, default=2, help="Batch size for processing messages.")
parser.add_argument("--backend", type=str, choices=["ipex", "hf"], default="ipex", help="Backend to optimize the model. Choose between 'ipex' and 'hf'.")
ars = parser.parse_args()

# Set the number of threads to use
torch.set_num_threads(ars.cores)
print("use torch thread:", torch.get_num_threads())

tokenizer = AutoTokenizer.from_pretrained(ars.model)
model = AutoModelForCausalLM.from_pretrained(ars.model, device_map="cpu", torch_dtype=torch.bfloat16)
model.eval()

if ars.backend == "ipex":
    import intel_extension_for_pytorch as ipex
    model = ipex.llm.optimize(model, dtype=torch.bfloat16)

print(f"use backend: {ars.backend} load model: {ars.model}")

tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>
model.pad_token_id = tokenizer.pad_token_id

messages = [
    {"role": "user", "content": "How can I effectively learn programming?"},
    {"role": "user", "content": "How to choose and care for different types of fabrics?"},
    {"role": "user", "content": "How do creation myths vary across religions?"},
    {"role": "user", "content": "Name a traditional dance from Greece."},
    {"role": "user", "content": "Describe the process of cellular respiration."},
    {"role": "user", "content": "How has mobile gaming changed the industry?"},
    {"role": "user", "content": "What are the benefits of using a standing desk?"},
    {"role": "user", "content": "Provide a detailed explanation of how multiplayer netcode works in online games?"},
    {"role": "user", "content": "What are the key differences between renewable and non-renewable energy sources?"},
    {"role": "user", "content": "How can I improve my time management skills effectively?"},
    {"role": "user", "content": "What are the steps involved in baking a perfect sourdough bread?"},
    {"role": "user", "content": "Explain the significance of the Renaissance period in European history."},
    {"role": "user", "content": "What are the primary causes and effects of deforestation worldwide?"},
    {"role": "user", "content": "How do plants adapt to survive in extreme desert environments?"},
    {"role": "user", "content": "What strategies can help reduce stress during exam preparation?"},
    {"role": "user", "content": "Describe the cultural importance of tea ceremonies in Japan."},
]

print(f"total messages: {len(messages)}, batch_size: {ars.batch}")
if ars.batch > len(messages):
    raise ValueError("batch_size should be less than or equal to the number of messages")

messages = [[messages[i]] for i in range(ars.batch)]

with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
    start = time.time()
    new_token_len = 0
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", padding=True)

    outputs = model.generate(input_ids=input_tensor.to(model.device), max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
    for i, output in enumerate(outputs):
        new_token_len += len(output) - len(input_tensor[i])
        # debug print
        # result = tokenizer.decode(output, skip_special_tokens=False)
        # print(result)
        # print("="*50)

    total_time = time.time() - start
    print(f"generate use :{total_time}")
    print(f"total new token: {new_token_len}")
    print(f"token/sec: {(new_token_len/total_time):.4f}")