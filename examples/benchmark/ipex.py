import time
import torch
torch.set_num_threads(8)
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import intel_extension_for_pytorch as ipex

model_name = "/monster/data/model/Meta-Llama-3.1-8B-Instruct/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.bfloat16)
model.eval()

model = ipex.llm.optimize(model, dtype=torch.bfloat16)

tokenizer.pad_token_id = 128004 # <|finetune_right_pad_id|>
model.pad_token_id = tokenizer.pad_token_id

message1 = [
    {"role": "user", "content": "How can I effectively learn programming?"},
]

message2 = [
    {"role": "user", "content": "Discuss the impact of virtual reality on modern gaming."},
]

with torch.no_grad(), torch.amp.autocast("cpu", dtype=torch.bfloat16):
    start = time.time()
    new_token_len = 0
    input_tensor = tokenizer.apply_chat_template([message1, message2], add_generation_prompt=True, return_tensors="pt", padding=True)

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

if __name__ == '__main__':
    pass
