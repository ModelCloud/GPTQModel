# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = ["gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", quantization_config=gptq_config)
quantized_model.save_pretrained("./opt-125m-gptq")
tokenizer.save_pretrained("./opt-125m-gptq")

model = AutoModelForCausalLM.from_pretrained("./opt-125m-gptq", device_map="auto")

print(tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0]))
