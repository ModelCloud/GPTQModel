# Copyright 2025 ModelCloud
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
