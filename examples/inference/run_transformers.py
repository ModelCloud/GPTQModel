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

from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ")
quantized_model = AutoModelForCausalLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ")
print(tokenizer.decode(quantized_model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(quantized_model.device))[0]))

