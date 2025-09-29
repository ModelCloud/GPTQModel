# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ")
quantized_model = AutoModelForCausalLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ")
print(tokenizer.decode(quantized_model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(quantized_model.device))[0]))

