from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ")
quantized_model = AutoModelForCausalLM.from_pretrained("TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ")
print(tokenizer.decode(quantized_model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(quantized_model.device))[0]))

