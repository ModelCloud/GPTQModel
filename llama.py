from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "meta-llama/Meta-Llama-3-8B"
quant_path = "Llama-3-8B-gptqmodel-4bit"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
quant_log, quantized_weights = model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)

# test post-quant inference
model = GPTQModel.load(quant_path)
result = model.generate("Uncovering deep insights begins with")[0]

# improve downstream task accuracy using EoRA
eora = True
if eora:
    # Construct the calibration dataset for EoRA
    # 
    # reset the model
    print("server down")