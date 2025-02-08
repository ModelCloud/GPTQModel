import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

# from gptqmodel.eora import get_eora, get_eora_optimize


bit = 4
model_id = "meta-llama/Llama-3.2-1B"
model = None

quant_path = "Llama-3.2-1B-gptqmodel-4bit"
fake_quant_path = "Llama-3.2-1B-gptqmodel-4bit-fakequantized/qw.pt"
eora_path = "Llama-3.2-1B-gptqmodel-4bit-eora-rank-128-v2/eora.pt"
quant_config = QuantizeConfig(bits=bit, group_size=128)


calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train"
).select(range(1024))["text"]

print(f"{type(calibration_dataset)}")

### 3-bit group_size = 128 leads to out: IndexError: index 192 is out of bounds when packing
model = GPTQModel.load(model_id, quant_config)

# increase `batch_size` to match gpu/vram specs to speed up quantization
quant_log, quantized_weights = model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)

torch.save(quantized_weights, fake_quant_path)
quantized_weights = torch.load(fake_quant_path, map_location='cpu')

## 4-bit gs=128 Acc: 0.2850

batch_size = 2
from test_prepare_dataset import construct_ARC

calibration_dataset = construct_ARC(nsamples=1024)
eora_rank = 128
model = GPTQModel.load(model_id, quant_config)

eora_weight = model.get_eora(calibration_dataset, batch_size, quantized_weights, eora_rank)

torch.save(eora_weight, eora_path)

eora_weight = torch.load(eora_path,  map_location='cpu')
print(eora_weight)
