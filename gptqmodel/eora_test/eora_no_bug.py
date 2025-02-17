import os

import safetensors
import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.adapter.adapter import Lora

# from gptqmodel.eora_test import get_eora, get_eora_optimize


bit = 4
model_id = "meta-llama/Llama-3.2-1B"
model = None

quant_path = "/root/projects/GPTQModel/Llama-3.2-1B-gptqmodel-4bit"
fake_quant_path = "../../Llama-3.2-1B-gptqmodel-4bit-fakequantized/qw.pt"
eora_path = "Llama-3.2-1B-gptqmodel-4bit-eora-rank-128-v2/"
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
model.quantize(calibration_dataset, batch_size=2)

model.save(quant_path)

## 4-bit gs=128 Acc: 0.2850

batch_size = 2
from test_prepare_dataset import construct_ARC

calibration_dataset = construct_ARC(nsamples=1024)
lora_rank = 128

eora = Lora(
    # for quant, path is save path. for load, it is loading path
    path=os.path.join(eora_path, "lora_adapter.safetensors"),
    rank=lora_rank,
)

GPTQModel.eora_generate(model_id_or_path=model_id, quantized_model_id_or_path=quant_path, adapter=eora,
                        calibration_dataset=calibration_dataset, batch_size=batch_size)
eora_weight = safetensors.torch.load_file(os.path.join(eora_path, "lora_adapter.safetensors"))
print(eora_weight)
