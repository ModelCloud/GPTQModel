from datasets import load_dataset
from gptqmodel import QuantizeConfig
from gptqmodel import GPTQModel
import torch
from gptqmodel.utils.eval import EVAL
from gptqmodel.eora import get_eora

bit = 3
model_id = "meta-llama/Llama-3.2-1B"
model = None

# 3-bit groupsize = 128 or -1 both have bugs
# quant_path = "Llama-3.2-1B-gptqmodel-3bit"
# fake_quant_path = "Llama-3.2-1B-gptqmodel-3bit-fakequantized/qw.pt"

quant_path = "Llama-3.2-1B-gptqmodel-4bit"
fake_quant_path = "Llama-3.2-1B-gptqmodel-4bit-fakequantized/qw.pt"
eora_path = "Llama-3.2-1B-gptqmodel-4bit-eora-rank-128/eora.pt"
quant_config = QuantizeConfig(bits=bit, group_size=128)

flag1 = False
if flag1:
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

# test post-quant inference
flag2 = False
if flag2:
  model = GPTQModel.load(quant_path)

  result = model.generate("Uncovering deep insights begins with")[0]

  lm_eval_results = GPTQModel.eval(quant_path, framework=EVAL.LM_EVAL, tasks=[EVAL.LM_EVAL.ARC_CHALLENGE])
  print(lm_eval_results)

# torch.save(quantized_weights, fake_quant_path)

quantized_weights = torch.load(fake_quant_path, map_location='cpu')

## 4-bit gs=128 Acc: 0.2850

flag3 = False
# improve downstream task accuracy using EoRA
if flag3:
  if model != None:
    del model

  data_name = "arc"
  eora_nsamples = 64
  eora_rank = 128
  dev = "cuda:0"
  # Construct the calibration dataset for EoRA
  eora_weight = get_eora(model_id=model_id, quant_config = quant_config, data_name=data_name, quantized_weights = quantized_weights, eora_nsamples=eora_nsamples, eora_rank =eora_rank, dev=dev)
  torch.save(eora_weight, eora_path)

eora_weight = torch.load(eora_path,  map_location='cpu')
print(eora_weight)