from datasets import load_dataset
from gptqmodel import QuantizeConfig
from gptqmodel import GPTQModel, BACKEND
import torch

from gptqmodel.quantization.config import EoRA
from gptqmodel.utils.eval import EVAL
from gptqmodel.eora import get_eora, get_eora_optimize

bit = 4
model_id = "meta-llama/Llama-3.2-1B"
model = None

# 3-bit groupsize = 128 or -1 both have bugs
# quant_path = "Llama-3.2-1B-gptqmodel-3bit"
# fake_quant_path = "Llama-3.2-1B-gptqmodel-3bit-fakequantized/qw.pt"

quant_path = "/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit"
fake_quant_path = "/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit-fakequantized/qw.pt"
eora_path = "/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit-eora-rank-128/eora.pt"
eora_path = "/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit-eora-rank-128-v2/eora.pt"
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

  # model.save(quant_path)

# test post-quant inference
flag2 = False
if flag2:
  model = GPTQModel.load(quant_path)

  result = model.generate("Uncovering deep insights begins with")[0]
  print(result)
  # lm_eval_results = GPTQModel.eval(quant_path, framework=EVAL.LM_EVAL, tasks=[EVAL.LM_EVAL.ARC_CHALLENGE])
  # print(lm_eval_results)

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
# print(eora_weight)

save = False
if save:
  from safetensors.torch import save_file
  import json
  lowrank_config = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": None,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": False,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "lora_alpha": 128,
    "lora_dropout": 0.1,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": 128,
    "rank_pattern": {},
    "revision": None,
    "target_modules": [
        "o_proj",
        "v_proj",
        "down_proj",
        "up_proj",
        "q_proj",
        "gate_proj",
        "k_proj"
    ],
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False
  }
  # Serializing json
  json_object = json.dumps(lowrank_config, indent=4)

  # Writing to the adapter_config.json
  with open(f"/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit-eora-rank-128-hf/adapter_config.json", "w") as outfile:
      outfile.write(json_object)
  ## save the lowrank weight

  save_file(eora_weight, f"/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit-eora-rank-128-hf/adapter_model.safetensors")

flag4 = True
if flag4:
  batch_size = 1
  from test_prepare_dataset import construct_ARC
  calibration_dataset = construct_ARC(nsamples=1024)
  eora_rank = 128
  eora_weight = get_eora_optimize(model_id, quant_config, quantized_weights, calibration_dataset, batch_size, eora_rank)
  torch.save(eora_weight, eora_path)
  print(eora_weight)

