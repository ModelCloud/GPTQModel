# PEFT LoRA Training and Adapter Loading

GPT-QModel GPTQ/AWQ checkpoints can be loaded through Transformers and fine-tuned with PEFT LoRA. The quantized base weights stay frozen; only the LoRA adapter parameters are trained.

Use a current PEFT release together with `optimum>=1.24.0`. Older PEFT/Optimum combinations may warn that the GPT-QModel quantized layer type is unsupported. The examples below were validated with PEFT 0.21.2, Optimum 2.3.0, Transformers 5.19.0, and PyTorch 2.14.

For training, select the trainable kernel for the checkpoint's quantization method: `gptq_torch` for GPTQ and `torch_awq` for AWQ. `torch_awq` is the Transformers-facing alias for GPT-QModel's `AWQ_TORCH` kernel. The default inference kernels do not propagate gradients through the quantized layers, so a backward pass will fail.

```bash
pip install "gptqmodel[hf]" peft
```

## Train a LoRA adapter

```py
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(quant_path)
trainable_backends = {"gptq": "gptq_torch", "awq": "torch_awq"}
method = config.quantization_config["quant_method"]
config.quantization_config["backend"] = trainable_backends[method]
base = AutoModelForCausalLM.from_pretrained(quant_path, config=config, device_map="auto")
model = get_peft_model(
    base,
    LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    ),
)

# Train with the Transformers Trainer or another training loop, then save only the adapter.
model.save_pretrained(adapter_path)
```

## Load an existing adapter

```py
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_path)
```

The adapter remains separate at inference time. This is the supported path for a quantized base model.

## Merging the adapter

PEFT cannot merge LoRA deltas into packed GPTQ/AWQ weights. Calling `merge_and_unload()` on a quantized model therefore raises for GPTQ:

```text
ValueError: Cannot merge LORA layers when the model is gptq quantized
```

For AWQ, PEFT's AWQ LoRA wrapper raises `NotImplementedError` instead.

This is intentional, not a GPT-QModel defect: the base model does not store dense floating-point weights that a LoRA delta can be added to.

To produce a standalone merged checkpoint, merge the adapter into the original full-precision model first, then quantize the merged model with GPT-QModel:

```py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

full_precision = AutoModelForCausalLM.from_pretrained(
    original_model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
merged = PeftModel.from_pretrained(full_precision, adapter_path).merge_and_unload()
merged.save_pretrained(merged_model_path)

# Quantize merged_model_path with the normal GPTQModel workflow.
```

The merged model is dense and larger than the quantized checkpoint. Quantize it again before deployment if the original memory savings are required.

For training-free post-quantization accuracy recovery, GPT-QModel also provides EoRA adapters; see [EoRA](eora/README.md).
