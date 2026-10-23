# PEFT LoRA Training and Adapter Loading

GPT-QModel GPTQ/AWQ checkpoints can be loaded through Transformers and fine-tuned with PEFT LoRA. The quantized base weights stay frozen; only the LoRA adapter parameters are trained.

Use a current PEFT release together with `optimum>=1.24.0`. Older PEFT/Optimum combinations may warn that the GPT-QModel quantized layer type is unsupported. The FSDP recipe was validated with PEFT `0.21.3.dev0` (main checkout), Optimum 2.3.0, Transformers 5.20.2, and PyTorch 2.14.

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

## FSDP LoRA fine-tuning

Use PEFT's maintained [`examples/sft`](https://github.com/huggingface/peft/tree/main/examples/sft) script with a current PEFT checkout. The recipe below uses FSDP to shard the GPTQ model and train only its LoRA adapter. It was validated with the 8B Llama 3.1 GPTQ checkpoint on four RTX 4090 GPUs.

Clone PEFT and install the SFT example dependencies in the GPT-QModel environment:

```bash
git clone https://github.com/huggingface/peft.git
cd peft/examples/sft
pip install -r requirements.txt
```

In PEFT's `examples/sft/utils.py`, set the GPTQ backend before loading the model. Replace the existing `from_pretrained` call in `create_and_prepare_model` with:

```py
config = transformers.AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
if config.quantization_config["quant_method"] == "gptq":
    config.quantization_config["backend"] = "gptq_torch"
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    **model_kwargs,
)
```

In `train.py`, after creating `SFTTrainer` and before `trainer.train()`, set PEFT's FSDP auto-wrap policy. This pairs with `fsdp_use_orig_params: false` in the config below:

```py
if model_args.use_peft_lora and trainer.is_fsdp_enabled:
    from peft.utils.other import fsdp_auto_wrap_policy

    trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)
```

Finally, replace the last `trainer.save_model()` call in `train.py` with the following so the gathered LoRA tensors are written using standard PEFT keys:

```py
import os
from safetensors.torch import save_file

if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    state_dict = trainer.accelerator.get_state_dict(trainer.model)
    trainer.save_model()

    if trainer.is_world_process_zero():
        adapter_state_dict = {
            key.replace(".default.weight", ".weight"): value.cpu().contiguous()
            for key, value in state_dict.items()
            if key.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
        }
        if not adapter_state_dict:
            raise RuntimeError("FSDP full state dict contains no GPTQ LoRA tensors")
        save_file(adapter_state_dict, os.path.join(training_args.output_dir, "adapter_model.safetensors"))
else:
    trainer.save_model()
```

With PEFT `0.21.3.dev0` and Transformers `5.20.2`, `trainer.save_model()` created an empty adapter weights file for this GPTQ FSDP setup. The code above then collects the LoRA weights from the full FSDP state dict and writes them to `adapter_model.safetensors`. PEFT can load the resulting adapter with `PeftModel.from_pretrained`.

From `examples/sft`, launch with the four-GPU config at [`docs/peft/fsdp_config.yaml`](peft/fsdp_config.yaml):

```bash
accelerate launch --config_file /path/to/GPTQModel/docs/peft/fsdp_config.yaml train.py \
  --seed 100 \
  --model_name_or_path hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
  --dataset_name smangrul/ultrachat-10k-chatml \
  --chat_template_format chatml \
  --add_special_tokens False \
  --append_concat_token False \
  --splits train,test \
  --max_length 2048 \
  --num_train_epochs 1 \
  --logging_steps 5 \
  --logging_strategy steps \
  --eval_strategy epoch \
  --save_strategy epoch \
  --bf16 True \
  --packing True \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --weight_decay 1e-4 \
  --warmup_steps 0 \
  --max_grad_norm 1.0 \
  --output_dir llama3-8b-gptq-lora-fsdp \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing True \
  --use_reentrant False \
  --dataset_text_field content \
  --use_flash_attn False \
  --use_peft_lora True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj \
  --use_4bit_quantization False
```

The checkpoint is already GPTQ-quantized, so keep `--use_4bit_quantization False` (it enables bitsandbytes QLoRA). This command uses BF16 and batch size 1 per GPU; lower `--max_length` if memory is limited. Checkpoints are written to `--output_dir`.

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
