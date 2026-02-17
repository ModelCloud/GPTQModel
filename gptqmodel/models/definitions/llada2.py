from functools import wraps

import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.moe_lifecycle import GateUpDownMoELifecycleHooks


def _expand_attention_mask_to_4d(
    attention_mask: torch.Tensor, dtype: torch.dtype = None
) -> torch.Tensor:
    if attention_mask is None or attention_mask.dim() == 4:
        return attention_mask

    if attention_mask.dim() != 2:
        return attention_mask

    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    dtype = dtype if dtype is not None else attention_mask.dtype

    causal_mask = (
        torch.tril(torch.ones((seq_len, seq_len), dtype=dtype, device=device))
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, 1, seq_len, seq_len)
    )

    padding_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)

    return causal_mask * padding_mask


def _patch_class_forward(cls):
    if getattr(cls, "_llada2_attention_mask_patched", False):
        return

    original_forward = cls.forward

    @wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            kwargs["attention_mask"] = _expand_attention_mask_to_4d(
                kwargs["attention_mask"]
            )
        elif len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].dim() == 2:
            args = list(args)
            args[1] = _expand_attention_mask_to_4d(args[1])
            args = tuple(args)
        return original_forward(self, *args, **kwargs)

    cls.forward = patched_forward
    cls._llada2_attention_mask_patched = True


class LLaDA2MoeQModel(BaseQModel):
    pre_lm_head_norm_module = "model.norm"

    dynamic_expert_index = "num_experts"

    layer_modules_strict = False

    moe_lifecycle_hooks = GateUpDownMoELifecycleHooks()

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "attention": (
                "query_key_value:0",
                "query_layernorm:!",
                "key_layernorm:!",
                "dense:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp:moe:?": {
                "gate": ("gate:!",),
                "": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "shared_experts": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                "experts": {
                    "#": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            },
        },
    ]

    def after_model_load(self, model, load_quantized_model=False):

        model = super().after_model_load(
            model, load_quantized_model=load_quantized_model
        )

        if hasattr(model, "model") and model.model is not None:
            _patch_class_forward(type(model.model))

        _patch_class_forward(type(model))

        return model
