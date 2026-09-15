# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel


class Spark2_5QModel(BaseQModel):
    """GPTQ definition for Spark-X2.5's hybrid-attention decoder."""

    require_trust_remote_code = True

    # The tags are verified by the Spark decoder's real forward path.  The
    # fused QKV projection and the MLP gate/up projections consume the same
    # activation tensor as their tagged sibling group.
    shared_input_verified_model_types = frozenset({"spark2_5"})

    pre_lm_head_norm_module = "model.norm"

    # Spark has no rotary-embedding module: it computes RoPE tensors directly
    # in the remote model's forward method.
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "q_k_v_proj:0:in=x",
                # The 16-head output gate is small and precision-sensitive.
                "g_proj:0:!",
                "out_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": (
                "gate_proj:0:in=x",
                "up_proj:0:in=x",
                "down_proj:1",
            ),
        },
    ]

    def before_model_load(self, model_local_path: str, load_quantized_model: bool):
        """Adapt Spark's legacy mask-helper keyword names to Transformers 5.x."""

        del load_quantized_model
        from functools import wraps
        from importlib import import_module

        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        remote_model_cls = get_class_from_dynamic_module(
            "modeling_spark.Spark2_5Model",
            model_local_path,
        )
        remote_module = import_module(remote_model_cls.__module__)

        for mask_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
            mask_fn = getattr(remote_module, mask_name, None)
            if not callable(mask_fn) or getattr(
                mask_fn, "_gptqmodel_spark_mask_compat", False
            ):
                continue

            @wraps(mask_fn)
            def mask_compat(*args, _mask_fn=mask_fn, **kwargs):
                if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                    kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
                # Transformers 5.x removed cache_position from these helper
                # signatures, while Spark's remote forward still forwards it.
                kwargs.pop("cache_position", None)
                return _mask_fn(*args, **kwargs)

            mask_compat._gptqmodel_spark_mask_compat = True
            setattr(remote_module, mask_name, mask_compat)

    def after_model_load(self, model, load_quantized_model: bool = False):
        """Expose the dtype-only weight view expected by Spark's remote forward."""

        if not load_quantized_model:
            return model

        import torch

        for layer in model.model.layers:
            gate_proj = layer.mlp.gate_proj
            if hasattr(gate_proj, "weight"):
                continue

            # Spark reads only ``gate_proj.weight.dtype`` to cast activations.
            # Quantized linear kernels intentionally store qweight/scales rather
            # than a dense weight, so a zero-sized, unregistered tensor supplies
            # that dtype without restoring or serializing a dense matrix.
            compute_dtype = getattr(gate_proj, "compute_dtype", None)
            if compute_dtype is None:
                compute_dtype = gate_proj.scales.dtype
            gate_proj.weight = torch.empty(0, dtype=compute_dtype)

        return model


__all__ = ["Spark2_5QModel"]
