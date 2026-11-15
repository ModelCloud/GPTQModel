# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect

import torch

from ..base import BaseQModel


class BloomQModel(BaseQModel):
    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attention": ("query_key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # BloomGelu between the two projections is not positively homogeneous.
            "mlp": ("dense_h_to_4h:0:nosrc", "dense_4h_to_h:1"),
        }
    ]

    def awq_module_replay_inputs(self, module, x, module_kwargs):
        """Provide arguments required for Bloom attention and MLP replay.

        Bloom passes residual and positional information directly into its submodules:
        - ``BloomMLP.forward(hidden_states, residual)``
        - ``BloomAttention.forward(hidden_states, residual, alibi, attention_mask, ...)``

        AWQ scores candidate scale losses by replaying the parent module, but per-linear
        activation capture only records the block input tensor. Supplying the input
        tensor as residual satisfies the interface without altering scale loss scoring,
        and synthesized alibi slopes and causal attention masks provide valid shapes.
        """
        x, module_kwargs = super().awq_module_replay_inputs(module, x, module_kwargs)
        try:
            parameters = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            return x, module_kwargs
        if "residual" not in parameters and "alibi" not in parameters:
            return x, module_kwargs

        kwargs = dict(module_kwargs)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if "residual" in parameters and not torch.is_tensor(kwargs.get("residual")):
            kwargs["residual"] = x
        if "attention_mask" in parameters and kwargs.get("attention_mask") is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            kwargs["attention_mask"] = torch.zeros(
                (batch_size, 1, seq_len, seq_len), dtype=x.dtype, device=x.device
            )
        if "alibi" in parameters and kwargs.get("alibi") is None:
            batch_size, seq_len = x.shape[0], x.shape[1]
            mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=x.device)
            num_heads = (
                getattr(module, "num_heads", None)
                or getattr(getattr(self, "model", None), "config", None)
                and getattr(self.model.config, "n_head", None)
                or getattr(getattr(self, "model", None), "config", None)
                and getattr(self.model.config, "num_attention_heads", None)
            )
            try:
                from transformers.models.bloom.modeling_bloom import build_alibi_tensor

                kwargs["alibi"] = build_alibi_tensor(mask, num_heads, dtype=x.dtype)
            except Exception:
                pass
        return x, kwargs


__all__ = ["BloomQModel"]
