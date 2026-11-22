# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect

import torch

from ..base import BaseQModel
from ..local import LocalModelLoader


class TeleChat2QModel(BaseQModel):
    loader = LocalModelLoader(
        "gptqmodel.models.ports.telechat2.configuration_telechat2.Telechat2Config",
        "gptqmodel.models.ports.telechat2.modeling_telechat2.Telechat2ForCausalLM",
        tokenizer_class="gptqmodel.models.ports.telechat2.tokenization_telechat2.TelechatTokenizerFactory",
    )
    fusable = None

    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # `query` and `key_value` are the attention's input projections
            # (modeling_telechat2.py:358-359). The old tree declared only
            # `dense`, so both stayed fp16 while the coverage report still
            # claimed every spec matched.
            "self_attention": ("query:0", "key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        # `key_value` interleaves K and V per KV head, so its output rows are not
        # the `dense` input channels even when the widths happen to match.
        unsafe = {id(getattr(getattr(module, "self_attention", None), "key_value", None))}
        if getattr(module, "apply_residual_connection_post_layernorm", False):
            # The residual is the norm output itself; a norm fold would rescale it.
            unsafe.update(id(getattr(module, name, None)) for name in ("input_layernorm", "post_attention_layernorm"))
        return [node for node in nodes if id(node.get("prev_op")) not in unsafe]

    def awq_module_replay_inputs(self, module, x, module_kwargs):
        """Give TeleChat2's attention replay the arguments its forward requires.

        AWQ scores the ``query``/``key_value`` scaling group by replaying their
        common parent, ``TelechatAttention``, whose forward signature is
        ``(hidden_states, residual, attention_mask, ...)``. The residual is an
        additive constant shared by the reference and candidate outputs, so the
        block input is enough for the scale search; a full causal bool mask
        matches the calibration replay (no padding).
        """

        x, module_kwargs = super().awq_module_replay_inputs(module, x, module_kwargs)
        try:
            parameters = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            return x, module_kwargs
        if "residual" not in parameters:
            return x, module_kwargs

        kwargs = dict(module_kwargs)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if not torch.is_tensor(kwargs.get("residual")):
            kwargs["residual"] = x
        if "attention_mask" in parameters and kwargs.get("attention_mask") is None:
            seq_len = int(x.shape[-2])
            kwargs["attention_mask"] = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=x.device
            ).triu(1)
        return x, kwargs

__all__ = ["TeleChat2QModel"]
