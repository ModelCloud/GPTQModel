# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect

import torch

from ..base import BaseQModel
from ..local import LocalModelLoader


class MptQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    pre_lm_head_norm_module = "transformer.norm_f"

    module_tree = [
        "transformer",
        "blocks",
        "#",
        {
            # MPT normalizes before attention and before the FFN; the tree used
            # to omit both norms, so AWQ could not build a single group. The FFN
            # output projection follows the GELU.
            "norm_1": ("norm_1:!",),
            "attn": ("Wqkv:0", "out_proj:1"),
            "norm_2": ("norm_2:!",),
            "ffn": ("up_proj:0:nosrc", "down_proj:1"),
        }
    ]

    def awq_module_replay_inputs(self, module, x, module_kwargs):
        """Supply the residual stream ``MptMLP.forward`` requires.

        ``MptBlock`` passes the pre-norm hidden state into the MLP and adds it
        back after ``down_proj``, so the forward signature is
        ``forward(hidden_states, residual)``. The per-linear activation capture
        never records that argument, so replaying the ``ffn`` block for the
        ``norm_2`` scaling group raised
        ``TypeError: MptMLP.forward() missing 1 required positional argument``
        and aborted the whole run (#4872).

        A zero residual is inert for the search: the term is added identically to
        the reference output and to every candidate scale's output, so it cancels
        out of the reconstruction loss the ratio grid minimizes.
        """

        x, module_kwargs = super().awq_module_replay_inputs(module, x, module_kwargs)
        try:
            parameters = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            return x, module_kwargs
        if "residual" not in parameters:
            return x, module_kwargs
        kwargs = dict(module_kwargs)
        if not torch.is_tensor(kwargs.get("residual")):
            kwargs["residual"] = torch.zeros_like(x)
        return x, kwargs
