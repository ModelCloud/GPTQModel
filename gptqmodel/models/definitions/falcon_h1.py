# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ..base import BaseQModel
from ..fusable import FusableConfig, LlamaFusableConfig


class FalconH1QModel(BaseQModel):
    fusable = FusableConfig("falcon_h1").merge(LlamaFusableConfig())

    pre_lm_head_norm_module = "model.final_layernorm"
    layers_node = "model.layers"

    module_tree= [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            # Attention and the Mamba mixer are parallel branches of one norm.
            "self_attn": ("q_proj:0:in=x", "k_proj:0:in=x", "v_proj:0:in=x", "o_proj:1"),
            # The SSM between in_proj and out_proj is not homogeneous.
            "mamba": ("in_proj:0:in=x:nosrc", "out_proj:1", "conv1d:!"),
            "pre_ff_layernorm": ("pre_ff_layernorm:!",),
            "feed_forward": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        }
    ]

    def _restore_falcon_h1_mup_buffers(self, model) -> None:
        """Rebuild the non-persistent MuP vector after a lazy shell load.

        ``FalconH1Mixer.mup_vector`` is registered with ``persistent=False`` and
        computed from the config in ``FalconH1Model.__init__``. LazyTurtle
        materializes the decoder from checkpoint tensors only, so the buffer
        stays on META unless it is re-registered here.
        """
        from transformers.models.falcon_h1.modeling_falcon_h1 import compute_mup_vector

        root = getattr(model, "model", model)
        layers = getattr(root, "layers", None)
        if layers is None:
            return

        config = getattr(root, "config", None) or getattr(model, "config", None)
        if config is None:
            return

        for layer in layers:
            mamba = getattr(layer, "mamba", None)
            if mamba is None or mamba.__class__.__name__ != "FalconH1Mixer":
                continue

            buffer = getattr(mamba, "mup_vector", None)
            if isinstance(buffer, torch.Tensor) and not getattr(buffer, "is_meta", False) and buffer.device.type != "meta":
                continue

            target_device = getattr(buffer, "device", None)
            if target_device is None or target_device.type == "meta":
                target_device = getattr(getattr(mamba, "D", None), "device", None)
            if target_device is None or target_device.type == "meta":
                target_device = torch.device("cpu")

            mamba.register_buffer(
                "mup_vector",
                compute_mup_vector(config).to(device=target_device),
                persistent=False,
            )

    def after_model_load(self, model, load_quantized_model=False):
        model = super().after_model_load(model, load_quantized_model=load_quantized_model)
        self._restore_falcon_h1_mup_buffers(model)
        return model

    def pre_quantize_generate_hook_start(self):
        self._restore_falcon_h1_mup_buffers(self.model)
