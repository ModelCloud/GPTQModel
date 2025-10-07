# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import pack_module


@torch.inference_mode()
def test_pack_gpu_raises_on_misaligned_qzeros():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU pack alignment test")

    target_index = 0
    torch.cuda.set_device(target_index)

    in_features = 128
    out_features = 16
    layer_name = "model.layers.0.linear_attn.in_proj_qkvz"

    linear = torch.nn.Linear(in_features, out_features, bias=False)

    quant_module = TorchQuantLinear(
        bits=4,
        group_size=in_features,
        sym=True,
        desc_act=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        backend=BACKEND.TORCH,
        name=layer_name,
        lm_head_name="lm_head",
        register_buffers=True,
    )

    qModules = {layer_name: quant_module}
    layers = {layer_name: linear}

    # Craft qzeros with a single column to reproduce the misalignment reported during pack_gpu.
    q_zeros = torch.zeros((1, out_features), dtype=torch.int32)
    q_scales = torch.ones((1, out_features), dtype=torch.float32)
    q_g_idx = torch.zeros(in_features, dtype=torch.int32)

    quant_config = QuantizeConfig(
        bits=4,
        group_size=in_features,
        desc_act=True,
        sym=True,
        pack_impl="gpu",
        pack_dtype=torch.int32,
        device=f"cuda:{target_index}",
        offload_to_disk=False,
    )

    lock = threading.Lock()

    with pytest.raises(ValueError, match="pack_gpu expected zeros second dimension divisible"):
        pack_module(
            name=layer_name,
            qModules=qModules,
            q_scales=q_scales,
            q_zeros=q_zeros,
            q_g_idx=q_g_idx,
            layers=layers,
            quant_linear_cls=TorchQuantLinear,
            lock=lock,
            quantize_config=quant_config,
        )
