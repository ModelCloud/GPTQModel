# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from ..fusable import LlamaFusableConfig
from .base_qwen2_vl import BaseQwen2VLGPTQ


class Qwen2VLQModel(BaseQwen2VLGPTQ):
    fusable = LlamaFusableConfig().derive("qwen2_vl", parent_types={
        "qkv": (
            "transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLAttention",
        ),
        "gate_up": (
            "transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2MLP",
        ),
    })

    pass
