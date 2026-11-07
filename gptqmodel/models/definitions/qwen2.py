# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from ..fusable import LlamaFusableConfig
from .llama import LlamaQModel


class Qwen2QModel(LlamaQModel):
    fusable = LlamaFusableConfig().derive("qwen2", parent_types={
        "qkv": (
            "transformers.models.qwen2.modeling_qwen2.Qwen2Attention",
        ),
        "gate_up": (
            "transformers.models.qwen2.modeling_qwen2.Qwen2MLP",
        ),
    })

    shared_input_verified_model_types = frozenset({"qwen2"})
