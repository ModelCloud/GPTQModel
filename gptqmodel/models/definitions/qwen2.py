# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from ..fusable import FusableConfig, LlamaFusableConfig
from .llama import LlamaQModel


class Qwen2QModel(LlamaQModel):
    solver_format_support = frozenset({("gptq", "mxfp4", "cpu")})

    # Qwen2 keeps the Llama decoder layout (an `x * w` RMSNorm and a
    # post-attention norm that feeds the MLP), so model-wide rotation is
    # exact. The declaration is per model type and is never inherited; see
    # `gptqmodel.models.base.rotation_supported`.
    supports_rotation = frozenset({"qwen2"})

    fusable = FusableConfig("qwen2").merge(LlamaFusableConfig())

    shared_input_verified_model_types = frozenset({"qwen2"})
