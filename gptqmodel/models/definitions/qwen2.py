# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

from ..fusable import FusableConfig, LlamaFusableConfig
from .llama import LlamaQModel


class Qwen2QModel(LlamaQModel):
    fusable = FusableConfig("qwen2").merge(LlamaFusableConfig())

    shared_input_verified_model_types = frozenset({"qwen2"})
