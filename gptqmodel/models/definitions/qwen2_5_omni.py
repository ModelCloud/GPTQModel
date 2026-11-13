# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..fusable import FusableConfig, LlamaFusableConfig
from .base_qwen2_5_omni import BaseQwen2_5_OmniGPTQ


class Qwen2_5_OmniGPTQ(BaseQwen2_5_OmniGPTQ):
    fusable = FusableConfig("qwen2_5_omni").merge(LlamaFusableConfig())

    pass
