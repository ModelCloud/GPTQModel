# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..fusable import FusableConfig, LlamaFusableConfig
from .base_qwen2_vl import BaseQwen2VLGPTQ


class Qwen2_5_VLQModel(BaseQwen2VLGPTQ):
    fusable = FusableConfig("qwen2_5_vl").merge(LlamaFusableConfig())

    pass
