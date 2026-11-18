# SPDX-FileCopyrightText: 2025 Xiaomi Corporation
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""MiMo configuration (Qwen2 plus the MTP layer count)."""

from ..native.qwen2.configuration_qwen2 import Qwen2Config


class MiMoConfig(Qwen2Config):
    model_type = "mimo"

    def __init__(self, *args, num_nextn_predict_layers=0, **kwargs):
        if isinstance(num_nextn_predict_layers, bool) or not isinstance(num_nextn_predict_layers, int):
            raise ValueError("num_nextn_predict_layers must be a non-negative integer")
        if num_nextn_predict_layers < 0:
            raise ValueError("num_nextn_predict_layers must be a non-negative integer")
        self.num_nextn_predict_layers = num_nextn_predict_layers
        super().__init__(*args, **kwargs)
