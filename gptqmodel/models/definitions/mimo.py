# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from . import LlamaQModel


# MiMo is based on Qwen2
# https://huggingface.co/XiaomiMiMo/MiMo-7B-RL/blob/main/modeling_mimo.py
class MimoQModel(LlamaQModel):
    require_trust_remote_code = True
