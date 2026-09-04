# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from . import LlamaQModel


class DeciLMQModel(LlamaQModel):
    require_trust_remote_code = True
    layer_modules_strict = False # nemotron ultra skips modules
