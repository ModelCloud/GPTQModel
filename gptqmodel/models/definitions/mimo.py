# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from . import LlamaQModel


# MiMo is based on Qwen2
class MimoQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.mimo.configuration_mimo.MiMoConfig",
        model_class="gptqmodel.models.ports.mimo.modeling_mimo.MiMoForCausalLM",
    )
    # MTP layers are full-precision auxiliary state outside `model.layers`, so
    # AWQ scale search only sees the ordinary Qwen2-style decoder layers.
