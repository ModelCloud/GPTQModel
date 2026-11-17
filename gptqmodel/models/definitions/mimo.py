# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..local import LocalModelLoader
from . import LlamaQModel


# MiMo is based on Qwen2
# Pinned source/provenance: ../ports/mimo/README.md
class MimoQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    require_trust_remote_code = False
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.mimo.configuration_mimo.MiMoConfig",
        model_class="gptqmodel.models.ports.mimo.modeling_mimo.MiMoForCausalLM",
    )

    # MTP is checkpoint-owned, full-precision auxiliary state outside the
    # causal-LM decoder loop, exactly as in the pinned upstream implementation.
    # Its concatenated hidden/token inputs are not ordinary Llama AWQ sources.
    def awq_get_modules_for_scaling(self, *args, **kwargs):
        """Plain AWQ only until fold/search/replay has been qualified."""
        return []
