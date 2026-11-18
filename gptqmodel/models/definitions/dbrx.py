# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ..base import BaseQModel
from ..local import LocalModelLoader


class DbrxQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.native.auto.NativeConfig",
        model_class="gptqmodel.models.ports.native.auto.NativeForCausalLM",
    )

    info = {"notes": "Use a defused `dbrx_converted` checkpoint from https://hf.co/ModelCloud with the package-owned loader."}

    @classmethod
    def after_defuser_conversion(cls, model):
        """Defuse the vendored ``DbrxExperts`` exactly as Defuser does the native class.

        Defuser selects its runtime patches by ``transformers.models`` class
        path, so the vendored class needs the same patch applied explicitly.
        """

        from defuser.modeling.runtime_defusion import patch_dbrx_experts

        from ..ports.native.dbrx.modeling_dbrx import DbrxExperts

        for module in list(model.modules()):
            if type(module) is DbrxExperts:
                patch_dbrx_experts(module)
