# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from ..quantization.config import FORMAT, METHOD, resolve_quant_format
from ..utils.model import get_module_by_name
from ..utils.paroquant import build_identity_rotation_buffers
from .awq_processor import AWQProcessor


class ParoQuantProcessor(AWQProcessor):
    """AWQ-derived processor that packs modules with ParoQuant-specific buffers."""

    def _select_qlinear_kernel_for_format(self, format_value: FORMAT):
        """Validates the requested format and returns the ParoQuant kernel class."""

        fmt = FORMAT(format_value) if not isinstance(format_value, FORMAT) else format_value
        if fmt != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PAROQUANT does not support this FORMAT: {format_value}")
        return ParoQuantQuantLinear

    def _resolve_qlinear_kernel(self, module_name=None):
        """Resolves the effective ParoQuant kernel after dynamic format overrides."""

        format_override = self.qcfg.dynamic_get(module_name, "format", None) if module_name else None
        target_format = resolve_quant_format(format_override or self.qcfg.format, self.qcfg.method)
        if target_format != FORMAT.PAROQUANT:
            raise ValueError(f"METHOD.PAROQUANT does not support dynamic format override `{target_format}`.")
        return ParoQuantQuantLinear

    def pack_module(self, module):
        """Runs standard AWQ packing and then installs identity rotation buffers."""

        super().pack_module(module)

        qmodule = get_module_by_name(self.gptq_model.model, module.full_name)
        if not isinstance(qmodule, ParoQuantQuantLinear):
            raise TypeError(
                f"Expected `{module.full_name}` to be packed as ParoQuantQuantLinear, got `{type(qmodule).__name__}`."
            )

        pairs, theta, channel_scales = build_identity_rotation_buffers(
            in_features=qmodule.in_features,
            group_size=qmodule.group_size,
            krot=qmodule.krot,
            device=qmodule.theta.device,
            dtype=qmodule.theta.dtype,
        )

        qmodule.pairs.copy_(pairs.to(device=qmodule.pairs.device, dtype=qmodule.pairs.dtype))
        qmodule.theta.copy_(theta.to(device=qmodule.theta.device, dtype=qmodule.theta.dtype))
        qmodule.channel_scales.copy_(
            channel_scales.to(device=qmodule.channel_scales.device, dtype=qmodule.channel_scales.dtype)
        )
        qmodule.post_init()

    def finalize(self, model, **kwargs):
        """Marks the model as ParoQuant-quantized before delegating shared finalization."""

        model.quantized = True
        model.quantize_config.method = METHOD.PAROQUANT
        super(AWQProcessor, self).finalize(model=model, **kwargs)

    @classmethod
    def name(cls) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "paroquant"
