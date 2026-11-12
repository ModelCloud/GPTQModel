# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Immutable choices for one quantization run after backend preflight."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Optional

from ..quantization.config import FORMAT, METHOD, BaseQuantizeConfig
from ..utils.backend import BACKEND


class ProcessorStep(str, Enum):
    NATIVE = "native"
    PREPROCESSOR = "preprocessor"
    GPTQ = "gptq"
    QQQ = "qqq"
    VPTQ = "vptq"
    AWQ = "awq"
    PARO = "paroquant"
    EXL3 = "exl3"
    FP8_GSQ = "fp8_gsq"
    WEIGHT_ONLY = "weight_only"
    EORA = "eora"


_CALIBRATED_PRIMARY = {
    METHOD.GPTQ: ProcessorStep.GPTQ,
    METHOD.QQQ: ProcessorStep.QQQ,
    METHOD.VPTQ: ProcessorStep.VPTQ,
    METHOD.AWQ: ProcessorStep.AWQ,
    METHOD.PARO: ProcessorStep.PARO,
    METHOD.EXL3: ProcessorStep.EXL3,
}
_CHECKPOINT_PRIMARY = {
    ProcessorStep.GPTQ,
    ProcessorStep.QQQ,
    ProcessorStep.AWQ,
    ProcessorStep.PARO,
    ProcessorStep.EXL3,
    ProcessorStep.WEIGHT_ONLY,
}
_PROCESSOR_TYPES = {
    ProcessorStep.NATIVE: ("native_processor", "NativeProcessor"),
    ProcessorStep.PREPROCESSOR: ("module_preprocessor", "ModulePreProcessor"),
    ProcessorStep.GPTQ: ("gptq_processor", "GPTQProcessor"),
    ProcessorStep.QQQ: ("qqq_processor", "QQQProcessor"),
    ProcessorStep.VPTQ: ("vptq_processor", "VPTQProcessor"),
    ProcessorStep.AWQ: ("awq_processor", "AWQProcessor"),
    ProcessorStep.PARO: ("paroquant_processor", "ParoQuantProcessor"),
    ProcessorStep.EXL3: ("exllamav3_processor", "EXL3Processor"),
    ProcessorStep.FP8_GSQ: ("fp8_gsq_processor", "FP8GSQProcessor"),
    ProcessorStep.WEIGHT_ONLY: ("weight_only_processor", "WeightOnlyProcessor"),
    ProcessorStep.EORA: ("eora_processor", "EoraProcessor"),
}


def processor_class(step: ProcessorStep):
    """Load only the selected processor, preserving optional method imports."""
    module_name, class_name = _PROCESSOR_TYPES[step]
    return getattr(import_module(f"..looper.{module_name}", package=__package__), class_name)


@dataclass(frozen=True)
class QuantizationRunPlan:
    """One resolved method, format, backend pair, and ordered processor chain.

    ``pack_backend`` selects the quantized kernel. ``loop_backend`` preserves the
    explicit caller backend passed to the looper, including AUTO semantics.
    """

    method: METHOD
    export_method: METHOD
    format: FORMAT
    pack_backend: Optional[BACKEND]
    loop_backend: Optional[BACKEND]
    batch_size: int
    weight_only: bool
    processors: tuple[ProcessorStep, ...]

    @classmethod
    def build(
        cls,
        config: BaseQuantizeConfig,
        *,
        format_code: FORMAT,
        pack_backend: Optional[BACKEND],
        loop_backend: Optional[BACKEND],
        batch_size: int,
        needs_lora: bool,
    ) -> "QuantizationRunPlan":
        has_gptaq = getattr(config, "gptaq", None) is not None
        has_foem = getattr(config, "foem", None) is not None
        weight_only = getattr(config, "uses_weight_only_lifecycle", lambda: False)()

        if weight_only:
            if needs_lora:
                raise NotImplementedError("Weight-only quantization does not support adapter/EoRA generation.")
            if has_gptaq:
                raise NotImplementedError("Weight-only quantization does not support GPTAQ/native activation capture.")
            if has_foem:
                raise NotImplementedError("Weight-only quantization does not support FOEM/native activation capture.")
            steps = (ProcessorStep.WEIGHT_ONLY,)
        else:
            if config.method == METHOD.EXL3:
                if needs_lora:
                    raise NotImplementedError("EXL3 quantization does not support adapter/EoRA generation.")
                if has_gptaq:
                    raise NotImplementedError("EXL3 quantization does not support GPTAQ/native activation capture.")
                if has_foem:
                    raise NotImplementedError("EXL3 quantization does not support FOEM/native activation capture.")
            if config.method == METHOD.FP8 and getattr(config, "gsq_calibration", False):
                if needs_lora:
                    raise NotImplementedError("Calibrated FP8 GSQ does not support adapter generation")
                primary = ProcessorStep.FP8_GSQ
            else:
                try:
                    primary = _CALIBRATED_PRIMARY[config.method]
                except KeyError as exc:
                    raise ValueError(f"No calibrated processor for {config.method}") from exc

            steps_list = []
            if has_gptaq or (has_foem and config.foem.alpha > 0):
                steps_list.append(ProcessorStep.NATIVE)
            if getattr(config, "preprocessors", None):
                steps_list.append(ProcessorStep.PREPROCESSOR)
            steps_list.append(primary)
            if needs_lora:
                steps_list.append(ProcessorStep.EORA)
            steps = tuple(steps_list)

        return cls(
            method=config.method,
            export_method=getattr(config, "export_quant_method", lambda: config.method)(),
            format=format_code,
            pack_backend=pack_backend,
            loop_backend=loop_backend,
            batch_size=batch_size,
            weight_only=weight_only,
            processors=steps,
        )

    @property
    def checkpoint_supported(self) -> bool:
        return (
            len(self.processors) in (1, 2)
            and self.processors[0] in _CHECKPOINT_PRIMARY
            and (len(self.processors) == 1 or self.processors[1] is ProcessorStep.EORA)
        )

    def describe(self) -> dict[str, object]:
        """Return the resolved choices without exposing mutable runtime objects."""
        return {
            "method": self.method.value,
            "export_method": self.export_method.value,
            "format": self.format.value,
            "pack_backend": self.pack_backend.value if self.pack_backend is not None else None,
            "loop_backend": self.loop_backend.value if self.loop_backend is not None else None,
            "batch_size": self.batch_size,
            "weight_only": self.weight_only,
            "processors": [step.value for step in self.processors],
            "checkpoint_supported": self.checkpoint_supported,
        }
