# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Immutable choices for one quantization run after backend preflight."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Optional

from ..quantization.config import (
    FORMAT,
    METHOD,
    QUANTIZE_BLACK_LIST,
    BaseQuantizeConfig,
)
from ..utils.backend import BACKEND, normalize_backend


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
# Declared checkpoint role per processor step. The values mirror each processor
# class's `lifecycle.checkpointing`; `test_checkpoint_roles_match_processor_lifecycles`
# keeps them in sync. Admission itself is decided by one shared predicate in
# `looper.loop_processor`, so preflight and the adapter cannot disagree.
_CHECKPOINT_ROLE_BY_STEP = {
    ProcessorStep.GPTQ: "primary",
    ProcessorStep.QQQ: "primary",
    ProcessorStep.AWQ: "primary",
    ProcessorStep.PARO: "primary",
    ProcessorStep.EXL3: "primary",
    ProcessorStep.WEIGHT_ONLY: "primary",
    ProcessorStep.EORA: "secondary",
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


def _backend_name(backend) -> Optional[str]:
    """Render a diagnostic backend without assuming the caller passed an enum."""
    if backend is None:
        return None
    if isinstance(backend, BACKEND):
        return backend.value
    return str(backend)


@dataclass(frozen=True)
class QuantizationRunPlan:
    """One resolved method, format, backend pair, and ordered processor chain.

    ``pack_backend`` selects the quantized kernel. ``loop_backend`` is the
    caller's backend after method-aware normalization, with AUTO preserved.
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
        # Inference-only formats declare no packing lifecycle. Refuse them here
        # as well as in `BaseQModel.quantize` so a caller that builds a plan
        # directly cannot get a weight-only chain with no implementation behind
        # it. Both entry points use one message.
        if config.method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {config.method}"
            )

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

        export_method = getattr(config, "export_quant_method", lambda: config.method)()
        return cls(
            method=config.method,
            export_method=export_method,
            format=format_code,
            pack_backend=pack_backend,
            # Store the normalized backend so diagnostics and downstream
            # consumers see the same typed value. AUTO stays AUTO.
            loop_backend=normalize_backend(loop_backend, quant_method=export_method),
            batch_size=batch_size,
            weight_only=weight_only,
            processors=steps,
        )

    @property
    def checkpoint_supported(self) -> bool:
        from ..looper.loop_processor import (
            CheckpointSupport,
            checkpoint_chain_supported,
        )

        roles = tuple(
            CheckpointSupport(
                _CHECKPOINT_ROLE_BY_STEP.get(step, CheckpointSupport.UNSUPPORTED.value)
            )
            for step in self.processors
        )
        return checkpoint_chain_supported(roles)

    def describe(self) -> dict[str, object]:
        """Return the resolved choices without exposing mutable runtime objects."""
        return {
            "method": self.method.value,
            "export_method": self.export_method.value,
            "format": self.format.value,
            "pack_backend": _backend_name(self.pack_backend),
            "loop_backend": _backend_name(self.loop_backend),
            "batch_size": self.batch_size,
            "weight_only": self.weight_only,
            "processors": [step.value for step in self.processors],
            "checkpoint_supported": self.checkpoint_supported,
        }
