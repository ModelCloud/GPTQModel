# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

from .autoround import AutoRoundWeightSchema
from .awq import AWQWeightSchema
from .base import BaseInputSchema, BaseWeightSchema
from .bitnet import BitnetWeightSchema
from .compressed_tensors import (
    CompressedTensorsInputSchema,
    CompressedTensorsWeightSchema,
)
from .fp8 import Fp8InputSchema, Fp8WeightSchema
from .gpt_oss_mxfp4 import GptOssMxfp4WeightSchema
from .gptq import GPTQWeightSchema
from .humming import HummingInputSchema, HummingWeightSchema
from .modelopt import ModeloptInputSchema, ModeloptWeightSchema
from .mxfp4 import Mxfp4WeightSchema

WEIGHT_SCHEMA_MAP: dict[str, type[BaseWeightSchema]] = {
    "auto-round": AutoRoundWeightSchema,
    "auto_round": AutoRoundWeightSchema,
    "awq": AWQWeightSchema,
    "bitnet": BitnetWeightSchema,
    "compressed-tensors": CompressedTensorsWeightSchema,
    "fp8": Fp8WeightSchema,
    "gptq": GPTQWeightSchema,
    "humming": HummingWeightSchema,
    "modelopt": ModeloptWeightSchema,
    "mxfp4": Mxfp4WeightSchema,
    "gpt_oss_mxfp4": GptOssMxfp4WeightSchema,
}

INPUT_SCHEMA_MAP: dict[str, type[BaseInputSchema]] = {
    "compressed-tensors": CompressedTensorsInputSchema,
    "fp8": Fp8InputSchema,
    "humming": HummingInputSchema,
    "modelopt": ModeloptInputSchema,
}

BaseWeightSchema.WEIGHT_SCHEMA_MAP = WEIGHT_SCHEMA_MAP
BaseInputSchema.INPUT_SCHEMA_MAP = INPUT_SCHEMA_MAP


__all__ = [
    "BaseInputSchema",
    "BaseWeightSchema",
    "HummingInputSchema",
    "HummingWeightSchema",
]
