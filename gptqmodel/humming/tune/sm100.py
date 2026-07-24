# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

from .. import dtypes
from .sm8x import Sm80Heuristics


# TODO (mgoin): add proper heuristics
class Sm100Heuristics(Sm80Heuristics):
    max_smem_size: int = 227 * 1024
    sm_version: int = 100
    b8_allowed_dtypes: list[dtypes.DataType] = [dtypes.int8, dtypes.float8e4m3, dtypes.float8e5m2]
