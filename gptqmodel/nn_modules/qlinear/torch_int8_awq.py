# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# Credit: int8 kernel sync adapted from Yintong Lu (yintong-lu), vLLM PR #35697.


from typing import Optional, Tuple

import torch

from ...adapter.adapter import Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND


class TorchInt8AwqLinear(AWQuantLinear):
    """Retired backend entry retained to provide actionable migration errors."""

    SUPPORTS_BACKENDS = [BACKEND.AWQ_TORCH_INT8]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 0}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_DEVICES = [DEVICE.CPU]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    REQUIRES_FORMAT_V2 = False
    SUPPORTS_RAW_PACKING = False

    QUANT_TYPE = "torch_int8_awq"

    def __init__(self, *args, **kwargs):
        _, error = self.validate_once()
        raise error

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        return False, NotImplementedError(
            "AWQ_TORCH_INT8 is retired: expanding AWQ INT4 weights to persistent INT8 is disabled. "
            "Use BACKEND.AUTO to select a packed kernel, or BACKEND.AWQ_TORCH for a portable fallback."
        )


__all__ = ["TorchInt8AwqLinear"]
