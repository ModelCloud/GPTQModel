# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Backend selection and execution settings for GPT-QModel.

Example: ``GPTQModel.load(path, backend=MarlinConfig(fp32=False))``.
"""

from .utils.backend import BACKEND
from .utils.backend_config import (
    AwqGemmConfig,
    AwqTorchConfig,
    AwqTritonConfig,
    BackendConfig,
    BitsAndBytesConfig,
    EXL3CUDAConfig,
    GGUFCPPConfig,
    GGUFCUDAConfig,
    GGUFTorchConfig,
    MarlinConfig,
    ParoCUDAConfig,
    ParoTritonConfig,
    RuntimeConfig,
    SwordfishConfig,
    TorchConfig,
    TTBlockFloatConfig,
)


__all__ = [
    "BACKEND", "BackendConfig", "MarlinConfig", "TorchConfig", "AwqTorchConfig", "AwqGemmConfig",
    "AwqTritonConfig", "SwordfishConfig", "GGUFTorchConfig", "GGUFCPPConfig", "GGUFCUDAConfig",
    "ParoCUDAConfig", "ParoTritonConfig", "BitsAndBytesConfig", "TTBlockFloatConfig", "EXL3CUDAConfig", "RuntimeConfig",
]
