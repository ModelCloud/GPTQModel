# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Pangolin native CUDA GEMV kernel for the planar ``gptq_p`` format.

Pangolin is the fastest path for decode-shape (small-M) planar inputs on
NVIDIA compute capability >= 8.0.  It falls back to the planar Triton fused
GEMV / dequant+matmul paths when the native extension is unavailable, the
g_idx is not block-uniform, or the input is not decode-shaped.
"""

import torch

from ...adapter.adapter import Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from . import FormatSupport
from .tritonv2 import TritonV2Linear


class PangolinQuantLinear(TritonV2Linear):
    """Quantized linear layer backed by the native Pangolin planar GEMV kernel.

    Handles ``gptq_p`` planar bit widths (3/5/6/7) where the continuous GPTQ
    kernels cannot operate because the packed codes are split into bit planes.
    2/4/8-bit ``gptq_p`` weights are bit-identical to the non-planar layout and
    remain with the existing Triton/Torch continuous kernels.
    """

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_PANGOLIN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # gptq_p planar bits; higher priority than TritonV2Linear so AUTO picks
    # Pangolin for decode-shaped planar layers when the runtime is available.
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.GPTQ_P: FormatSupport(priority=50, bits=(3, 5, 6, 7)),
    }
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.int16, torch.int8]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "pangolin"


__all__ = ["PangolinQuantLinear"]
