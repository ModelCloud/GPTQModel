# SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GPTQModel's lightweight NF12 inference API for BF16 weights.

The native libraries are built from GPTQModel's vendored sources on first use.
Importing this package does not initialize the full GPTQModel framework.
"""

from ._library import library_path, load
from .api import NF12Weight, decode, embedding, encode, linear, linear_bf16_fp32, linear_bf16t, linear_fp32, pack_bf16t
from .moe import (
    NF12MoEWeight,
    encode_moe,
    fused_moe,
    moe_align,
    moe_combine,
    moe_linear,
    moe_linear_bf16t,
    moe_silu_mul,
    pack_bf16t_moe,
)
from .nn import NF12Embedding, NF12Linear


__version__ = "1.0.0"

__all__ = [
    "NF12Embedding",
    "NF12Linear",
    "NF12MoEWeight",
    "NF12Weight",
    "decode",
    "embedding",
    "encode",
    "encode_moe",
    "fused_moe",
    "library_path",
    "linear",
    "linear_bf16_fp32",
    "linear_bf16t",
    "linear_fp32",
    "load",
    "moe_align",
    "moe_combine",
    "moe_linear",
    "moe_linear_bf16t",
    "moe_silu_mul",
    "pack_bf16t",
    "pack_bf16t_moe",
]
