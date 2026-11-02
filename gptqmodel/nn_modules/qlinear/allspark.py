# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSpark W8A16 linear (bits=8, group_size=-1, desc_act=False).

Load-only for now: the module takes GPTQ-format ``qweight`` / ``scales`` and
repacks them into the kernel layout in ``post_init``. Raw-weight packing is a
separate decision, see the draft PR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.allspark import (
    ALLSPARK_K_ALIGN,
    ALLSPARK_N_ALIGN,
    allspark_gemm,
    allspark_repack,
    prewarm_allspark_extension,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger


log = setup_logger()


class AllSparkLinear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_ALLSPARK]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # 0 keeps the format registered but opts out of AUTO selection: AllSpark overlaps
    # Marlin on bits=8/group_size=-1 and should only run when asked for explicitly.
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 0, FORMAT.GPTQ_V2: 0}
    SUPPORTS_BITS = [8]
    SUPPORTS_GROUP_SIZE = [-1]
    SUPPORTS_DESC_ACT = [False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [ALLSPARK_K_ALIGN]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [ALLSPARK_N_ALIGN]
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    def __init__(
        self,
        bits: int,
        group_size: int,
        desc_act: bool,
        sym: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        register_buffers: bool = False,
        adapter=None,
        **kwargs,
    ):
        if bits != 8:
            raise ValueError(f"AllSpark supports bits=8 only, got {bits}")
        if group_size != -1:
            raise ValueError(f"AllSpark supports group_size=-1 only, got {group_size}")
        if desc_act:
            raise ValueError("AllSpark does not support desc_act")

        self.compute_dtype = kwargs.get("dtype") or torch.float16

        super().__init__(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", None),
            adapter=adapter,
            register_buffers=False,
            **kwargs,
        )

        self.pack_factor = 32 // bits
        # GPTQ layout: 4 codes per int32 along the input dimension.
        self.register_parameter(
            "qweight",
            nn.Parameter(
                torch.empty(in_features // self.pack_factor, out_features, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        self.register_parameter(
            "scales",
            nn.Parameter(
                torch.empty(1, out_features, dtype=self.compute_dtype), requires_grad=False
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=self.compute_dtype)
            )
        else:
            self.bias = None

        self._repacked = False

    @torch.no_grad()
    def post_init(self):
        """Unpack the GPTQ 8-bit codes and reorder them into the kernel layout."""
        if self._repacked:
            return
        if not prewarm_allspark_extension():
            raise RuntimeError("AllSpark extension failed to build")
        qweight = self.qweight.data
        k4, n = qweight.shape
        k = k4 * self.pack_factor
        shifts = (torch.arange(self.pack_factor, device=qweight.device, dtype=torch.int32)
                  * 8).view(self.pack_factor, 1)
        codes = ((qweight.view(k4, 1, n) >> shifts) & 0xFF).reshape(k, n).to(torch.uint8)

        scales = self.scales.data.reshape(-1).contiguous()
        packed, packed_scales = allspark_repack(
            codes.contiguous(), scales, self.out_features, self.in_features
        )
        self.register_buffer("qweight_packed", packed, persistent=False)
        self.register_buffer("scales_packed", packed_scales, persistent=False)
        self._repacked = True
        del self.qweight, self.scales
        self.qweight = None
        self.scales = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._repacked:
            self.post_init()
        shape = x.shape[:-1] + (self.out_features,)
        x2d = x.reshape(-1, self.in_features)
        out = allspark_gemm(x2d, self.qweight_packed, self.scales_packed, self.out_features)
        out = out.reshape(shape)
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)
