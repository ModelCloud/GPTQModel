# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSpark W8A16 linear (bits=8, group_size=-1, desc_act=False).

The module takes GPTQ-format ``qweight`` / ``scales`` and repacks them into
the kernel layout in ``post_init``.

Two device paths share this module and the ``allspark`` backend:

* CUDA (Ampere+) -> ``gptqmodel_ext/allspark`` (cp.async + mma.sync).
* ROCm RDNA4 (gfx1200/gfx1201) -> ``gptqmodel_ext/allspark_hip``
  (v_wmma 16x16x16, see ``gptqmodel/utils/allspark_hip.py``).

The packed weight layout is *not* interchangeable between them: the CUDA
kernel tiles N in 32-column blocks, the WMMA port pads N to 128. ``post_init``
picks the matching repack, and ``qweight_packed``/``scales_packed`` are
non-persistent buffers so a checkpoint never carries the wrong layout.
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
    allspark_supported,
    prewarm_allspark_extension,
)
from ...utils.allspark_hip import (
    allspark_hip_available,
    allspark_hip_gemm,
    allspark_hip_repack,
    prewarm_allspark_hip_extension,
    validate_allspark_hip_device,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM


log = setup_logger()


class AllSparkLinear(GPTQQuantLinear):
    # GPTQ sub-kernel of the shared AllSpark backend. ``BACKEND.ALLSPARK`` plus
    # the checkpoint method/format selects this class; no GPTQ-specific backend
    # name is required.
    SUPPORTS_BACKENDS = [BACKEND.ALLSPARK]
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
    # GPTQ loading/repacking is supported; the inherited generic packer expects
    # buffers this load-only module does not allocate.
    SUPPORTS_RAW_PACKING = False
    SUPPORTS_AUTO_PADDING = False
    # The ROCm port has the same K contract (16) and pads N up to 128 inside its
    # repack, so the CUDA-side 16-column contract is sufficient for both paths.
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [ALLSPARK_K_ALIGN]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [ALLSPARK_N_ALIGN]
    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    @classmethod
    def validate_device(cls, device):
        super().validate_device(device)
        if IS_ROCM:
            validate_allspark_hip_device(device)

    @classmethod
    def validate_once(cls):
        """Reject devices outside the compiled kernel's architecture contract."""
        if IS_ROCM and not allspark_hip_available():
            return False, NotImplementedError(
                "AllSpark on ROCm requires an RDNA4 device (gfx1200/gfx1201)"
            )
        if not IS_ROCM and not allspark_supported():
            return False, NotImplementedError("AllSpark requires CUDA compute capability 80-120")
        return True, None

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
        # Pick the kernel family once: the packed layouts are not interchangeable
        # and a wrong pairing produces garbage rather than an error.
        self._hip = IS_ROCM
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
        qweight = self.qweight.data
        self.validate_device(qweight.device)
        k4, n = qweight.shape
        k = k4 * self.pack_factor
        shifts = (torch.arange(self.pack_factor, device=qweight.device, dtype=torch.int32)
                  * 8).view(self.pack_factor, 1)
        codes = ((qweight.view(k4, 1, n) >> shifts) & 0xFF).reshape(k, n).to(torch.uint8)

        scales = self.scales.data.reshape(-1).contiguous()
        if self._hip:
            with torch.cuda.device(qweight.device):
                if not prewarm_allspark_hip_extension():
                    raise RuntimeError("AllSpark HIP extension failed to build")
                packed, packed_scales, _ = allspark_hip_repack(
                    codes.contiguous(), scales, self.out_features, self.in_features
                )
        else:
            if not prewarm_allspark_extension():
                raise RuntimeError("AllSpark extension failed to build")
            packed, packed_scales, _ = allspark_repack(
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
        x2d = x.reshape(-1, self.in_features).contiguous()
        if self._hip:
            out = allspark_hip_gemm(
                x2d, self.qweight_packed, self.scales_packed, self.out_features
            )
        else:
            out = allspark_gemm(x2d, self.qweight_packed, self.scales_packed, self.out_features)
        out = out.reshape(shape)
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)
