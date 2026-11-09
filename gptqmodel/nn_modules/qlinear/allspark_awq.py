# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSpark AWQ W8A16 linear (RDNA4 gfx1200/gfx1201 and Ampere+ CUDA).

The AllSpark GEMM already had a zero-point branch (``has_zp``); this module adds
the missing half: AWQ's on-disk layout. GPTQ-8bit and AWQ-8bit differ only by a
per-group zero point, so both share one kernel:

* GPTQ 8-bit: codes packed along K, neutral zero point 128, per-channel scale.
* AWQ 8-bit: codes packed along N, arbitrary per-group zero point and scale.

The AWQ tensors are unpacked with the shared AWQ packer helpers, then repacked
into the fragment order of the active extension: the WMMA order in
``gptqmodel_ext/allspark_hip`` on ROCm, the N32K16 order in
``gptqmodel_ext/allspark`` on CUDA. Only the packed form is retained; the dense
weight never exists outside ``post_init``.

The two kernels apply the zero point at different points, which is the one
device-specific detail here:

* HIP subtracts the zero inside the fp32 dequant, ``(code - zero) * scale``, so
  it consumes the raw AWQ zero code.
* CUDA subtracts it after its bias-128 conversion, ``((code - 128) - zero) *
  scale``, so it consumes ``zero - 128``.
"""

from __future__ import annotations

import torch

from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.allspark import (
    ALLSPARK_K_ALIGN,
    ALLSPARK_N_ALIGN,
    allspark_gemm,
    allspark_repack_awq,
    allspark_supported,
    prewarm_allspark_extension,
)
from ...utils.allspark_hip import (
    allspark_hip_available,
    allspark_hip_gemm,
    allspark_hip_repack_awq,
    prewarm_allspark_hip_extension,
    validate_allspark_hip_device,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM


log = setup_logger()


class AllSparkAwqLinear(AWQuantLinear):
    # AWQ sub-kernel of the shared AllSpark backend. ``BACKEND.ALLSPARK`` plus
    # the checkpoint method/format selects this class; no AWQ-specific backend
    # name is required.
    SUPPORTS_BACKENDS = [BACKEND.ALLSPARK]
    SUPPORTS_METHODS = [METHOD.AWQ]
    # 0 keeps the format registered but opts out of AUTO selection, like the
    # GPTQ module: the numbers come first, promotion to AUTO later.
    SUPPORTS_FORMATS = {FORMAT.GEMM: 0}
    SUPPORTS_BITS = [8]
    # Per-channel only: one scale/zero row covers all of K.
    SUPPORTS_GROUP_SIZE = [-1]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [ALLSPARK_K_ALIGN]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [ALLSPARK_N_ALIGN]
    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "allspark"

    @classmethod
    def validate_device(cls, device):
        super().validate_device(device)
        if IS_ROCM:
            validate_allspark_hip_device(device)

    @classmethod
    def validate_once(cls):
        if IS_ROCM:
            if not allspark_hip_available():
                return False, NotImplementedError(
                    "AWQ AllSpark requires an RDNA4 device (gfx1200/gfx1201)"
                )
            return True, None
        if not allspark_supported():
            return False, NotImplementedError(
                "AWQ AllSpark requires an Ampere+ CUDA device (compute capability 80-120)"
            )
        return True, None

    def __init__(
        self,
        bits: int,
        group_size: int,
        sym: bool,
        desc_act: bool,
        in_features: int,
        out_features: int,
        bias: bool = False,
        pack_dtype: torch.dtype = torch.int32,
        adapter=None,
        register_buffers: bool = False,
        **kwargs,
    ):
        if bits != 8:
            raise ValueError(f"AWQ AllSpark supports bits=8 only, got {bits}")
        self.compute_dtype = kwargs.get("dtype") or torch.float16
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.ALLSPARK),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        if register_buffers:
            if self.scales is not None and self.scales.dtype != self.compute_dtype:
                self.scales = self.scales.to(dtype=self.compute_dtype)
            if self.bias is not None and self.bias.dtype != self.compute_dtype:
                self.bias = self.bias.to(dtype=self.compute_dtype)
        self._repacked = False
        # Pick the kernel family once. The two extensions use different packed
        # layouts (WMMA fragments on ROCm, N32K16 on CUDA), and pairing a buffer
        # with the wrong kernel yields garbage rather than an error.
        self._hip = IS_ROCM

    @property
    def _kernel_group_size(self) -> int:
        # The kernel keeps one scale/zero row per output channel, so the only
        # accepted grouping is "all of K". ``group_size == in_features`` is how
        # the base class spells -1 internally.
        return -1

    def pack(self, linear: torch.nn.Module, scales: torch.Tensor,
             zeros: torch.Tensor, g_idx: torch.Tensor = None):
        """Quantise a dense weight into the AWQ 8-bit on-disk layout.

        Required because the inherited GPTQ packer packs along K; AWQ packs
        along N. Uses the shared ``pack_awq`` helper so a checkpoint written
        here is byte-compatible with the other AWQ backends.
        """
        from ...quantization.awq.utils.packing_utils import pack_awq

        del g_idx
        assert scales is not None and zeros is not None

        if self.requested_group_size not in (-1, self.in_features):
            raise ValueError(
                "AWQ AllSpark supports group_size=-1 only, got "
                f"{self.requested_group_size}"
            )
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales

        scale_dtype = self.scales.dtype if getattr(self, "scales", None) is not None else self.compute_dtype
        self.register_buffer("scales", scales.clone().to(scale_dtype))
        if linear.bias is not None:
            bias_dtype = (linear.bias.dtype
                          if linear.bias.dtype in (torch.float16, torch.bfloat16)
                          else scale_dtype)
            self.register_buffer("bias", linear.bias.clone().to(bias_dtype))
        else:
            self.bias = None

        weight = linear.weight.data
        group_index = torch.arange(self.in_features, device=weight.device) // self.group_size
        intweight = torch.round(
            (weight.t() + scale_zeros.index_select(0, group_index))
            / self.scales.index_select(0, group_index)
        ).clamp_(0, self.maxq).to(torch.int32).contiguous()

        qweight, qzeros = pack_awq(intweight, zeros.to(torch.int32), self.bits)
        self.register_buffer("qweight", qweight)
        self.register_buffer("qzeros", qzeros)

    @torch.no_grad()
    def post_init(self):
        if self._repacked:
            return

        self.validate_device(self.qweight.device)
        # Honor dtype moves performed after construction, before repacking.
        scales = self.scales.data.reshape(-1, self.out_features).contiguous()
        if self._hip:
            with torch.cuda.device(self.qweight.device):
                if not prewarm_allspark_hip_extension():
                    raise RuntimeError("AllSpark HIP extension failed to build")
                packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
                    self.qweight.data,
                    self.qzeros.data,
                    scales,
                    self.out_features,
                    self.in_features,
                    self._kernel_group_size,
                )
        else:
            if not prewarm_allspark_extension():
                raise RuntimeError("AllSpark CUDA extension failed to build")
            packed, packed_scales, packed_zeros = allspark_repack_awq(
                self.qweight.data,
                self.qzeros.data,
                scales,
                self.out_features,
                self.in_features,
                self._kernel_group_size,
            )
        self.register_buffer("qweight_packed", packed, persistent=False)
        self.register_buffer("scales_packed", packed_scales, persistent=False)
        self.register_buffer("zeros_packed", packed_zeros, persistent=False)
        self._repacked = True
        del self.qweight, self.qzeros, self.scales
        self.qweight = None
        self.qzeros = None
        self.scales = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._repacked:
            self.post_init()
        shape = x.shape[:-1] + (self.out_features,)
        x2d = x.reshape(-1, self.in_features).contiguous()
        if self._hip:
            out = allspark_hip_gemm(
                x2d,
                self.qweight_packed,
                self.scales_packed,
                self.out_features,
                b_qzeros=self.zeros_packed,
                group_size=self._kernel_group_size,
            )
        else:
            out = allspark_gemm(
                x2d,
                self.qweight_packed,
                self.scales_packed,
                self.out_features,
                b_qzeros=self.zeros_packed,
            )
        out = out.reshape(shape)
        bias = getattr(self, "bias", None)
        if bias is not None:
            out = out + bias
        return out.to(x.dtype)
