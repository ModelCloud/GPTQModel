# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AllSpark AWQ W8A16 linear for RDNA4 (gfx1200 / gfx1201) only.

The AllSpark GEMM already had a zero-point branch (``has_zp``); this module adds
the missing half: AWQ's on-disk layout. GPTQ-8bit and AWQ-8bit differ only by a
per-group zero point, so both share one kernel:

* GPTQ 8-bit: codes packed along K, neutral zero point 128, per-channel scale.
* AWQ 8-bit: codes packed along N, arbitrary per-group zero point and scale.

The AWQ tensors are unpacked with the shared AWQ packer helpers, then repacked
into the WMMA fragment order used by ``gptqmodel_ext/allspark_hip``. Only the
packed form is retained; the dense weight never exists outside ``post_init``.

CUDA (Ampere) is deliberately not offered: the CUDA AllSpark kernel applies the
zero point *after* its bias-128 conversion, so it expects a pre-adjusted
``zero - 128`` buffer, and that path is not exercised by this PR.
"""

from __future__ import annotations

import torch

from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import AWQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.allspark_hip import (
    ALLSPARK_HIP_K_ALIGN,
    ALLSPARK_HIP_N_ALIGN,
    allspark_hip_gemm,
    allspark_hip_repack_awq,
    allspark_hip_supported,
    prewarm_allspark_hip_extension,
)
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM


log = setup_logger()


def _device_type(device) -> str:
    return getattr(device, "type", device)


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
    # Per-channel only. AWQ's grouped qzeros put a different zero point on each
    # K-group, which forces a scale/zero reload per 16-wide k-tile inside the
    # GEMV and costs 2-6% at group_size=128 (~18% at 32) on gfx1201 with no
    # accuracy benefit for the decode shapes this kernel targets. With -1 there
    # is one group over all of K, so scales/zeros are (1, N) and the kernel
    # loads them once outside the k-loop.
    SUPPORTS_GROUP_SIZE = [-1]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = False
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [ALLSPARK_HIP_K_ALIGN]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [ALLSPARK_HIP_N_ALIGN]
    SUPPORTS_DEVICES = [DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = []
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    QUANT_TYPE = "allspark"

    @classmethod
    def validate_device(cls, device):
        # ROCm runs in torch's CUDA namespace, so a ``torch.device("cuda")``
        # selector is ambiguous. It means ROCm when HIP is the active runtime
        # and must be rejected otherwise: this module is RDNA4-only.
        if IS_ROCM and _device_type(device) == "cuda":
            return
        super().validate_device(device)

    @classmethod
    def validate_once(cls):
        if not allspark_hip_supported():
            return False, NotImplementedError(
                "AWQ AllSpark requires an RDNA4 device (gfx1200/gfx1201)"
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

        scale_dtype = scales.dtype if scales.dtype in (torch.float16, torch.bfloat16) else torch.float16
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
        if not prewarm_allspark_hip_extension():
            raise RuntimeError("AllSpark HIP extension failed to build")

        scales = self.scales.data.reshape(-1, self.out_features).to(self.compute_dtype)
        packed, packed_scales, packed_zeros = allspark_hip_repack_awq(
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
        x2d = x.reshape(-1, self.in_features)
        out = allspark_hip_gemm(
            x2d,
            self.qweight_packed,
            self.scales_packed,
            self.out_features,
            b_qzeros=self.zeros_packed,
            group_size=self._kernel_group_size,
        )
        out = out.reshape(shape)
        bias = getattr(self, "bias", None)
        if bias is not None:
            out = out + bias
        return out.to(x.dtype)
