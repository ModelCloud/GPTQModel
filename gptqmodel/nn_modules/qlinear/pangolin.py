# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Pangolin fused GPU kernels for the ``gptq_p`` format.

Pangolin is the fastest path for decode-shape (small-M) planar inputs on
NVIDIA compute capability >= 8.0.  It falls back to the planar Triton fused
GEMV / dequant+matmul paths when the native extension is unavailable, the
g_idx is not block-uniform, or the input is not decode-shaped.
"""

import weakref

import torch

from ...adapter.adapter import Lora
from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from . import FormatSupport
from .tritonv2 import TritonV2Linear


class PangolinQuantLinear(TritonV2Linear):
    """Quantized linear layer backed by the native Pangolin planar GEMV kernel.

    CUDA handles split-plane 3/5/6/7-bit weights. Apple Metal handles every
    2--8-bit ``gptq_p`` layout in one fused packed-weight GEMV, including the
    bit-identical continuous 2/4/8-bit layouts.
    """

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_PANGOLIN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    # gptq_p planar bits; higher priority than TritonV2Linear so AUTO picks
    # Pangolin for decode-shaped planar layers when the runtime is available.
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.GPTQ_P: FormatSupport(priority=50, bits=(2, 3, 4, 5, 6, 7, 8)),
    }
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 96, 128, 192, 256, 384, 512, 1024]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.MPS]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32, PLATFORM.DARWIN]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.int16, torch.int8]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = True

    QUANT_TYPE = "pangolin"

    @classmethod
    def validate_once(cls):
        from ...utils.pangolin_mps import pangolin_mps_supported

        if pangolin_mps_supported():
            return True, None
        return super().validate_once()

    @classmethod
    def validate(cls, **args):
        device = args.get("device")
        if device == DEVICE.MPS:
            from ...utils.pangolin_mps import pangolin_mps_supported

            if not pangolin_mps_supported():
                return False, NotImplementedError(
                    "Pangolin Metal requires torch.mps.compile_shader support."
                )
        valid, error = cls.cached_validate_once()
        if not valid:
            return valid, error
        # TritonV2's generic fused 3-bit path requires symmetric weights, but
        # Pangolin Metal consumes the stored qzeros and supports affine GPTQ-P.
        # Device is absent during construction and is validated explicitly by
        # the loader before construction, as with Apple-only 2/4/8-bit support.
        apple_asymmetric_3bit = (
            args.get("bits") == 3
            and args.get("format") == FORMAT.GPTQ_P
            and args.get("sym") is False
            and device in (None, DEVICE.MPS)
        )
        if apple_asymmetric_3bit:
            valid, error = cls._validate(**args)
        else:
            valid, error = super().validate(**args)
        if not valid:
            return valid, error
        if device == DEVICE.MPS:
            if args.get("pack_dtype") != torch.int32:
                return False, NotImplementedError(
                    "Pangolin Metal currently supports int32 packed words only."
                )
            if args.get("dtype") != torch.float16:
                return False, NotImplementedError(
                    "Pangolin Metal currently supports float16 inference only."
                )
            if args.get("trainable"):
                return False, NotImplementedError(
                    "Pangolin Metal currently supports inference only."
                )
        # Constructor-time validation does not carry the selected device. The
        # loader performs the device-aware validation before construction, so
        # reject this combination only when a non-MPS device is explicit.
        elif device is not None and args.get("bits") in (2, 4, 8):
            return False, NotImplementedError(
                "Pangolin CUDA supports split-plane 3/5/6/7-bit weights; "
                "2/4/8-bit Pangolin is currently Apple Metal only."
            )
        return True, None

    def post_init(self):
        super().post_init()
        if self.g_idx.device.type == "mps":
            from ...utils.pangolin import g_idx_block_uniform

            self._pangolin_g_idx_ref = weakref.ref(self.g_idx)
            self._pangolin_g_idx_version = self.g_idx._version
            self._pangolin_g_idx_block_uniform = g_idx_block_uniform(self.g_idx)

    def forward(self, x):
        if x.device.type != "mps" or self.training:
            return super().forward(x)

        from ...utils.pangolin_mps import pangolin_mps_gemv

        x = self._apply_rotation_to_input(x)
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.shape[0] == 0:
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)
        if self.padded_in_features != self.in_features:
            x_flat = torch.nn.functional.pad(
                x_flat, (0, self.padded_in_features - self.in_features)
            )
        block_uniform = (
            getattr(self, "_pangolin_g_idx_ref", lambda: None)() is self.g_idx
            and getattr(self, "_pangolin_g_idx_version", -1) == self.g_idx._version
            and getattr(self, "_pangolin_g_idx_block_uniform", False)
            and not (x_flat.shape[0] == 1 and self.bits == 2)
        )
        out = pangolin_mps_gemv(
            x_flat.contiguous(),
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            planar=self.planar,
            _g_idx_validated=True,
            _g_idx_block_uniform=block_uniform,
        )[:, : self.out_features].reshape(out_shape)
        if self.bias is not None:
            out.add_(self.bias)
        if self.adapter:
            out = self.adapter.apply(x=x, out=out)
        return out.to(dtype=x.dtype)


__all__ = ["PangolinQuantLinear"]
