# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models._const import DEVICE
from ...quantization.dtype import dequantize_f4_e2m1
from ...utils.torch import dynamo_is_compiling
from ._numerics import (
    bias_narrows_losslessly,
    linear_needs_wide_bias,
    linear_wide_bias,
    promote_bias,
)


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except Exception:
    NVFP4Tensor = None


# Bound for the transient decoded-weight scratch of the wide fallback. A full
# FP32 copy of a large projection would be hundreds of MiB, so the fallback
# decodes output-channel tiles and releases each one after its own product.
_WIDE_DECODE_TILE_BYTES = 64 * 1024 * 1024

# Magnitudes of the E2M1 code set. ``6`` is the largest and ``0.5`` the smallest
# non-zero code, so ``scale * code`` for one block is bounded by these.
_E2M1_MAX_CODE = 6.0
_E2M1_MIN_CODE = 0.5


def _tensor_version(tensor: torch.Tensor) -> int:
    """Return ``tensor._version`` or ``-1`` when the counter is unavailable.

    A checkpoint tensor materialized under ``torch.inference_mode`` raises
    ``RuntimeError`` for this attribute. That state cannot observe an in-place
    rewrite either, so it reuses the recorded answer exactly like a captured
    graph does instead of failing the forward.
    """

    try:
        return int(tensor._version)
    except RuntimeError:
        return -1

# E4M3 itself bounds an NVFP4 block scale to ``[2**-9, 448]``. ``448 * 6`` and
# ``2**-9 * 0.5`` are both inside FP16's finite range, so a checkpoint that
# stores its scales in E4M3 needs no scan at all. Every other scale dtype
# (E8M0, or the FP32 effective scale after a ModelOpt fold) can leave the range
# and is checked.
_BOUNDED_SCALE_DTYPES = tuple(
    dtype for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
    ) if dtype is not None
)


class TorchFP4Linear(nn.Module):
    """Execute one linear layer directly from NVFP4-packed checkpoint weights."""

    # CPU uses the portable decode; accelerator execution uses torchao NVFP4,
    # whose NVIDIA path has no validated HIP implementation.
    SUPPORTS_DEVICES = [DEVICE.CPU, DEVICE.CUDA]

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_block_size: int,
        orig_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_block_size = int(weight_block_size)
        self.orig_dtype = orig_dtype
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale)
        # Recorded for graph capture/tracing and reused on accelerator devices,
        # where re-deriving the answer would synchronize the host every forward.
        # CPU forwards re-derive it from the live buffer on every call.
        self._narrow_decode_scale_key: Optional[tuple[torch.dtype, int, str]] = None
        self._narrow_decode_at_load = self._narrow_decode_decidable(orig_dtype)
        if isinstance(bias, torch.Tensor):
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def _native_weight(self) -> "NVFP4Tensor":
        """Wrap the stored packed weight buffers into the torchao NVFP4 tensor view."""

        if NVFP4Tensor is None:
            raise RuntimeError("TorchFP4Linear requires torchao NVFP4Tensor support.")
        return NVFP4Tensor(
            self.weight,
            self.weight_scale,
            block_size=self.weight_block_size,
            orig_dtype=self.orig_dtype,
        )

    def _epilogue_bias(
        self, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor | None, bool]:
        """Return the bias for the narrow epilogue plus whether it must stay wide.

        ``TorchFP4Linear`` keeps the checkpoint bias at its stored width, which
        is normally FP32. The epilogue only has to go wide when a stored value
        is not exactly representable in the activation dtype: an exact cast
        returns the same bias, so the cheap low-precision epilogue still applies.
        The comparison is a few microseconds against a full-weight decode, so it
        is computed per call rather than cached; caching it would go stale after
        an in-place ``.data`` write, which does not bump the version counter.
        """

        if not linear_needs_wide_bias(x, bias):
            return promote_bias(x, bias), False
        if bias_narrows_losslessly(bias, x.dtype):
            return bias.to(dtype=x.dtype), False
        return bias, True

    def _narrow_decode_decidable(self, dtype: torch.dtype) -> bool:
        """Return :meth:`_narrow_decode_in_range` for the stored scale grid."""

        if not dtype.is_floating_point or dtype in (torch.float32, torch.float64):
            return True
        scale = self.weight_scale
        if scale.numel() == 0 or scale.dtype in _BOUNDED_SCALE_DTYPES:
            return True
        info = torch.finfo(dtype)
        # Smallest sub-normal: a non-zero weight below it decodes to zero.
        smallest = info.tiny * info.eps
        # The stored scale may be an FP8 dtype, which has no `abs` kernel.
        magnitudes = scale.to(torch.float32).abs()
        low, high = torch.aminmax(magnitudes)
        low = float(low)
        if low == 0.0:
            # A zero block decodes to zeros, so it binds neither bound; find the
            # smallest non-zero scale for the underflow test.
            nonzero = magnitudes[magnitudes > 0]
            if nonzero.numel() == 0:
                return True
            low = float(nonzero.amin())
        return (
            float(high) * _E2M1_MAX_CODE <= info.max
            and low * _E2M1_MIN_CODE >= smallest
        )

    def _narrow_decode_in_range(self, dtype: torch.dtype) -> bool:
        """Whether decoding the weight directly to ``dtype`` is lossless in range.

        The effective weight is ``e2m1_code * effective_scale``, and ModelOpt
        folds ``weight_scale_2`` into that scale, so neither term is bounded by
        the E4M3 range of the stored ``weight_scale``. One block spans
        ``[0.5, 6] * scale``; if any bound leaves ``dtype``'s finite range or
        collapses a non-zero weight to zero, decoding straight to ``dtype``
        destroys information that the final, representable dot product needed.
        Only the pack-level scale extremes are inspected, never the decoded
        weight, and an E4M3-stored scale is bounded by E4M3 itself
        (``[2**-9, 448]``), whose extremes still fit FP16. A CPU forward
        re-reads the live scale every call, so even an
        in-place ``.data`` write (which does not bump a version counter) is
        observed. A device forward must round-trip the reduction to the host,
        so it reuses the recorded answer until the stored scale is rewritten or
        moved (reported by its version counter).

        A captured or traced graph cannot reduce the scale on the host at all,
        so it replays the load-time answer, exactly like
        ``bias_narrows_losslessly`` keeps the wide epilogue when compiling.
        Mutating the scale after load is therefore observed eagerly but not
        inside an already-captured graph.
        """

        if dynamo_is_compiling():
            return self._narrow_decode_at_load
        scale = self.weight_scale
        if scale.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return self._narrow_decode_at_load
        if scale.device.type == "cpu":
            # No host sync, so read the live buffer and observe `.data` writes
            # that do not bump a version counter.
            return self._narrow_decode_decidable(dtype)
        # A device reduction must come back to the host to branch. Reuse the
        # recorded answer until the stored scale is rewritten or moved, which
        # is what a version bump reports.
        key = (dtype, _tensor_version(scale), str(scale.device))
        if key != self._narrow_decode_scale_key:
            self._narrow_decode_at_load = self._narrow_decode_decidable(dtype)
            self._narrow_decode_scale_key = key
        return self._narrow_decode_at_load

    def _block_scaled_tiles(self) -> bool:
        """Whether ``weight``/``weight_scale`` form one scale per input block."""

        packed = self.weight
        scale = self.weight_scale
        block_size = self.weight_block_size
        return (
            packed.dim() == 2
            and scale.dim() == 2
            and scale.shape[0] == packed.shape[0]
            and scale.shape[1] > 0
            and block_size > 0
            and packed.shape[1] * 2 == scale.shape[1] * block_size
        )

    def _wide_weight_tile(self, start: int, stop: int) -> torch.Tensor:
        """Decode output channels ``[start, stop)`` to FP32 without narrowing.

        Every E2M1 code is exact in BF16, so decoding the codes and applying
        the FP32 block scale afterwards performs a single rounding of the
        effective weight instead of one that can saturate to ``inf`` or
        underflow to zero before the scale is applied.
        """

        packed = self.weight[start:stop]
        scale = self.weight_scale[start:stop]
        if self._block_scaled_tiles():
            rows, blocks = scale.shape
            codes = dequantize_f4_e2m1(
                packed,
                scale=None,
                axis=None,
                target_dtype=torch.bfloat16,
            ).to(torch.float32)
            # Apply the block scale in place so the tile costs one FP32 buffer
            # plus the exact BF16 codes, not three copies of the weight.
            blocks_view = codes.view(rows, blocks, self.weight_block_size)
            blocks_view.mul_(scale.to(torch.float32).unsqueeze(-1))
            return blocks_view.view(rows, -1)
        # Any other scale layout is decoded by the shared reference so its
        # broadcast rules apply to the whole weight at once.
        return dequantize_f4_e2m1(
            packed,
            scale=scale,
            axis=None,
            target_dtype=torch.float32,
        )

    def _wide_forward(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Multiply, add the stored bias, and round once to ``x.dtype``.

        The decoded weight, the reduction, and the bias stay wide, so a
        representable result is never produced from a weight that saturated to
        ``inf`` or underflowed to zero. Each dense tile is local to its own
        product, so no expanded weight survives the call.
        """

        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        if self.weight.dim() != 2:
            # A pack without the ``[out_features, in_features / 2]`` layout has
            # no output-channel axis to tile; decode it in one piece.
            weight = dequantize_f4_e2m1(
                self.weight,
                scale=self.weight_scale,
                axis=None,
                target_dtype=compute_dtype,
            )
            return linear_wide_bias(x, weight, bias)
        out_features = int(self.weight.shape[0])
        out_shape = x.shape[:-1] + (out_features,)
        if x.numel() == 0 or out_features == 0:
            output = torch.empty(out_shape, dtype=compute_dtype, device=x.device)
        else:
            flat_in = x.shape[-1]
            # A layout the tiled decoder cannot express is decoded in one piece.
            tile_rows = (
                max(1, _WIDE_DECODE_TILE_BYTES // max(1, flat_in * 4))
                if self._block_scaled_tiles()
                else out_features
            )
            # Autocast would narrow the FP32 operands again, which is exactly
            # the round this path exists to defer.
            with torch.autocast(device_type=x.device.type, enabled=False):
                flat_x = x.reshape(-1, flat_in).to(compute_dtype)
                pieces = [
                    torch.mm(
                        flat_x,
                        self._wide_weight_tile(start, stop).to(compute_dtype).t(),
                    )
                    for start, stop in (
                        (start, min(out_features, start + tile_rows))
                        for start in range(0, out_features, tile_rows)
                    )
                ]
                output = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
                output = output.reshape(out_shape)
                if isinstance(bias, torch.Tensor):
                    output = output + bias.to(device=output.device, dtype=compute_dtype)
        return output.to(x.dtype)

    def _cpu_forward(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        """Decode live FP4 buffers on CPU and round only the visible output."""

        if not self._narrow_decode_in_range(x.dtype):
            return self._wide_forward(x, bias)
        weight = dequantize_f4_e2m1(
            self.weight,
            scale=self.weight_scale,
            axis=None,
            target_dtype=x.dtype,
        )
        bias, wide = self._epilogue_bias(x, bias)
        if wide:
            return linear_wide_bias(x, weight, bias)
        return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch one dense linear projection through the NVFP4 tensor wrapper."""

        bias = self.bias
        if x.device.type == "cpu" and self.weight.device.type == "cpu":
            return self._cpu_forward(x, bias)
        # The torchao wrapper dequantizes at ``orig_dtype``; an activation of any
        # other dtype has no native execution and would round the weight twice.
        if x.dtype != self.orig_dtype or not self._narrow_decode_in_range(x.dtype):
            return self._wide_forward(x, bias)
        bias, wide = self._epilogue_bias(x, bias)
        if wide:
            # The activation dtype cannot represent the stored bias, and the
            # torchao wrapper would narrow it before the reduction. Decode the
            # packed buffers wide only for this call and keep the epilogue wide.
            return self._wide_forward(x, bias)
        return F.linear(x, self._native_weight(), bias)
