# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Range-aware decode dtype selection for AWQ dense fallbacks."""

import torch

from ...utils.torch import dynamo_is_compiling
from ._numerics import needs_wide_bias_epilogue, scales_narrowed_by
from ._runtime_cache import tensor_cache_key


# The dtypes `_decode_dtype_for_activation` can ask about: the two half
# activation dtypes, plus the FP32 fallback. Their range answers are computed
# eagerly so a compiled graph can read them as plain Python constants (#4794).
_DECODE_RANGE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class AwqDecodeRangeMixin:
    """Choose a decode dtype that keeps every decoded AWQ weight representable.

    The checkpoint stores one scale per group and integer codes in
    ``[0, 2**bits - 1]``, so a decoded weight is ``(code - zero) * scale``
    with magnitude at most ``(2**bits - 1) * |scale|``. Decoding into a half
    activation dtype is range-safe only when that bound is finite in the dtype
    and the smallest nonzero scale still produces a nonzero weight. Above the
    largest finite value the weight becomes ``inf``; below the smallest
    subnormal it becomes ``0``. Either corruption happens inside the decoder,
    before any FP32 accumulation can help.

    The bound is conservative: it does not decode the packed codes, so it may
    pick a wide decode for a layer whose actual codes stay in range. A false
    positive spends a wider decode instead of returning wrong output.
    """

    # Defaults make the mixin usable before the first explicit refresh. A
    # caller that wants a compiled forward to match eager must call
    # ``_refresh_decode_range_fits`` after the scales are final.
    _decode_range_key = None
    _decode_range_fits = True
    _decode_range_fits_by_dtype = None

    def _stored_decode_range_fits(self, dtype: torch.dtype) -> bool:
        """Whether every representable decoded weight stays in ``dtype``'s range."""

        scales = getattr(self, "scales", None)
        if scales is None or scales.numel() == 0 or scales.device.type == "meta":
            return True

        # A host-side value would become a data-dependent branch in a compiled
        # graph and cannot be read during CUDA graph capture. Read the answer
        # computed outside the compiled region instead of returning `False` for
        # every dtype: `False` here escalates the decode to FP64, and FP64
        # promotes the whole GEMM to FP64 (#4794). The check must still precede
        # the cache key: `tensor_cache_key` reads `.is_inference()`/`data_ptr()`
        # on the host, which dynamo rejects and a CUDA-graph capture cannot
        # cross.
        if dynamo_is_compiling() or (
            scales.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        ):
            precomputed = getattr(self, "_decode_range_fits_by_dtype", None)
            if precomputed is None:
                return False
            return precomputed.get(dtype, False)

        key = (tensor_cache_key(scales), dtype, self.bits)
        if key == self._decode_range_key:
            return self._decode_range_fits

        fits = self._compute_decode_range_fits(dtype)
        self._decode_range_key = key
        self._decode_range_fits = fits
        # Keep the compiled-region answer in step with an eager recompute (a
        # caller that mutates the scales in place and calls an eager forward
        # before compiling).
        precomputed = getattr(self, "_decode_range_fits_by_dtype", None)
        if precomputed is not None:
            precomputed[dtype] = fits
        return fits

    def _compute_decode_range_fits(self, dtype: torch.dtype) -> bool:
        """Whether the stored scales keep every decoded weight in ``dtype``.

        Check the actual cast the decoder will perform, in the decoder's dtype:
        a scale that overflows, or a nonzero scale that rounds to zero, is not
        representable there, and neither is a scale whose largest code
        (`2**bits - 1`) makes the product overflow. The reduction runs on the
        scale's device, so this must never be called from a compiled region or
        a CUDA-graph capture.
        """

        span = (1 << self.bits) - 1
        magnitude = self.scales.detach().abs()
        converted = magnitude.to(dtype)
        # One reduction (one host sync), not three: this also runs for every
        # dtype on each device/dtype move.
        representable = (
            converted.isfinite()
            & (converted * span).isfinite()
            & ~((magnitude > 0) & (converted == 0))
        )
        return bool(representable.all())

    def _refresh_decode_range_fits(self) -> None:
        """Recompute the per-dtype answers read inside a compiled graph.

        A compiled graph cannot read the scale tensor on the host, so the answer
        it needs is decided here, eagerly, and read back as a plain Python
        constant (dynamo guards on it, so a change recompiles). This runs
        wherever the scales can change: ``post_init()``, a device/dtype move and
        a state-dict load (both via ``_apply``/``_load_from_state_dict`` ->
        ``clear_weight_cache()``), and a training transition. An in-place
        mutation of the scales must call ``clear_weight_cache()`` explicitly,
        the same contract ``tensor_cache_key`` documents.
        """

        scales = getattr(self, "scales", None)
        if scales is None or scales.numel() == 0 or scales.device.type == "meta":
            self._decode_range_fits_by_dtype = dict.fromkeys(_DECODE_RANGE_DTYPES, True)
            return
        self._decode_range_fits_by_dtype = {
            dtype: self._compute_decode_range_fits(dtype)
            for dtype in _DECODE_RANGE_DTYPES
        }

    def _checkpoint_scale_dtype(self) -> torch.dtype | None:
        """The stored scale dtype, ignoring a runtime in-place widening.

        ``TorchCpuAvx512AwqLinear`` widens its scales to FP32 in place when it
        installs the fused layout, so the live buffer no longer describes the
        checkpoint precision. That class records it separately and overrides
        this hook.
        """

        scales = getattr(self, "scales", None)
        return None if not isinstance(scales, torch.Tensor) else scales.dtype

    def _wide_bias_needs_stored_scales(self, activation_dtype: torch.dtype) -> bool:
        """Whether a wide bias epilogue needs the checkpoint scale precision.

        A bias stored wider than the activation is added to the FP32 product
        before the single output cast (``linear_wide_bias``), so the decoder is
        the last place a scale can be rounded. Narrowing an FP32 checkpoint
        scale to a half dtype there loses bits that no later epilogue can
        recover: an exactly representable FP16 output comes back as zero when
        the bias cancels the product (#5764). The test is dtype-only so a
        compiled graph or a CUDA-graph capture keeps a plain Python constant.
        """

        return needs_wide_bias_epilogue(
            getattr(self, "bias", None), activation_dtype
        ) and scales_narrowed_by(self._checkpoint_scale_dtype(), activation_dtype)

    def _decode_dtype_for_activation(self, activation_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype that decodes the checkpoint without precision loss.

        FP32 and wider activations already decode at their own precision.
        FP16/BF16 activations decode in their own dtype only while the stored
        scales keep every decoded weight finite and nonzero-representable, and
        only when the activation dtype can hold those scale values exactly;
        otherwise the decode uses FP32 and the product accumulates wide, so a
        stored scale is never rounded away inside the decoder. The decision is
        keyed by the live scale tensor, so a checkpoint load, a device/dtype
        move, or a ``clear_weight_cache()`` selects the next decode instead of
        reusing a stale answer.
        """

        if activation_dtype not in (torch.float16, torch.bfloat16):
            return activation_dtype
        scale_is_lossy = self._wide_bias_needs_stored_scales(activation_dtype)
        if self._stored_decode_range_fits(activation_dtype) and not scale_is_lossy:
            return activation_dtype
        # FP32 holds every FP16 checkpoint scale exactly and covers the whole
        # BF16 scale range in every ordinary case. Only a checkpoint whose
        # scale magnitude exceeds the FP32 weight range (a BF16 scale above
        # ~2.3e37 at four bits) needs the wider decode.
        if self._stored_decode_range_fits(torch.float32):
            return torch.float32
        return torch.float64


__all__ = ["AwqDecodeRangeMixin"]
