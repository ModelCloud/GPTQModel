# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# ParoQuant Triton runtime implementation adapted from the ParoQuant paper and
# public project:
# https://arxiv.org/html/2511.10645v2
# https://github.com/z-lab/paroquant

"""ParoQuant Triton-backed quantized linear layer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ...models._const import DEVICE, PLATFORM
from ...quantization import FORMAT
from ...quantization.paroquant.modules.triton.gemm import (
    _paroquant_prepare_splitk_compiled_launch,
    _paroquant_rotation_gemm_splitk_triton_compiled,
    _paroquant_rotation_gemm_splitk_triton_prepared,
    _paroquant_rotation_gemm_splitk_triton_prepare,
    _paroquant_rotation_gemm_splitk_triton_unchecked,
    _paroquant_splitk_compiled_launch_supported,
    _paroquant_splitk_fp16_prefill_shape,
    _paroquant_splitk_launch_config,
    _paroquant_splitk_output_config,
    _paroquant_triton_current_stream,
    paroquant_dequantize_triton,
    paroquant_gemm_triton_decode,
    paroquant_gemm_triton_prefill,
    paroquant_rotation_gemm_triton_decode,
    paroquant_rotation_gemm_triton_prefill,
)
from ...utils import has_gil_disabled
from ...utils.backend import BACKEND
from ...utils.paroquant import build_paroquant_rotation_lookup
from .paroquant import ParoLinear


def _tensor_runtime_version(tensor: torch.Tensor) -> int | None:
    """Read a mutation version when the tensor tracks one."""
    return None if tensor.is_inference() else tensor._version


@dataclass(eq=False, slots=True)
class _MegakernelSplitKPreparedState:
    """Module-local live-buffer guard around one prepared compiled launch."""

    compiled_kernel: object
    prepared_launch: object
    plan_cache_key: tuple[object, ...]
    qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: torch.Tensor
    bias: torch.Tensor | None
    pairs: torch.Tensor
    pairs_version: int | None
    theta: torch.Tensor
    theta_version: int | None
    source_channel_scales: torch.Tensor
    source_channel_scales_version: int | None


class ParoQuantTritonLinear(ParoLinear):
    """Use Triton fused kernels for ParoQuant prefill/decode execution."""

    SUPPORTS_BACKENDS = [BACKEND.PAROQUANT_TRITON]
    SUPPORTS_METHODS = ParoLinear.SUPPORTS_METHODS
    SUPPORTS_FORMATS = {FORMAT.PAROQUANT: 0}
    SUPPORTS_BITS = ParoLinear.SUPPORTS_BITS
    SUPPORTS_GROUP_SIZE = ParoLinear.SUPPORTS_GROUP_SIZE
    SUPPORTS_DESC_ACT = ParoLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = ParoLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = ParoLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = ParoLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = ParoLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = ParoLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = ParoLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = ParoLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = ParoLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = ParoLinear.SUPPORTS_DTYPES
    QUANT_TYPE = "awq_paroquant_triton"
    PAROQUANT_TRITON_AUTOTUNE = os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE", "1") != "0"
    PAROQUANT_TRITON_AUTOTUNE_WARMUP = max(0, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE_WARMUP", "5")))
    PAROQUANT_TRITON_AUTOTUNE_ITERS = max(1, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE_ITERS", "20")))
    PAROQUANT_TRITON_AUTOTUNE_MARGIN = max(0.0, float(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_AUTOTUNE_MARGIN", "0.05")))
    PAROQUANT_TRITON_DECODE_MAX_ROWS = max(1, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_DECODE_MAX_ROWS", "8")))
    PAROQUANT_TRITON_MEGAKERNEL = os.environ.get("GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL", "1") != "0"
    PAROQUANT_TRITON_MEGAKERNEL_DECODE_SPLITK = (
        os.environ.get("GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_DECODE_SPLITK", "1") != "0"
    )
    PAROQUANT_TRITON_MEGAKERNEL_PREFILL_SPLITK = (
        os.environ.get("GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_PREFILL_SPLITK", "1") != "0"
    )
    PAROQUANT_TRITON_MEGAKERNEL_DECODE_COMPILED_LAUNCH = (
        os.environ.get("GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_DECODE_COMPILED_LAUNCH", "1") != "0"
    )
    PAROQUANT_TRITON_MEGAKERNEL_PREFILL_COMPILED_LAUNCH = (
        os.environ.get("GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_PREFILL_COMPILED_LAUNCH", "1") != "0"
    )
    PAROQUANT_TRITON_MEGAKERNEL_MAX_K = max(
        128, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_MAX_K", "12288"))
    )
    PAROQUANT_TRITON_MEGAKERNEL_PREFILL_MAX_N = max(
        8, int(os.environ.get("GPTQMODEL_PAROQUANT_TRITON_MEGAKERNEL_PREFILL_MAX_N", "4096"))
    )

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        """Validate the Triton and CUDA runtime prerequisites once per process."""
        from packaging import version
        from triton import __version__ as triton_version

        triton_v = version.parse(triton_version)

        if triton_v < version.parse("2.0.0"):
            raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")

        if has_gil_disabled() and triton_v < version.parse("3.4.0"):
            raise Exception("GIL is disabled and not compatible with current Triton. Please upgrade to Triton >= 3.4.0")

        if not torch.cuda.is_available():
            raise RuntimeError("ParoQuant Triton requires CUDA.")

        return True, None

    def __init__(self, *args, **kwargs):
        """Initialize Triton autotune settings and the per-shape plan cache."""
        kwargs.setdefault("backend", BACKEND.PAROQUANT_TRITON)
        super().__init__(*args, **kwargs)
        self.paroquant_triton_autotune_enabled = self.PAROQUANT_TRITON_AUTOTUNE
        self.paroquant_triton_autotune_warmup = self.PAROQUANT_TRITON_AUTOTUNE_WARMUP
        self.paroquant_triton_autotune_iters = self.PAROQUANT_TRITON_AUTOTUNE_ITERS
        self.paroquant_triton_autotune_margin = self.PAROQUANT_TRITON_AUTOTUNE_MARGIN
        self.paroquant_triton_decode_max_rows = self.PAROQUANT_TRITON_DECODE_MAX_ROWS
        self.paroquant_triton_megakernel_enabled = self.PAROQUANT_TRITON_MEGAKERNEL
        self.paroquant_triton_megakernel_decode_splitk_enabled = (
            self.PAROQUANT_TRITON_MEGAKERNEL_DECODE_SPLITK
        )
        self.paroquant_triton_megakernel_prefill_splitk_enabled = (
            self.PAROQUANT_TRITON_MEGAKERNEL_PREFILL_SPLITK
        )
        self.paroquant_triton_megakernel_decode_compiled_launch_enabled = (
            self.PAROQUANT_TRITON_MEGAKERNEL_DECODE_COMPILED_LAUNCH
        )
        self.paroquant_triton_megakernel_prefill_compiled_launch_enabled = (
            self.PAROQUANT_TRITON_MEGAKERNEL_PREFILL_COMPILED_LAUNCH
        )
        self.paroquant_triton_megakernel_max_k = self.PAROQUANT_TRITON_MEGAKERNEL_MAX_K
        self.paroquant_triton_megakernel_prefill_max_n = self.PAROQUANT_TRITON_MEGAKERNEL_PREFILL_MAX_N
        self._plan_cache: dict[tuple[object, ...], str] = {}
        self._megakernel_rotation_cache_key: tuple[object, ...] | None = None
        self._megakernel_partner: torch.Tensor | None = None
        self._megakernel_decode_partner: torch.Tensor | None = None
        self._megakernel_cos: torch.Tensor | None = None
        self._megakernel_sin: torch.Tensor | None = None
        self._megakernel_splitk_scratch: dict[
            tuple[object, ...],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._megakernel_splitk_last_scratch: tuple[
            torch.device,
            int,
            int,
            int,
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        self._megakernel_splitk_compiled: dict[tuple[object, ...], object | bool] = {}
        self._megakernel_splitk_prepared: dict[
            tuple[object, ...],
            _MegakernelSplitKPreparedState,
        ] = {}

    def post_init(self):
        """Defer to the shared ParoQuant setup without forcing compute dtype."""
        super().post_init()
        self._clear_megakernel_rotation_cache()

    def clear_autotune(self):
        """Drop cached plan decisions after major module state changes."""
        super().clear_autotune()
        self._plan_cache = {}

    def _apply(self, fn):
        result = super()._apply(fn)
        self._clear_megakernel_rotation_cache()
        self._plan_cache = {}
        return result

    def __getstate__(self):
        """Exclude process-local Triton launchers from serialized module state."""
        state = super().__getstate__()
        state["_megakernel_splitk_last_scratch"] = None
        state["_megakernel_splitk_compiled"] = {}
        state["_megakernel_splitk_prepared"] = {}
        return state

    def _clear_megakernel_rotation_cache(self) -> None:
        self._megakernel_rotation_cache_key = None
        self._megakernel_partner = None
        self._megakernel_decode_partner = None
        self._megakernel_cos = None
        self._megakernel_sin = None
        self._megakernel_splitk_scratch = {}
        self._megakernel_splitk_last_scratch = None
        self._megakernel_splitk_compiled = {}
        self._megakernel_splitk_prepared = {}

    def _megakernel_rotation_metadata(
        self,
        x_flat: torch.Tensor,
        *,
        decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        buffers = self._buffers
        pairs = buffers["pairs"]
        theta = buffers["theta"]
        channel_scales = buffers["channel_scales"]
        cache_key = (
            pairs.device,
            pairs.data_ptr(),
            None if pairs.is_inference() else pairs._version,
            theta.device,
            theta.data_ptr(),
            None if theta.is_inference() else theta._version,
            channel_scales.device,
            channel_scales.data_ptr(),
            None if channel_scales.is_inference() else channel_scales._version,
            self.group_size,
            self.krot,
            x_flat.dtype,
        )
        if self._megakernel_rotation_cache_key != cache_key:
            if torch.cuda.is_current_stream_capturing():
                return None
            partner, cos, sin = build_paroquant_rotation_lookup(
                pairs,
                theta,
                group_size=self.group_size,
            )
            self._megakernel_partner = partner
            local_dtype = torch.int8 if x_flat.dtype == torch.bfloat16 and self.krot == 8 else torch.int16
            # Group-local offsets are bounded by 127; use the measured dtype without changing lookup values.
            self._megakernel_decode_partner = torch.remainder(partner, self.group_size).to(dtype=local_dtype)
            self._megakernel_cos = cos
            self._megakernel_sin = sin
            self._megakernel_rotation_cache_key = cache_key
            self._clear_rotation_runtime_cache()

        if (
            self._megakernel_partner is None
            or self._megakernel_decode_partner is None
            or self._megakernel_cos is None
            or self._megakernel_sin is None
        ):
            return None

        _, channel_scales = self._ensure_rotation_runtime_dtype(x_flat.device, x_flat.dtype)
        if channel_scales is None:
            return None
        partner = self._megakernel_decode_partner if decode else self._megakernel_partner
        return partner, self._megakernel_cos, self._megakernel_sin, channel_scales

    def _large_k_decode_splitk_ready(
        self,
        x_flat: torch.Tensor,
        *,
        rows: int | None = None,
    ) -> bool:
        """Return whether a measured large-K BF16 decode schedule is available."""
        rows = int(x_flat.shape[0]) if rows is None else rows
        return (
            x_flat.device.type == "cuda"
            and self.krot == 8
            and self._megakernel_large_k_decode_splitk_factor(
                x_flat.dtype,
                rows=rows,
                in_features=self.in_features,
                out_features=self.out_features,
            )
            is not None
            and torch.cuda.get_device_properties(x_flat.device).multi_processor_count == 124
        )

    def _megakernel_ready(self, x_flat: torch.Tensor) -> bool:
        large_k_decode_splitk = self._large_k_decode_splitk_ready(x_flat)
        is_decode = (
            x_flat.shape[0] <= self.paroquant_triton_decode_max_rows
            or large_k_decode_splitk
        )
        extended_wide_prefill_candidate = (
            not is_decode
            and x_flat.device.type == "cuda"
            and x_flat.dtype == torch.float16
            and self.in_features == 2048
            and 2176 <= self.out_features <= 4096
            and self.out_features % 128 == 0
            and self.krot == 8
        )
        extended_wide_prefill = False
        if extended_wide_prefill_candidate:
            sm_count = torch.cuda.get_device_properties(x_flat.device).multi_processor_count
            n_tiles = self.out_features // 128
            bm16_ctas = ((x_flat.shape[0] + 15) // 16) * n_tiles
            bm32_ctas = ((x_flat.shape[0] + 31) // 32) * n_tiles
            bm16_waves = (bm16_ctas + sm_count - 1) // sm_count
            bm32_waves = (bm32_ctas + sm_count - 1) // sm_count
            extended_wide_prefill = sm_count == 124 and bm16_waves == 2 * bm32_waves and bm32_waves == 1
        return (
            self.paroquant_triton_megakernel_enabled
            and not self.training
            and not torch.is_grad_enabled()
            and x_flat.device.type == "cuda"
            and torch.cuda.get_device_capability(x_flat.device) == (8, 0)
            and x_flat.dtype in (torch.float16, torch.bfloat16)
            and x_flat.is_contiguous()
            and self.group_size == 128
            and self.krot in (1, 8)
            and self.in_features % 128 == 0
            and self.in_features <= self.paroquant_triton_megakernel_max_k
            and (self.in_features <= 2048 or large_k_decode_splitk)
            and self.out_features % 8 == 0
            and (is_decode or self.out_features <= self.paroquant_triton_megakernel_prefill_max_n)
            and (is_decode or self.out_features <= 2048 or extended_wide_prefill)
        )

    @staticmethod
    def _sync_benchmark_device(device: torch.device) -> None:
        """Synchronize CUDA timing measurements used by autotune."""
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def _classify_forward_kind(self, x: torch.Tensor, x_flat: torch.Tensor) -> str:
        """Classify the workload as decode or prefill for plan selection."""
        if x.dim() >= 3 and x.shape[-2] == 1 and x_flat.shape[0] <= self.paroquant_triton_decode_max_rows:
            return "decode"
        if x_flat.shape[0] <= self.paroquant_triton_decode_max_rows:
            return "decode"
        if self._large_k_decode_splitk_ready(x_flat):
            return "decode"
        return "prefill"

    def _forward_triton_dense(self, rotated: torch.Tensor) -> torch.Tensor:
        """Dense fallback used when fused Triton plans are unavailable or slower."""
        weight = paroquant_dequantize_triton(self.qweight, self.scales, self.qzeros)
        if weight.dtype != rotated.dtype or weight.device != rotated.device:
            weight = weight.to(device=rotated.device, dtype=rotated.dtype)

        out = torch.matmul(rotated, weight)
        if self.bias is not None:
            out = out + self.bias.to(device=rotated.device, dtype=rotated.dtype)
        return out

    def _forward_triton_decode(self, rotated: torch.Tensor) -> torch.Tensor:
        """Run the fused Triton kernel optimized for small-row decode workloads."""
        out = paroquant_gemm_triton_decode(rotated, self.qweight, self.scales, self.qzeros)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_triton_prefill(self, rotated: torch.Tensor) -> torch.Tensor:
        """Run the fused Triton kernel optimized for larger prefill batches."""
        out = paroquant_gemm_triton_prefill(rotated, self.qweight, self.scales, self.qzeros)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_triton_megakernel(
        self,
        kind: str,
        x_flat: torch.Tensor,
        *,
        input_rows: int | None = None,
        output_shape: tuple[int, ...] | None = None,
        compiled_only: bool = False,
    ) -> torch.Tensor | None:
        """Run one launch that rotates, dequantizes, multiplies, and adds bias."""
        rows = int(x_flat.shape[0]) if input_rows is None else input_rows
        if kind == "decode":
            splitk_scratch = self._megakernel_decode_splitk_scratch(
                x_flat,
                rows=input_rows,
                validated=compiled_only,
            )
            compiled_launch_enabled = self.paroquant_triton_megakernel_decode_compiled_launch_enabled
        else:
            splitk_scratch = self._megakernel_prefill_splitk_scratch(x_flat)
            compiled_launch_enabled = self.paroquant_triton_megakernel_prefill_compiled_launch_enabled
        metadata = self._megakernel_rotation_metadata(
            x_flat,
            decode=kind == "decode" or splitk_scratch is not None,
        )
        if metadata is None:
            raise RuntimeError("ParoQuant megakernel rotation metadata is unavailable during CUDA graph capture.")
        partner, cos, sin, channel_scales = metadata
        buffers = self._buffers
        try:
            qweight = buffers["qweight"]
            scales = buffers["scales"]
            qzeros = buffers["qzeros"]
        except KeyError:
            qweight = self.qweight
            scales = self.scales
            qzeros = self.qzeros
        bias = buffers["bias"] if "bias" in buffers else self.bias
        if (
            scales.device != x_flat.device
            or scales.dtype != x_flat.dtype
            or not scales.is_contiguous()
            or (
                bias is not None
                and (bias.device != x_flat.device or bias.dtype != x_flat.dtype or not bias.is_contiguous())
            )
        ):
            self._ensure_runtime_dtype(device=x_flat.device, dtype=x_flat.dtype)
            scales = buffers["scales"]
            bias = buffers["bias"] if "bias" in buffers else self.bias
        if splitk_scratch is not None:
            partials, counters, stream = splitk_scratch
            split_k = (
                self._megakernel_decode_splitk_factor(x_flat, rows=rows)
                if kind == "decode"
                else 16
            )
            if split_k is None:
                raise RuntimeError("ParoQuant split-K scratch has no measured launch factor.")
            compiled_key = (
                x_flat.device,
                rows,
                self.out_features,
                x_flat.dtype,
            )
            compiled_kernel = self._megakernel_splitk_compiled.get(compiled_key)
            if compiled_launch_enabled and compiled_kernel is not None and compiled_kernel is not False:
                try:
                    prepared_state = (
                        self._cache_megakernel_splitk_prepared(
                            x_flat,
                            rows=rows,
                            stream=stream,
                            split_k=split_k,
                            compiled_kernel=compiled_kernel,
                            qweight=qweight,
                            scales=scales,
                            qzeros=qzeros,
                            partner=partner,
                            cos=cos,
                            sin=sin,
                            channel_scales=channel_scales,
                            bias=bias,
                            partials=partials,
                            counters=counters,
                        )
                        if kind == "decode"
                        else None
                    )
                    return _paroquant_rotation_gemm_splitk_triton_compiled(
                        compiled_kernel,
                        x_flat,
                        qweight,
                        scales,
                        qzeros,
                        partner,
                        cos,
                        sin,
                        channel_scales,
                        bias,
                        partials,
                        counters,
                        stream=stream,
                        split_k=split_k,
                        input_rows=input_rows,
                        result_shape=output_shape,
                        prepared_launch=(
                            None if prepared_state is None else prepared_state.prepared_launch
                        ),
                    )
                except (AttributeError, TypeError):
                    self._megakernel_splitk_compiled[compiled_key] = False
                    compiled_kernel = False
            if compiled_only:
                return None
            if compiled_launch_enabled and compiled_kernel is not False:
                result, compiled_kernel = _paroquant_rotation_gemm_splitk_triton_prepare(
                    x_flat,
                    qweight,
                    scales,
                    qzeros,
                    partner,
                    cos,
                    sin,
                    channel_scales,
                    bias,
                    partials,
                    counters,
                    split_k=split_k,
                    result_shape=output_shape,
                )
                self._megakernel_splitk_compiled[compiled_key] = (
                    False if compiled_kernel is None else compiled_kernel
                )
                if compiled_kernel is not None and kind == "decode":
                    self._cache_megakernel_splitk_prepared(
                        x_flat,
                        rows=rows,
                        stream=stream,
                        split_k=split_k,
                        compiled_kernel=compiled_kernel,
                        qweight=qweight,
                        scales=scales,
                        qzeros=qzeros,
                        partner=partner,
                        cos=cos,
                        sin=sin,
                        channel_scales=channel_scales,
                        bias=bias,
                        partials=partials,
                        counters=counters,
                    )
                return result
            return _paroquant_rotation_gemm_splitk_triton_unchecked(
                x_flat,
                qweight,
                scales,
                qzeros,
                partner,
                cos,
                sin,
                channel_scales,
                bias,
                partials,
                counters,
                split_k=split_k,
                result_shape=output_shape,
            )
        if compiled_only:
            return None
        kernel = (
            paroquant_rotation_gemm_triton_decode
            if kind == "decode"
            else paroquant_rotation_gemm_triton_prefill
        )
        result = kernel(
            x_flat,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
        )
        return result if output_shape is None else result.reshape(output_shape)

    @staticmethod
    def _megakernel_fp16_splitk_shape(*, rows: int, out_features: int) -> bool:
        """Return whether an FP16 decode shape has a measured split-K schedule."""
        return 1 <= rows <= 8 and out_features in {512, 2048, 8192}

    @staticmethod
    def _megakernel_fp16_prefill_splitk_shape(*, rows: int, out_features: int) -> bool:
        """Return whether an FP16 prefill shape has a measured split-K schedule."""
        return _paroquant_splitk_fp16_prefill_shape(rows=rows, out_features=out_features)

    @staticmethod
    def _megakernel_large_k_decode_splitk_factor(
        input_dtype: torch.dtype,
        *,
        rows: int,
        in_features: int,
        out_features: int,
    ) -> int | None:
        """Return a measured BF16 Qwen-width decode split factor."""
        if input_dtype != torch.bfloat16 or rows not in {1, 2, 4, 8, 16, 32}:
            return None
        if in_features == 4096 and out_features in {1024, 4096, 12288}:
            return 32
        if in_features == 12288 and out_features == 4096:
            return 96 if rows == 1 else 32
        return None

    def _megakernel_decode_splitk_factor(
        self,
        x_flat: torch.Tensor,
        *,
        rows: int,
    ) -> int | None:
        """Return the split factor for one measured decode shape."""
        if self.in_features == 2048 and self.out_features in {512, 2048, 8192}:
            if x_flat.dtype == torch.bfloat16 and 1 <= rows <= 8:
                return 16
            if x_flat.dtype == torch.float16 and self._megakernel_fp16_splitk_shape(
                rows=rows,
                out_features=self.out_features,
            ):
                return 16
        return self._megakernel_large_k_decode_splitk_factor(
            x_flat.dtype,
            rows=rows,
            in_features=self.in_features,
            out_features=self.out_features,
        )

    def _megakernel_splitk_scratch_for_shape(
        self,
        x_flat: torch.Tensor,
        *,
        rows: int,
        split_k: int,
        compiled_launch_enabled: bool,
        validated: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Return stream-owned or capture-owned compact scratch for one measured split-K shape."""
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            compiled_key = (
                x_flat.device,
                rows,
                self.out_features,
                x_flat.dtype,
            )
            compiled_kernel = self._megakernel_splitk_compiled.get(compiled_key)
            if (
                not compiled_launch_enabled
                or compiled_kernel is None
                or compiled_kernel is False
                or not _paroquant_splitk_compiled_launch_supported(compiled_kernel)
            ):
                return None
        stream_id = _paroquant_triton_current_stream(x_flat.device)
        block_m, _ = _paroquant_splitk_launch_config(
            x_flat.dtype,
            rows=rows,
            in_features=self.in_features,
            out_features=self.out_features,
            split_k=split_k,
        )
        block_n, _, _, _, _, _ = _paroquant_splitk_output_config(
            x_flat.dtype,
            rows=rows,
            in_features=self.in_features,
            out_features=self.out_features,
            split_k=split_k,
        )
        num_pid_m = (rows + block_m - 1) // block_m
        num_pid_n = (self.out_features + block_n - 1) // block_n
        num_tiles = num_pid_m * num_pid_n
        # Reserve one counter per output tile; paired schedules consume the leading half.
        counter_tiles = num_tiles
        # K=2048 decode scratch remains reusable across FP16/BF16 launch geometries. The large-K route is
        # BF16-only, so retaining its exact measured row tile avoids multiplying per-layer scratch by up to 8x.
        scratch_block_m = max(block_m, 8) if self.in_features == 2048 and rows <= 8 else block_m
        partial_elements = num_tiles * split_k * scratch_block_m * block_n
        if capturing:
            if torch.cuda.get_device_properties(x_flat.device).multi_processor_count != 124:
                return None
            partials = torch.empty(
                partial_elements,
                device=x_flat.device,
                dtype=torch.float32,
            )
            # This reset is captured as a graph node. Each capture receives private allocator-owned scratch.
            counters = torch.zeros(counter_tiles, device=x_flat.device, dtype=torch.int32)
            return partials, counters, stream_id
        if validated and self._megakernel_splitk_last_scratch is not None:
            (
                last_device,
                last_stream,
                last_rows,
                last_out_features,
                last_partials,
                last_counters,
            ) = self._megakernel_splitk_last_scratch
            if (
                last_device == x_flat.device
                and last_stream == stream_id
                and last_rows == rows
                and last_out_features == self.out_features
            ):
                return last_partials, last_counters, stream_id
        cache_key = (
            x_flat.device,
            stream_id,
            rows,
            self.out_features,
        )
        scratch = self._megakernel_splitk_scratch.get(cache_key)
        if scratch is not None:
            if validated:
                self._megakernel_splitk_last_scratch = (*cache_key, *scratch)
            return scratch[0], scratch[1], stream_id
        sm_count = torch.cuda.get_device_properties(x_flat.device).multi_processor_count
        if sm_count != 124:
            return None

        partials = torch.empty(
            partial_elements,
            device=x_flat.device,
            dtype=torch.float32,
        )
        counters = torch.zeros(counter_tiles, device=x_flat.device, dtype=torch.int32)
        scratch = (partials, counters)
        self._megakernel_splitk_scratch[cache_key] = scratch
        if validated:
            self._megakernel_splitk_last_scratch = (*cache_key, *scratch)
        return partials, counters, stream_id

    def _megakernel_decode_splitk_scratch(
        self,
        x_flat: torch.Tensor,
        *,
        rows: int | None = None,
        validated: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Return eager stream-owned or capture-owned scratch for measured FP16/BF16 split-K decode."""
        rows = int(x_flat.shape[0]) if rows is None else rows
        split_k = self._megakernel_decode_splitk_factor(x_flat, rows=rows)
        if split_k is None:
            return None
        measured_large_k = self._large_k_decode_splitk_ready(x_flat, rows=rows)
        if not validated and (
            not self.paroquant_triton_megakernel_decode_splitk_enabled
            or not (
                1 <= rows <= min(self.paroquant_triton_decode_max_rows, 8)
                or measured_large_k
            )
            or self.krot != 8
        ):
            return None
        return self._megakernel_splitk_scratch_for_shape(
            x_flat,
            rows=rows,
            split_k=split_k,
            compiled_launch_enabled=self.paroquant_triton_megakernel_decode_compiled_launch_enabled,
            validated=validated,
        )

    def _megakernel_prefill_splitk_scratch(
        self,
        x_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        """Return scratch only for the measured FP16 K=2048 prefill split-K bands."""
        rows = int(x_flat.shape[0])
        if (
            not self.paroquant_triton_megakernel_prefill_splitk_enabled
            or x_flat.dtype != torch.float16
            or self.in_features != 2048
            or self.krot != 8
            or not self._megakernel_fp16_prefill_splitk_shape(
                rows=rows,
                out_features=self.out_features,
            )
        ):
            return None
        return self._megakernel_splitk_scratch_for_shape(
            x_flat,
            rows=rows,
            split_k=16,
            compiled_launch_enabled=self.paroquant_triton_megakernel_prefill_compiled_launch_enabled,
            validated=False,
        )

    def _cache_megakernel_splitk_prepared(
        self,
        x_flat: torch.Tensor,
        *,
        rows: int,
        stream: int,
        split_k: int,
        compiled_kernel: object,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        partner: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        channel_scales: torch.Tensor,
        bias: torch.Tensor | None,
        partials: torch.Tensor,
        counters: torch.Tensor,
    ) -> _MegakernelSplitKPreparedState | None:
        """Cache one non-capturing launch after all runtime tensors have been validated."""
        if torch.cuda.is_current_stream_capturing():
            return None
        prepared_key = (x_flat.device, stream, rows, x_flat.dtype)
        existing = self._megakernel_splitk_prepared.get(prepared_key)
        if existing is not None and self._megakernel_splitk_prepared_matches(existing):
            return existing
        buffers = self._buffers
        pairs = buffers.get("pairs")
        theta = buffers.get("theta")
        source_channel_scales = buffers.get("channel_scales")
        if (
            pairs is None
            or theta is None
            or source_channel_scales is None
            or buffers.get("qweight") is not qweight
            or buffers.get("scales") is not scales
            or buffers.get("qzeros") is not qzeros
            or buffers.get("bias") is not bias
        ):
            return None
        prepared_launch = _paroquant_prepare_splitk_compiled_launch(
            compiled_kernel,
            qweight,
            scales,
            qzeros,
            partner,
            cos,
            sin,
            channel_scales,
            bias,
            partials,
            counters,
            stream=stream,
            input_dtype=x_flat.dtype,
            rows=rows,
            in_features=self.in_features,
            split_k=split_k,
        )
        plan_cache_key = (
            "decode",
            rows,
            self.in_features,
            self.out_features,
            x_flat.dtype,
            x_flat.device,
        )
        state = _MegakernelSplitKPreparedState(
            compiled_kernel=compiled_kernel,
            prepared_launch=prepared_launch,
            plan_cache_key=plan_cache_key,
            qweight=qweight,
            scales=scales,
            qzeros=qzeros,
            bias=bias,
            pairs=pairs,
            pairs_version=_tensor_runtime_version(pairs),
            theta=theta,
            theta_version=_tensor_runtime_version(theta),
            source_channel_scales=source_channel_scales,
            source_channel_scales_version=_tensor_runtime_version(source_channel_scales),
        )
        self._megakernel_splitk_prepared[prepared_key] = state
        self._megakernel_splitk_last_scratch = (
            x_flat.device,
            stream,
            rows,
            self.out_features,
            partials,
            counters,
        )
        return state

    def _megakernel_splitk_prepared_matches(
        self,
        state: _MegakernelSplitKPreparedState,
    ) -> bool:
        """Check identities and only the versions that feed derived rotation metadata."""
        buffers = self._buffers
        pairs = buffers.get("pairs")
        theta = buffers.get("theta")
        source_channel_scales = buffers.get("channel_scales")
        compiled_key = (
            state.qweight.device,
            state.plan_cache_key[1],
            self.out_features,
            state.plan_cache_key[4],
        )
        return (
            self._plan_cache.get(state.plan_cache_key) == "decode_megakernel"
            and self._megakernel_splitk_compiled.get(compiled_key) is state.compiled_kernel
            and state.qweight is buffers.get("qweight")
            and state.scales is buffers.get("scales")
            and state.qzeros is buffers.get("qzeros")
            and state.bias is buffers.get("bias")
            and state.pairs is pairs
            and state.theta is theta
            and state.source_channel_scales is source_channel_scales
            and (state.pairs_version is None or state.pairs_version == state.pairs._version)
            and (state.theta_version is None or state.theta_version == state.theta._version)
            and (
                state.source_channel_scales_version is None
                or state.source_channel_scales_version == state.source_channel_scales._version
            )
        )

    def _run_plan(self, plan: str, x_flat: torch.Tensor) -> torch.Tensor:
        """Dispatch one named execution plan."""
        if plan == "decode_megakernel":
            return self._forward_triton_megakernel("decode", x_flat)
        if plan == "prefill_megakernel":
            return self._forward_triton_megakernel("prefill", x_flat)
        if plan == "cuda_awq":
            out = self._forward_cuda_awq_fused(x_flat)
            if out is not None:
                return out

        rotated = self._rotate_inputs(x_flat)
        if plan == "cuda_awq":
            out = self._forward_cuda_awq_kernel(rotated)
            if out is None:
                raise RuntimeError("ParoQuant CUDA AWQ fallback is unavailable for this input.")
            return out
        if plan == "dense":
            return self._forward_triton_dense(rotated)
        if plan == "decode_fused":
            return self._forward_triton_decode(rotated)
        if plan == "prefill_fused":
            return self._forward_triton_prefill(rotated)
        raise ValueError(f"Unknown ParoQuant Triton plan: {plan}")

    @staticmethod
    def _legacy_default_plan(kind: str) -> str:
        return "decode_fused" if kind == "decode" else "prefill_fused"

    @staticmethod
    def _megakernel_plan(kind: str) -> str:
        return "decode_megakernel" if kind == "decode" else "prefill_megakernel"

    def _candidate_plans(self, kind: str, x_flat: torch.Tensor) -> list[str]:
        """Return the execution plans worth benchmarking for a workload class."""
        legacy_default = self._legacy_default_plan(kind)
        plans = [self._megakernel_plan(kind)] if self._megakernel_ready(x_flat) else [legacy_default]
        if kind == "decode":
            alternatives = ["cuda_awq", "decode_fused", "dense", "prefill_fused"]
        else:
            alternatives = ["cuda_awq", "prefill_fused", "dense", "decode_fused"]
        return plans + [plan for plan in alternatives if plan not in plans]

    def _benchmark_plan(self, plan: str, kind: str, x_flat: torch.Tensor) -> float:
        """Measure median steady-state module latency for one candidate plan."""
        cache_key = self._plan_cache_key(kind, x_flat)
        missing = object()
        previous_plan = self._plan_cache.get(cache_key, missing)
        self._plan_cache[cache_key] = plan

        def run_cached_plan() -> torch.Tensor:
            selected_plan = self._select_plan(kind, x_flat)
            if selected_plan != plan:
                raise RuntimeError(f"ParoQuant autotune plan changed from {plan} to {selected_plan} during timing.")
            return self._run_plan(selected_plan, x_flat)

        try:
            with torch.inference_mode():
                for _ in range(self.paroquant_triton_autotune_warmup):
                    run_cached_plan()
                self._sync_benchmark_device(x_flat.device)

                starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.paroquant_triton_autotune_iters)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.paroquant_triton_autotune_iters)]
                for index in range(self.paroquant_triton_autotune_iters):
                    starts[index].record()
                    run_cached_plan()
                    ends[index].record()
                ends[-1].synchronize()
                samples = [
                    starts[index].elapsed_time(ends[index]) for index in range(self.paroquant_triton_autotune_iters)
                ]
        finally:
            if previous_plan is missing:
                self._plan_cache.pop(cache_key, None)
            else:
                self._plan_cache[cache_key] = previous_plan

        samples.sort()
        midpoint = len(samples) // 2
        if len(samples) % 2:
            return samples[midpoint]
        return (samples[midpoint - 1] + samples[midpoint]) / 2

    def _plan_cache_key(self, kind: str, x_flat: torch.Tensor) -> tuple[object, ...]:
        return (
            kind,
            int(x_flat.shape[0]),
            int(x_flat.shape[1]),
            self.out_features,
            x_flat.dtype,
            x_flat.device,
        )

    def _select_plan(self, kind: str, x_flat: torch.Tensor) -> str:
        """Pick the best plan, bounded by a bias toward the default fused path."""
        legacy_default = self._legacy_default_plan(kind)
        if self.training or torch.is_grad_enabled():
            return legacy_default

        cache_key = self._plan_cache_key(kind, x_flat)
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            return legacy_default

        candidates = self._candidate_plans(kind, x_flat)
        default_plan = candidates[0]
        if not self.paroquant_triton_autotune_enabled:
            self._plan_cache[cache_key] = default_plan
            return default_plan

        timings = {}
        for plan in candidates:
            try:
                timings[plan] = self._benchmark_plan(plan, kind, x_flat)
            except Exception:
                continue

        if timings:
            best_plan = min(timings, key=timings.get)
            best_time = timings[best_plan]
            retained_default = default_plan if default_plan in timings else best_plan
            default_time = timings[retained_default]

            if best_plan != retained_default and best_time > default_time * (
                1.0 - self.paroquant_triton_autotune_margin
            ):
                best_plan = retained_default
        else:
            best_plan = legacy_default

        self._plan_cache[cache_key] = best_plan
        if best_plan not in {"decode_megakernel", "prefill_megakernel"}:
            self._clear_megakernel_rotation_cache()
        return best_plan

    def _fallback_megakernel(
        self,
        kind: str,
        x_flat: torch.Tensor,
        cache_key: tuple[object, ...],
    ) -> torch.Tensor:
        """Demote a failed mega-kernel plan through the established fallbacks."""
        fallback_plan = "cuda_awq"
        try:
            out = self._run_plan(fallback_plan, x_flat)
        except Exception:
            fallback_plan = self._legacy_default_plan(kind)
            out = self._run_plan(fallback_plan, x_flat)
        self._plan_cache[cache_key] = fallback_plan
        return out

    def forward(self, x: torch.Tensor):
        """Rotate inputs, pick a Triton plan, and preserve adapter semantics."""
        original_shape = x.shape[:-1] + (self.out_features,)
        # A warm compiled split-K launch consumes only the activation pointer and explicit M/K. Passing a
        # contiguous caller tensor directly avoids constructing the otherwise-identical two-dimensional view.
        rows = x.numel() // self.in_features if x.dim() >= 1 and x.shape[-1] == self.in_features else 0
        decode_cache_key = ("decode", rows, self.in_features, self.out_features, x.dtype, x.device)
        cached_decode_plan = self._plan_cache.get(decode_cache_key)
        if cached_decode_plan == "cuda_awq" and self.adapter is None:
            out = self._forward_cuda_awq_fused(x)
            if out is not None:
                return out.reshape(original_shape)
        if (
            self._megakernel_splitk_prepared
            and self.adapter is None
            and not self.training
            and not torch.is_grad_enabled()
            and self.paroquant_triton_megakernel_decode_splitk_enabled
            and self.paroquant_triton_megakernel_decode_compiled_launch_enabled
            and "_forward_triton_megakernel" not in self.__dict__
            and type(self)._forward_triton_megakernel is ParoQuantTritonLinear._forward_triton_megakernel
            and rows
            and x.device.type == "cuda"
            and x.dtype in {torch.float16, torch.bfloat16}
            and x.is_contiguous()
            and self.krot == 8
            and not torch.cuda.is_current_stream_capturing()
        ):
            stream = _paroquant_triton_current_stream(x.device)
            prepared_key = (x.device, stream, rows, x.dtype)
            prepared_state = self._megakernel_splitk_prepared.get(prepared_key)
            if prepared_state is not None:
                if self._megakernel_splitk_prepared_matches(prepared_state):
                    try:
                        return _paroquant_rotation_gemm_splitk_triton_prepared(
                            prepared_state.prepared_launch,
                            x,
                            prepared_state.qweight,
                            result_shape=original_shape,
                        )
                    except Exception:
                        x_flat = x.reshape(-1, x.shape[-1])
                        out = self._fallback_megakernel(
                            "decode",
                            x_flat,
                            prepared_state.plan_cache_key,
                        )
                        return out.reshape(original_shape)
                self._megakernel_splitk_prepared.pop(prepared_key, None)

        split_k = (
            self._megakernel_decode_splitk_factor(x, rows=rows)
            if rows and (cached_decode_plan is None or cached_decode_plan == "decode_megakernel")
            else None
        )
        if (
            self.adapter is None
            and not self.training
            and not torch.is_grad_enabled()
            and self.paroquant_triton_megakernel_decode_splitk_enabled
            and self.paroquant_triton_megakernel_decode_compiled_launch_enabled
            and x.dim() >= 1
            and x.shape[-1] == self.in_features
            and x.device.type == "cuda"
            and x.dtype in {torch.float16, torch.bfloat16}
            and x.is_contiguous()
            and self.krot == 8
            and split_k is not None
        ):
            cache_key = decode_cache_key
            compiled_key = (x.device, rows, self.out_features, x.dtype)
            compiled_kernel = self._megakernel_splitk_compiled.get(compiled_key)
            if (
                self._plan_cache.get(cache_key) == "decode_megakernel"
                and compiled_kernel is not None
                and compiled_kernel is not False
                and (
                    1 <= rows <= min(self.paroquant_triton_decode_max_rows, 8)
                    or self._large_k_decode_splitk_ready(x, rows=rows)
                )
            ):
                direct_output_shape = original_shape if self.out_features in {512, 2048} else None
                try:
                    direct_out = self._forward_triton_megakernel(
                        "decode",
                        x,
                        input_rows=rows,
                        output_shape=direct_output_shape,
                        compiled_only=True,
                    )
                except Exception:
                    x_flat = x.reshape(-1, x.shape[-1])
                    out = self._fallback_megakernel("decode", x_flat, cache_key)
                    return out.reshape(original_shape)
                if direct_out is not None:
                    return direct_out if direct_output_shape is not None else direct_out.reshape(original_shape)

        x_flat = x.reshape(-1, x.shape[-1])

        prefill_cache_key = ("prefill", rows, self.in_features, self.out_features, x_flat.dtype, x_flat.device)
        if decode_cache_key in self._plan_cache:
            kind = "decode"
            cache_key = decode_cache_key
        elif prefill_cache_key in self._plan_cache:
            kind = "prefill"
            cache_key = prefill_cache_key
        else:
            kind = self._classify_forward_kind(x, x_flat)
            cache_key = self._plan_cache_key(kind, x_flat)
        plan = None if self.training or torch.is_grad_enabled() else self._plan_cache.get(cache_key)
        if plan is None:
            plan = self._select_plan(kind, x_flat)
        output_is_shaped = False
        # Avoid the final view only for the measured split-K gates where direct shaped allocation wins.
        megakernel_output_shape = (
            original_shape
            if (
                self.adapter is None
                and kind == "decode"
                and x_flat.dtype == torch.bfloat16
                and self.out_features in {512, 2048}
            )
            else None
        )
        try:
            if plan == "decode_megakernel":
                out = self._forward_triton_megakernel(
                    "decode",
                    x_flat,
                    output_shape=megakernel_output_shape,
                )
                output_is_shaped = megakernel_output_shape is not None
            elif plan == "prefill_megakernel":
                out = self._forward_triton_megakernel(
                    "prefill",
                    x_flat,
                    output_shape=megakernel_output_shape,
                )
                output_is_shaped = megakernel_output_shape is not None
            else:
                out = self._run_plan(plan, x_flat)
        except Exception:
            if plan not in {"decode_megakernel", "prefill_megakernel"}:
                raise
            out = self._fallback_megakernel(kind, x_flat, cache_key)
            output_is_shaped = False

        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)

        return out if output_is_shaped else out.reshape(original_shape)


__all__ = ["ParoQuantTritonLinear"]
