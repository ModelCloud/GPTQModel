# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch

from ...adapter.adapter import Adapter
from ...models._const import DEVICE
from ...quantization import FORMAT, METHOD
from ...quantization.awq.utils.packing_utils import dequantize_gemm, reverse_awq_order, unpack_awq
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from .torch import TorchLinear, _right_shift_unpack
from .torch_awq import AwqTorchLinear


_KOMODO_CACHE_ENV = "GPTQMODEL_KOMODO_CACHE_WEIGHTS"
# Dense dequantized weight caching is opt-in only. The Marlin-like path should
# keep weights quantized/prepacked, not persist a dense dequantized copy.
# Native int4 is the default Komodo path. Set GPTQMODEL_KOMODO_NATIVE_INT4=0
# to force the exact torch-style fallback when investigating numerical drift.
_KOMODO_NATIVE_INT4_ENV = "GPTQMODEL_KOMODO_NATIVE_INT4"
_KOMODO_DROP_SOURCE_ENV = "GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS"
_KOMODO_EAGER_PREPACK_ENV = "GPTQMODEL_KOMODO_EAGER_PREPACK"
_KOMODO_PREPACK_TILE_N_ENV = "GPTQMODEL_KOMODO_PREPACK_TILE_N"
_NativePlan = tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None]


def _native_int4_enabled() -> bool:
    return env_flag(_KOMODO_NATIVE_INT4_ENV, default=True)


def _drop_source_weights_enabled() -> bool:
    return env_flag(_KOMODO_DROP_SOURCE_ENV, default=True)


def _eager_native_prepack_enabled() -> bool:
    return env_flag(_KOMODO_EAGER_PREPACK_ENV, default=True)


def _native_int4_group_size(group_size: int, in_features: int) -> int | None:
    if group_size >= in_features:
        return 0
    if group_size < 32 or group_size % 32 != 0:
        return None
    return group_size


def _native_prepack_tile_n(out_features: int, pack_factor: int) -> int:
    if out_features % pack_factor != 0:
        raise RuntimeError(
            f"Komodo native int4 requires out_features to be divisible by {pack_factor}; got {out_features}."
        )

    raw = os.getenv(_KOMODO_PREPACK_TILE_N_ENV)
    if raw is None:
        tile_n = 1024
    else:
        try:
            tile_n = int(raw)
        except ValueError as err:
            raise RuntimeError(f"{_KOMODO_PREPACK_TILE_N_ENV} must be an integer; got `{raw}`.") from err

    if tile_n <= 0 or tile_n >= out_features:
        return out_features

    tile_n = (tile_n // pack_factor) * pack_factor
    return max(pack_factor, tile_n)


def _packed_weight_empty_like_tile(tile: torch.Tensor, out_features: int, pack_factor: int) -> torch.Tensor:
    return tile.new_empty((tile.shape[0], out_features // pack_factor))


def _npu_int4_ops_available() -> bool:
    try:
        npu_ops = torch.ops.npu
        return hasattr(npu_ops, "npu_convert_weight_to_int4pack") and hasattr(
            npu_ops, "npu_weight_quant_batchmatmul"
        )
    except (AttributeError, RuntimeError):
        return False


def _npu_stream_key(device: torch.device) -> int:
    device = torch.device(device)
    if device.index is not None:
        return device.index
    return torch.npu.current_device()


class _KomodoNativePlanMixin:
    _native_plan_cache: dict
    _native_plan_pending: dict
    _native_prepack_streams: dict
    _native_source_buffer_names: tuple[str, ...] = ("qweight", "qzeros", "scales")

    def _apply(self, fn):
        result = super()._apply(fn)
        if getattr(self, "_native_post_initialized", False) and not getattr(self, "_native_source_dropped", False):
            self.clear_native_cache()
            self._maybe_eager_native_prepack()
        return result

    def _native_key(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
        return torch.device(device), dtype

    def native_plan_prepacked(self, *, device: torch.device | None = None, dtype: torch.dtype = torch.float16) -> bool:
        if device is None:
            device = self.runtime_device()
        if device is None:
            return False

        key = self._native_key(device=torch.device(device), dtype=dtype)
        return key in self._native_plan_cache or key in self._native_plan_pending

    def _native_prepack_stream(self, device: torch.device):
        key = _npu_stream_key(device)
        stream = self._native_prepack_streams.get(key)
        if stream is None:
            stream = torch.npu.Stream(device=torch.device(device))
            self._native_prepack_streams[key] = stream
        return stream

    def _clear_pending_native_plans(self) -> None:
        for _, event, _ in self._native_plan_pending.values():
            try:
                event.synchronize()
            except RuntimeError:
                pass
        self._native_plan_pending.clear()

    def clear_native_cache(self):
        if hasattr(self, "_native_plan_pending"):
            self._clear_pending_native_plans()
        if getattr(self, "_native_source_dropped", False):
            return
        if hasattr(self, "_native_g_idx_plan_cache"):
            self._native_g_idx_plan_cache = None
        if hasattr(self, "_native_plan_cache"):
            self._native_plan_cache.clear()
        if hasattr(self, "_native_prepack_streams"):
            self._native_prepack_streams.clear()

    def enable_source_weight_drop(self, enabled: bool = True):
        if not enabled and getattr(self, "_native_source_dropped", False):
            raise RuntimeError("Komodo source weights have already been dropped and cannot be restored.")
        self._drop_source_weights_after_native_pack = enabled
        return self

    def _native_source_available(self) -> bool:
        for name in self._native_source_buffer_names:
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                continue
            return False
        return True

    def _maybe_drop_native_source_weights(self, *, force: bool = False) -> None:
        if not getattr(self, "_drop_source_weights_after_native_pack", False):
            return
        if getattr(self, "_native_source_dropped", False):
            return
        if (self.training and not force) or not self._native_plan_cache:
            return

        for device, _ in tuple(self._native_plan_cache):
            if device.type == "npu":
                torch.npu.synchronize(device)

        for name in self._native_source_buffer_names:
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                setattr(self, name, tensor.detach().new_empty((0,)))
        if hasattr(self, "_cached_weights"):
            self._cached_weights.clear()
        self._native_source_dropped = True

    def _maybe_eager_native_prepack(self) -> bool:
        if not _eager_native_prepack_enabled() or not _native_int4_enabled() or not _npu_int4_ops_available():
            return False
        if getattr(self, "_native_source_dropped", False) or not self._native_source_available():
            return False

        device = self.runtime_device()
        if device is None:
            return False
        device = torch.device(device)
        dtype = torch.float16
        if not self._can_prefetch_native_plan(device=device, dtype=dtype, allow_training=True):
            return False

        key = self._native_key(device=device, dtype=dtype)
        if key in self._native_plan_cache or key in self._native_plan_pending:
            return True

        plan = self._build_native_plan(device=device, dtype=dtype)
        self._native_plan_cache[key] = plan
        self._maybe_drop_native_source_weights(force=True)
        return True

    def enable_lookahead(self, enabled: bool = True):
        self._lookahead_enabled = enabled
        if not enabled:
            self._lookahead_next = None
        return self

    def set_lookahead_next(self, module):
        if module is None:
            self._lookahead_next = None
            return self

        if isinstance(module, (list, tuple)):
            targets = tuple(target for target in module if target is not None)
            if not targets:
                self._lookahead_next = None
                return self
            for target in targets:
                if not hasattr(target, "prefetch_native_plan"):
                    raise TypeError("Komodo lookahead targets must support prefetch_native_plan().")
            self._lookahead_next = targets
            return self

        if not hasattr(module, "prefetch_native_plan"):
            raise TypeError("Komodo lookahead target must support prefetch_native_plan().")
        self._lookahead_next = module
        return self

    def _maybe_schedule_lookahead(self, dtype: torch.dtype):
        if not getattr(self, "_lookahead_enabled", False) or self.training:
            return
        next_module = getattr(self, "_lookahead_next", None)
        if next_module is None:
            return

        def schedule(module):
            device = module.runtime_device()
            if device is None:
                return
            if module.native_plan_prepacked(device=device, dtype=dtype):
                return
            module.prefetch_native_plan(device=device, dtype=dtype)

        if isinstance(next_module, tuple):
            for module in next_module:
                schedule(module)
        else:
            schedule(next_module)

    def _consume_pending_native_plan(self, key: tuple[torch.device, torch.dtype]):
        pending = self._native_plan_pending.pop(key, None)
        if pending is None:
            return None

        _, event, plan = pending
        if getattr(self, "_drop_source_weights_after_native_pack", False):
            event.synchronize()
        else:
            torch.npu.current_stream(key[0]).wait_event(event)
        self._native_plan_cache[key] = plan
        self._maybe_drop_native_source_weights()
        return plan

    def prefetch_native_plan(self, *, device: torch.device | None = None, dtype: torch.dtype = torch.float16) -> bool:
        """Start NPU int4 prepack on a side stream for first-use latency hiding."""

        if device is None:
            device = self.runtime_device()
        if device is None:
            return False

        device = torch.device(device)
        if not self._can_prefetch_native_plan(device=device, dtype=dtype):
            return False

        key = self._native_key(device=device, dtype=dtype)
        if key in self._native_plan_cache or key in self._native_plan_pending:
            return False
        if getattr(self, "_native_source_dropped", False):
            return False

        stream = self._native_prepack_stream(device)
        with torch.npu.stream(stream):
            plan = self._build_native_plan(device=device, dtype=dtype)
            event = torch.npu.Event()
            event.record(stream)

        self._native_plan_pending[key] = (stream, event, plan)
        return True


def _assert_fp16_inference_input(x: torch.Tensor, module_name: str) -> None:
    if x.dtype != torch.float16:
        raise RuntimeError(
            f"{module_name} currently supports only torch.float16 inference on NPU; got {x.dtype}."
        )


def _weight_quant_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    offsets: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    return torch.ops.npu.npu_weight_quant_batchmatmul(
        x,
        weight,
        scales,
        offsets,
        None,
        None,
        None,
        group_size,
    )


class KomodoLinear(_KomodoNativePlanMixin, TorchLinear):
    """Ascend NPU GPTQ int4 kernel with Marlin-style prepacked steady-state execution."""

    _native_source_buffer_names = ("qweight", "qzeros", "scales", "g_idx", "wf_unsqueeze_zero", "wf_unsqueeze_neg_one")

    SUPPORTS_BACKENDS = [BACKEND.GPTQ_KOMODO]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 35, FORMAT.GPTQ_V2: 35}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = TorchLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = TorchLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = TorchLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = TorchLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = TorchLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = TorchLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = TorchLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = [DEVICE.NPU]
    SUPPORTS_PLATFORM = TorchLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = TorchLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = TorchLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = TorchLinear.REQUIRES_FORMAT_V2

    QUANT_TYPE = "komodo"

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
        adapter: Adapter = None,
        register_buffers: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.GPTQ_KOMODO)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        self.enable_weight_cache(env_flag(_KOMODO_CACHE_ENV, default=False))
        self._native_plan_cache: dict[tuple[torch.device, torch.dtype], _NativePlan] = {}
        self._native_plan_pending: dict[
            tuple[torch.device, torch.dtype], tuple[torch.npu.Stream, torch.npu.Event, _NativePlan]
        ] = {}
        self._native_prepack_streams: dict[int, torch.npu.Stream] = {}
        self._native_g_idx_plan_cache: tuple[bool, torch.Tensor | None] | None = None
        self._drop_source_weights_after_native_pack = _drop_source_weights_enabled()
        self._native_source_dropped = False
        self._native_post_initialized = False

    def post_init(self):
        super().post_init()
        self.clear_native_cache()
        self._native_post_initialized = True
        self._maybe_eager_native_prepack()

    def train(self, mode: bool = True):
        if mode and getattr(self, "_native_source_dropped", False):
            raise RuntimeError("KomodoLinear cannot enter training mode after source quant weights are dropped.")
        previous = self.training
        result = super().train(mode=mode)
        if previous != mode and mode:
            self.clear_native_cache()
        return result

    def clear_weight_cache(self):
        super().clear_weight_cache()
        self.clear_native_cache()

    def _native_g_idx_plan(self) -> tuple[bool, torch.Tensor | None]:
        cached = self._native_g_idx_plan_cache
        if cached is not None:
            return cached

        native_group_size = _native_int4_group_size(self.group_size, self.in_features)
        if native_group_size is None:
            result = (False, None)
            self._native_g_idx_plan_cache = result
            return result

        g_idx = self.g_idx.detach().to(device="cpu", dtype=torch.int64)
        if native_group_size == 0:
            expected = torch.zeros(self.in_features, dtype=torch.int64)
        else:
            expected = torch.arange(self.in_features, dtype=torch.int64) // native_group_size

        if torch.equal(g_idx, expected):
            result = (True, None)
            self._native_g_idx_plan_cache = result
            return result

        perm = torch.argsort(g_idx, stable=True)
        if torch.equal(g_idx[perm], expected):
            result = (True, perm.to(dtype=torch.int64))
        else:
            result = (False, None)
        self._native_g_idx_plan_cache = result
        return result

    def _can_use_native_int4(self, x: torch.Tensor, compute_dtype: torch.dtype) -> bool:
        if not _native_int4_enabled() or not _npu_int4_ops_available():
            return False
        if self.training or x.requires_grad or x.device.type != "npu":
            return False
        if compute_dtype != torch.float16:
            return False
        if self.bits != 4 or x.shape[-1] != self.in_features:
            return False
        key = self._native_key(device=x.device, dtype=compute_dtype)
        if key in self._native_plan_cache:
            return True
        if getattr(self, "_native_source_dropped", False):
            return False
        if not self._native_source_available():
            return False
        if self.qweight.device != x.device or self.qzeros.device != x.device or self.scales.device != x.device:
            return False
        supported, _ = self._native_g_idx_plan()
        return supported

    def _can_prefetch_native_plan(
        self, *, device: torch.device, dtype: torch.dtype, allow_training: bool = False
    ) -> bool:
        if not _native_int4_enabled() or not _npu_int4_ops_available():
            return False
        if (self.training and not allow_training) or dtype != torch.float16 or device.type != "npu":
            return False
        if self.bits != 4:
            return False
        key = self._native_key(device=device, dtype=dtype)
        if key in self._native_plan_cache:
            return True
        if getattr(self, "_native_source_dropped", False):
            return False
        if not self._native_source_available():
            return False
        if self.qweight.device != device or self.qzeros.device != device or self.scales.device != device:
            return False
        supported, _ = self._native_g_idx_plan()
        return supported

    def _build_native_plan(self, *, device: torch.device, dtype: torch.dtype) -> _NativePlan:
        supported, input_perm_cpu = self._native_g_idx_plan()
        if not supported:
            raise RuntimeError("Komodo native int4 plan requested for an unsupported GPTQ g_idx layout.")

        input_perm = None
        if input_perm_cpu is not None:
            input_perm = input_perm_cpu.to(device=device, non_blocking=self.g_idx.device.type == "cpu")

        tile_n = _native_prepack_tile_n(self.out_features, self.pack_factor)
        packed_weight = None
        for start in range(0, self.out_features, tile_n):
            width = min(tile_n, self.out_features - start)
            qweight_tile = self.qweight.narrow(1, start, width)
            weight = torch.bitwise_and(
                _right_shift_unpack(
                    qweight_tile.unsqueeze(1).expand(-1, self.pack_factor, -1),
                    self.wf_unsqueeze_neg_one,
                    self.dequant_dtype,
                ),
                self.maxq,
            )
            weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2]).to(torch.int32)
            if input_perm is not None:
                weight = weight.index_select(0, input_perm)

            signed_weight = (weight - 8).contiguous()
            packed_tile = torch.ops.npu.npu_convert_weight_to_int4pack(signed_weight)
            if packed_weight is None:
                packed_weight = _packed_weight_empty_like_tile(packed_tile, self.out_features, self.pack_factor)
            packed_start = start // self.pack_factor
            packed_width = width // self.pack_factor
            packed_weight.narrow(1, packed_start, packed_width).copy_(packed_tile)
            del weight, signed_weight, packed_tile

        if packed_weight is None:
            raise RuntimeError("Komodo native int4 plan requested for an empty weight.")

        zeros = self._stream_decode_qzeros().to(device=device)
        scales = self.scales.to(device=device, dtype=dtype).contiguous()
        offsets = (8 - zeros.to(torch.int32)).to(device=device, dtype=dtype).contiguous()
        native_group_size = _native_int4_group_size(self.group_size, self.in_features)
        if native_group_size is None:
            raise RuntimeError("Komodo native int4 plan requested for an unsupported group size.")

        return packed_weight, scales, offsets, native_group_size, input_perm

    def _native_plan(self, *, device: torch.device, dtype: torch.dtype) -> _NativePlan:
        key = self._native_key(device=device, dtype=dtype)
        cached = self._native_plan_cache.get(key)
        if cached is not None:
            return cached

        pending = self._consume_pending_native_plan(key)
        if pending is not None:
            return pending

        if getattr(self, "_native_source_dropped", False):
            raise RuntimeError("Komodo native source weights were dropped before a native plan was available.")
        plan = self._build_native_plan(device=device, dtype=dtype)
        self._native_plan_cache[key] = plan
        self._maybe_drop_native_source_weights()
        return plan

    def _native_forward(self, x: torch.Tensor):
        _assert_fp16_inference_input(x, self.__class__.__name__)
        input_dtype = x.dtype
        compute_dtype = torch.float16
        out_shape = x.shape[:-1] + (self.out_features,)
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or not x_flat.is_contiguous():
            x_flat = x_flat.to(dtype=compute_dtype).contiguous()

        packed_weight, scales, offsets, native_group_size, input_perm = self._native_plan(
            device=x_flat.device, dtype=compute_dtype
        )
        if input_perm is not None:
            x_flat = x_flat.index_select(1, input_perm)
        out = _weight_quant_matmul(x_flat, packed_weight, scales, offsets, native_group_size)
        out = out.reshape(out_shape)

        if self.bias is not None:
            bias = self.bias
            if bias.device != out.device or bias.dtype != out.dtype:
                bias = bias.to(device=out.device, dtype=out.dtype)
            out.add_(bias)
        if self.adapter:
            out = self.adapter.apply(x=x_flat, out=out)
        if input_dtype == torch.float32:
            out = out.to(torch.float32)
        self._maybe_schedule_lookahead(compute_dtype)
        return out

    def forward(self, x: torch.Tensor):
        _assert_fp16_inference_input(x, self.__class__.__name__)
        compute_dtype = torch.float16
        if self._can_use_native_int4(x, compute_dtype):
            return self._native_forward(x)
        return super().forward(x)


class AwqKomodoLinear(_KomodoNativePlanMixin, AwqTorchLinear):
    """Ascend NPU AWQ int4 kernel with optional dense fallback caching."""

    _native_source_buffer_names = ("qweight", "qzeros", "scales")

    SUPPORTS_BACKENDS = [BACKEND.AWQ_KOMODO]
    SUPPORTS_METHODS = [METHOD.AWQ]
    SUPPORTS_FORMATS = {FORMAT.GEMM: 35}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = AwqTorchLinear.SUPPORTS_DESC_ACT
    SUPPORTS_SYM = AwqTorchLinear.SUPPORTS_SYM
    SUPPORTS_SHARDS = AwqTorchLinear.SUPPORTS_SHARDS
    SUPPORTS_TRAINING = AwqTorchLinear.SUPPORTS_TRAINING
    SUPPORTS_AUTO_PADDING = AwqTorchLinear.SUPPORTS_AUTO_PADDING
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = AwqTorchLinear.SUPPORTS_IN_FEATURES_DIVISIBLE_BY
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = AwqTorchLinear.SUPPORTS_OUT_FEATURES_DIVISIBLE_BY
    SUPPORTS_DEVICES = [DEVICE.NPU]
    SUPPORTS_PLATFORM = AwqTorchLinear.SUPPORTS_PLATFORM
    SUPPORTS_PACK_DTYPES = AwqTorchLinear.SUPPORTS_PACK_DTYPES
    SUPPORTS_ADAPTERS = AwqTorchLinear.SUPPORTS_ADAPTERS
    SUPPORTS_DTYPES = [torch.float16]
    REQUIRES_FORMAT_V2 = AwqTorchLinear.REQUIRES_FORMAT_V2

    QUANT_TYPE = "komodo_awq"

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
        adapter: Adapter = None,
        register_buffers: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("backend", BACKEND.AWQ_KOMODO)
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs,
        )
        self._cache_enabled = env_flag(_KOMODO_CACHE_ENV, default=False)
        self._cached_weights: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._native_plan_cache: dict[tuple[torch.device, torch.dtype], _NativePlan] = {}
        self._native_plan_pending: dict[
            tuple[torch.device, torch.dtype], tuple[torch.npu.Stream, torch.npu.Event, _NativePlan]
        ] = {}
        self._native_prepack_streams: dict[int, torch.npu.Stream] = {}
        self._lookahead_enabled = env_flag("GPTQ_TORCH_LOOKAHEAD", default=False)
        self._lookahead_next = None
        self._drop_source_weights_after_native_pack = _drop_source_weights_enabled()
        self._native_source_dropped = False
        self._native_post_initialized = False

    def post_init(self):
        super().post_init()
        self.clear_weight_cache()
        self._native_post_initialized = True
        self._maybe_eager_native_prepack()

    def train(self, mode: bool = True):
        if mode and getattr(self, "_native_source_dropped", False):
            raise RuntimeError("AwqKomodoLinear cannot enter training mode after source quant weights are dropped.")
        previous = self.training
        result = super().train(mode=mode)
        if previous != mode and mode:
            self.clear_weight_cache()
        return result

    def enable_weight_cache(self, enabled: bool = True):
        self._cache_enabled = enabled
        if not enabled:
            self.clear_weight_cache()
        return self

    def clear_weight_cache(self):
        self._cached_weights.clear()
        self.clear_native_cache()

    def _cached_weight_key(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
        return torch.device(device), dtype

    def _dequantized_weight(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = self._cached_weight_key(device=device, dtype=dtype)
        cached = self._cached_weights.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached

        weight = dequantize_gemm(
            qweight=self.qweight,
            qzeros=self.qzeros,
            scales=self.scales,
            bits=self.bits,
            group_size=self.group_size,
        )
        if weight.dtype != dtype or weight.device != device or not weight.is_contiguous():
            weight = weight.to(device=device, dtype=dtype).contiguous()

        if self._cache_enabled and not self.training:
            self._cached_weights[key] = weight.detach()
        return weight

    def _can_use_native_int4(self, x: torch.Tensor, compute_dtype: torch.dtype) -> bool:
        if not _native_int4_enabled() or not _npu_int4_ops_available():
            return False
        if self.training or x.requires_grad or x.device.type != "npu":
            return False
        if compute_dtype != torch.float16:
            return False
        if self.bits != 4 or x.shape[-1] != self.in_features:
            return False
        key = self._native_key(device=x.device, dtype=compute_dtype)
        if key in self._native_plan_cache:
            return True
        if getattr(self, "_native_source_dropped", False):
            return False
        if not self._native_source_available():
            return False
        if self.qweight.device != x.device or self.qzeros.device != x.device or self.scales.device != x.device:
            return False
        return _native_int4_group_size(self.group_size, self.in_features) is not None

    def _can_prefetch_native_plan(
        self, *, device: torch.device, dtype: torch.dtype, allow_training: bool = False
    ) -> bool:
        if not _native_int4_enabled() or not _npu_int4_ops_available():
            return False
        if (self.training and not allow_training) or dtype != torch.float16 or device.type != "npu":
            return False
        if self.bits != 4:
            return False
        key = self._native_key(device=device, dtype=dtype)
        if key in self._native_plan_cache:
            return True
        if getattr(self, "_native_source_dropped", False):
            return False
        if not self._native_source_available():
            return False
        if self.qweight.device != device or self.qzeros.device != device or self.scales.device != device:
            return False
        return _native_int4_group_size(self.group_size, self.in_features) is not None

    def _build_native_plan(self, *, device: torch.device, dtype: torch.dtype) -> _NativePlan:
        max_val = (1 << self.bits) - 1

        tile_n = _native_prepack_tile_n(self.out_features, self.pack_factor)
        packed_weight = None
        izeros_tiles = []
        for start in range(0, self.out_features, tile_n):
            width = min(tile_n, self.out_features - start)
            packed_start = start // self.pack_factor
            packed_width = width // self.pack_factor
            qweight_tile = self.qweight.narrow(1, packed_start, packed_width)
            qzeros_tile = self.qzeros.narrow(1, packed_start, packed_width)
            iweight, izeros = unpack_awq(qweight_tile, qzeros_tile, self.bits)
            iweight, izeros = reverse_awq_order(iweight, izeros, self.bits)
            iweight = torch.bitwise_and(iweight, max_val).to(torch.int32)
            izeros = torch.bitwise_and(izeros, max_val).reshape(self.scales.shape[0], width)

            signed_weight = (iweight - 8).contiguous()
            packed_tile = torch.ops.npu.npu_convert_weight_to_int4pack(signed_weight)
            if packed_weight is None:
                packed_weight = _packed_weight_empty_like_tile(packed_tile, self.out_features, self.pack_factor)
            packed_weight.narrow(1, packed_start, packed_width).copy_(packed_tile)
            izeros_tiles.append(izeros)
            del iweight, signed_weight, packed_tile

        if packed_weight is None:
            raise RuntimeError("Komodo native int4 plan requested for an empty weight.")
        izeros = torch.cat(izeros_tiles, dim=1) if len(izeros_tiles) > 1 else izeros_tiles[0]
        scales = self.scales.to(device=device, dtype=dtype).contiguous()
        offsets = (8 - izeros.to(torch.int32)).to(device=device, dtype=dtype).contiguous()
        native_group_size = _native_int4_group_size(self.group_size, self.in_features)
        if native_group_size is None:
            raise RuntimeError("Komodo native int4 plan requested for an unsupported group size.")

        return packed_weight, scales, offsets, native_group_size, None

    def _native_plan(self, *, device: torch.device, dtype: torch.dtype) -> _NativePlan:
        key = self._native_key(device=device, dtype=dtype)
        cached = self._native_plan_cache.get(key)
        if cached is not None:
            return cached

        pending = self._consume_pending_native_plan(key)
        if pending is not None:
            return pending

        if getattr(self, "_native_source_dropped", False):
            raise RuntimeError("Komodo native source weights were dropped before a native plan was available.")
        plan = self._build_native_plan(device=device, dtype=dtype)
        self._native_plan_cache[key] = plan
        self._maybe_drop_native_source_weights()
        return plan

    def _native_forward(self, x: torch.Tensor):
        _assert_fp16_inference_input(x, self.__class__.__name__)
        input_dtype = x.dtype
        compute_dtype = torch.float16
        original_shape = x.shape[:-1] + (self.out_features,)
        device = x.device
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or x_flat.device != device:
            x_flat = x_flat.to(device=device, dtype=compute_dtype)
        elif not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        packed_weight, scales, offsets, native_group_size, _ = self._native_plan(device=device, dtype=compute_dtype)
        output = _weight_quant_matmul(x_flat, packed_weight, scales, offsets, native_group_size)

        if self.bias is not None:
            bias = self.bias
            if bias.device != output.device or bias.dtype != output.dtype:
                bias = bias.to(device=output.device, dtype=output.dtype)
            output = output + bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        if output.dtype != input_dtype:
            output = output.to(dtype=input_dtype)

        self._maybe_schedule_lookahead(compute_dtype)
        return output.reshape(original_shape)

    def forward(self, x: torch.Tensor):
        _assert_fp16_inference_input(x, self.__class__.__name__)
        input_dtype = x.dtype
        compute_dtype = torch.float16
        if self._can_use_native_int4(x, compute_dtype):
            return self._native_forward(x)

        original_shape = x.shape[:-1] + (self.out_features,)
        device = x.device
        x_flat = x.reshape(-1, x.shape[-1])
        if x_flat.dtype != compute_dtype or x_flat.device != device:
            x_flat = x_flat.to(device=device, dtype=compute_dtype)
        elif not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        self._ensure_runtime_dtype(device=device, dtype=compute_dtype)
        weight = self._dequantized_weight(device=device, dtype=compute_dtype)
        output = torch.matmul(x_flat, weight)

        if self.bias is not None:
            output = output + self.bias

        if self.adapter:
            output = self.adapter.apply(x=x_flat, out=output)

        if output.dtype != input_dtype:
            output = output.to(dtype=input_dtype)

        self._maybe_schedule_lookahead(compute_dtype)
        return output.reshape(original_shape)


__all__ = ["AwqKomodoLinear", "KomodoLinear"]
