# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from vllm at https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/gptq_marlin.py

from collections import Counter
from dataclasses import dataclass
import os
import threading
from typing import List, Optional, Tuple

import numpy as np
import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import FormatSupport, GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.logger import setup_logger
from ...utils.marlin import (
    _marlin_capability_supported,
    _transform_param,
    apply_gptq_marlin_linear,
    gptq_marlin_gemm,
    gptq_marlin_repack,
    marlin_import_exception,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_runtime_available,
    marlin_runtime_error,
    marlin_sort_g_idx,
    replace_parameter,
)
from ...utils.marlin_lora import (
    apply_marlin_fused_lora,
    marlin_lora_cuda_up_add_enabled,
    prepare_marlin_fused_lora,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


# Sample process-level policy once when each MarlinLinear is created. Automatic
# shape selection happens here so route statistics are observable; the native
# dispatcher still validates the quantization contract and launch geometry.
_PACKED_PREFILL_ENV = "GPTQMODEL_MARLIN_PACKED_PREFILL"
_PACKED_PREFILL_MIN_ROWS_ENV = "GPTQMODEL_MARLIN_PACKED_PREFILL_MIN_ROWS"
_LORA_MEGA_KERNEL_WORKSPACE_BLOCKS = 192
_LORA_MEGA_KERNEL_WORKSPACE_BLOCKS_BY_RANK = {192: 224, 256: 256}
_PACKED_PREFILL_CONFIG_ENV = "GPTQMODEL_MARLIN_PACKED_PREFILL_CONFIG"
_PACKED_PREFILL_STATS_ENV = "GPTQMODEL_MARLIN_PACKED_PREFILL_STATS"
_PACKED_PREFILL_MIN_ROWS_DEFAULT = 1024
_PACKED_PREFILL_PROFILED_HARDWARE = (8, 0, 124)


@dataclass(frozen=True, slots=True)
class _PackedPrefillRoute:
    dtype: str
    size_k: int
    size_n: int
    min_m: int
    max_m: int
    config: int


@dataclass(frozen=True, slots=True)
class _PackedPrefillDecision:
    config: int
    reason: str


# Reuse common fast-path decisions instead of allocating per layer.
_PACKED_PREFILL_DISABLED = _PackedPrefillDecision(0, "disabled")
_PACKED_PREFILL_DECODE = _PackedPrefillDecision(0, "decode")


# Offline-tuned on 124-SM sm_80 boards. New promotion requires repeated results
# from multiple physical GPUs: either the one-sided 95% confidence lower bound
# is at least 1.05x ordinary Marlin or every GPU clears a raw 1.07x margin.
# K/N are the unpadded logical projection sizes; padded tails intentionally stay
# on ordinary Marlin in automatic mode.
_PACKED_PREFILL_ROUTES = (
    # Retained promoted Llama-3.2-1B exact points.
    _PackedPrefillRoute("fp16", 2048, 8192, 1024, 1024, 1),
    _PackedPrefillRoute("fp16", 2048, 8192, 2048, 2048, 2),
    _PackedPrefillRoute("bf16", 2048, 8192, 1024, 1024, 1),
    _PackedPrefillRoute("bf16", 2048, 8192, 2048, 2048, 2),
    # Llama 8B and Qwen3 8B.
    _PackedPrefillRoute("fp16", 4096, 4096, 4097, 8192, 2),
    _PackedPrefillRoute("fp16", 4096, 12288, 1025, 8192, 2),
    _PackedPrefillRoute("fp16", 4096, 14336, 1025, 8192, 2),
    _PackedPrefillRoute("fp16", 12288, 4096, 8000, 8192, 2),
    _PackedPrefillRoute("fp16", 14336, 4096, 6144, 8192, 2),
    _PackedPrefillRoute("bf16", 4096, 4096, 2049, 4096, 1),
    _PackedPrefillRoute("bf16", 4096, 4096, 4097, 8192, 2),
    _PackedPrefillRoute("bf16", 4096, 12288, 1025, 8192, 2),
    _PackedPrefillRoute("bf16", 4096, 14336, 1025, 8192, 2),
    _PackedPrefillRoute("bf16", 12288, 4096, 6144, 8192, 2),
    _PackedPrefillRoute("bf16", 14336, 4096, 6144, 8192, 2),
    # Qwen3 32B.
    _PackedPrefillRoute("fp16", 5120, 8192, 2049, 8192, 2),
    _PackedPrefillRoute("fp16", 5120, 25600, 1024, 8192, 2),
    _PackedPrefillRoute("fp16", 8192, 5120, 4096, 8192, 2),
    _PackedPrefillRoute("fp16", 25600, 5120, 4097, 8192, 2),
    _PackedPrefillRoute("bf16", 5120, 8192, 2049, 8192, 2),
    _PackedPrefillRoute("bf16", 5120, 25600, 1024, 8192, 2),
    _PackedPrefillRoute("bf16", 8192, 5120, 2049, 4096, 1),
    _PackedPrefillRoute("bf16", 8192, 5120, 4097, 8192, 2),
    _PackedPrefillRoute("bf16", 25600, 5120, 4097, 8192, 2),
    # Llama 70B.
    _PackedPrefillRoute("fp16", 8192, 8192, 3072, 8192, 2),
    _PackedPrefillRoute("fp16", 28672, 8192, 3072, 8192, 2),
    _PackedPrefillRoute("bf16", 8192, 8192, 2049, 8192, 2),
    _PackedPrefillRoute("bf16", 28672, 8192, 2049, 8192, 2),
)

# Most models repeat a few projection shapes across every layer.
_PACKED_PREFILL_ROUTES_BY_SHAPE = {
    key: tuple(
        route
        for route in _PACKED_PREFILL_ROUTES
        if (route.dtype, route.size_k, route.size_n) == key
    )
    for key in {
        (route.dtype, route.size_k, route.size_n)
        for route in _PACKED_PREFILL_ROUTES
    }
}

_PACKED_PREFILL_ROUTE_STATS: Counter[tuple] = Counter()
_PACKED_PREFILL_ROUTE_STATS_LOCK = threading.Lock()


def _packed_prefill_dtype_name(dtype: torch.dtype) -> str | None:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    return None


def _select_packed_prefill_config(
    *,
    major: int,
    minor: int,
    sms: int,
    dtype: torch.dtype,
    rows: int,
    size_k: int,
    size_n: int,
) -> tuple[int, str]:
    """Return the offline-tuned config and a stable miss reason."""
    dtype_name = _packed_prefill_dtype_name(dtype)
    if dtype_name is None:
        return 0, "dtype_miss"
    if (major, minor, sms) != _PACKED_PREFILL_PROFILED_HARDWARE:
        return 0, "hardware_miss"

    routes = _PACKED_PREFILL_ROUTES_BY_SHAPE.get((dtype_name, size_k, size_n))
    if routes is None:
        return 0, "shape_miss"
    for route in routes:
        if route.min_m <= rows <= route.max_m:
            return route.config, "hit"
    return 0, "m_miss"


def _record_packed_prefill_route(
    *,
    decision: _PackedPrefillDecision,
    hardware: tuple[int, int, int] | None,
    dtype: torch.dtype,
    rows: int,
    size_k: int,
    size_n: int,
) -> None:
    major, minor, sms = hardware or (-1, -1, -1)
    key = (
        decision.reason,
        major,
        minor,
        sms,
        _packed_prefill_dtype_name(dtype) or str(dtype),
        rows,
        size_k,
        size_n,
        decision.config,
    )
    with _PACKED_PREFILL_ROUTE_STATS_LOCK:
        _PACKED_PREFILL_ROUTE_STATS[key] += 1


def reset_marlin_packed_prefill_route_stats() -> None:
    """Reset opt-in process-level packed-prefill route counters."""
    with _PACKED_PREFILL_ROUTE_STATS_LOCK:
        _PACKED_PREFILL_ROUTE_STATS.clear()


def get_marlin_packed_prefill_route_stats(*, reset: bool = False) -> dict:
    """Snapshot opt-in route counters in a JSON-serializable form."""
    with _PACKED_PREFILL_ROUTE_STATS_LOCK:
        snapshot = _PACKED_PREFILL_ROUTE_STATS.copy()
        if reset:
            _PACKED_PREFILL_ROUTE_STATS.clear()

    routes = []
    reasons: Counter[str] = Counter()
    for key, count in sorted(snapshot.items()):
        reason, major, minor, sms, dtype, rows, size_k, size_n, config = key
        reasons[reason] += count
        routes.append(
            {
                "reason": reason,
                "compute_capability": f"{major}.{minor}" if major >= 0 else None,
                "multiprocessor_count": sms if sms >= 0 else None,
                "dtype": dtype,
                "m": rows,
                "k": size_k,
                "n": size_n,
                "config": config,
                "count": count,
            }
        )
    auto_hits = reasons["hit"]
    auto_misses = sum(
        count for reason, count in reasons.items() if reason not in ("hit", "manual_config")
    )
    auto_total = auto_hits + auto_misses
    return {
        "total": sum(snapshot.values()),
        "auto_hits": auto_hits,
        "auto_misses": auto_misses,
        "auto_hit_rate": auto_hits / auto_total if auto_total else 0.0,
        "manual_attempts": reasons["manual_config"],
        "by_reason": dict(sorted(reasons.items())),
        "routes": routes,
    }


def _packed_prefill_enabled() -> bool:
    """Enable conservative automatic packed-prefill routing by default."""
    return env_flag(_PACKED_PREFILL_ENV, default=True)


class MarlinLinear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_MARLIN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMAT_BIT_MAP = {
        FORMAT.GPTQ: FormatSupport(priority=90, bits=(4, 8)),
        FORMAT.GPTQ_V2: FormatSupport(priority=90, bits=(4, 8)),
        FORMAT.MARLIN: FormatSupport(priority=90, bits=(4, 8)),
    }
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    # GPTQ's int32 checkpoint layout stores K/N in 32-value blocks. Marlin
    # consumes 64-column N tiles and complete K groups, so post_init() pads the
    # runtime tensors to those larger shapes and forward() restores the logical
    # output shape.
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "marlin"

    # (num_bits, is_sym) -> quant_type
    TYPE_MAP = {
        (4, True): scalar_types.uint4b8,
        (8, True): scalar_types.uint8b128,
    }

    @staticmethod
    def _ceil_multiple(value: int, divisor: int) -> int:
        return ((value + divisor - 1) // divisor) * divisor

    @classmethod
    def _validate(
        cls,
        bits: int = 4,
        group_size: int = 128,
        desc_act: bool = False,
        sym: bool = False,
        pack_dtype: torch.dtype = None,
        dtype: Optional[torch.dtype] = None,
        dynamic: Optional[dict] = None,
        in_features: int = None,
        out_features: int = None,
        device: Optional[DEVICE] = None,
        trainable: Optional[bool] = None,
        adapter: Optional[Adapter] = None,
        format: Optional[FORMAT] = None,
    ) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            pack_dtype=pack_dtype,
            dtype=dtype,
            dynamic=dynamic,
            in_features=in_features,
            out_features=out_features,
            device=device,
            trainable=trainable,
            adapter=adapter,
            format=format,
        )
        if not ok:
            return ok, err

        needs_k_padding = (
            in_features is not None
            and group_size not in (-1, in_features)
            and in_features % group_size != 0
        )
        needs_n_padding = out_features is not None and out_features % 64 != 0
        if needs_k_padding and desc_act:
            return False, NotImplementedError(
                f"{cls}: automatic K padding is not supported with desc_act=True; "
                f"in_features={in_features}, group_size={group_size}."
            )
        if adapter is not None and (needs_k_padding or needs_n_padding):
            return False, NotImplementedError(
                f"{cls}: automatic K/N padding is not supported with adapters; "
                f"in_features={in_features}, out_features={out_features}, group_size={group_size}."
            )
        return True, None

    def __init__(
            self, bits: int,
            group_size: int,
            desc_act: bool,
            sym: bool,
            in_features: int,
            out_features: int,
            bias: bool = False,
            pack_dtype: torch.dtype = torch.int32,
            register_buffers: bool = False,
            adapter: Adapter = None,
            **kwargs):
        if marlin_import_exception is not None:
            raise ValueError(
                "Trying to use the marlin backend, but the runtime requirements were not met: "
                f"{marlin_import_exception}"
            )

        # self.original_in_features = in_features
        # self.original_out_features = out_features

        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        self.compute_dtype = kwargs.get("dtype") or torch.float16
        self.fp32 = env_flag("GPTQMODEL_MARLIN_USE_FP32", default=True)
        self.packed_prefill = _packed_prefill_enabled()
        self.packed_prefill_stats = env_flag(_PACKED_PREFILL_STATS_ENV, default=False)
        self._packed_prefill_hardware: tuple[int, int, int] | None = None
        try:
            self.packed_prefill_min_rows = int(
                os.environ.get(
                    _PACKED_PREFILL_MIN_ROWS_ENV,
                    str(_PACKED_PREFILL_MIN_ROWS_DEFAULT),
                )
            )
            self.packed_prefill_config = int(
                os.environ.get(_PACKED_PREFILL_CONFIG_ENV, "0")
            )
        except ValueError as exc:
            raise ValueError(
                f"{_PACKED_PREFILL_MIN_ROWS_ENV} and {_PACKED_PREFILL_CONFIG_ENV} must be integers."
            ) from exc
        if self.packed_prefill_min_rows < 1:
            raise ValueError(
                f"{_PACKED_PREFILL_MIN_ROWS_ENV} must be a positive integer."
            )
        if not 0 <= self.packed_prefill_config <= 4:
            raise ValueError(f"{_PACKED_PREFILL_CONFIG_ENV} must be between 0 and 4.")

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.GPTQ_MARLIN),
            adapter=adapter,
            register_buffers=False, # do not register buffers in super()
            **kwargs)

        self.padded_in_features = (
            self.in_features
            if self.requested_group_size == -1
            else self._ceil_multiple(self.in_features, self.group_size)
        )
        self.padded_out_features = self._ceil_multiple(self.out_features, 64)

        if not self.fp32:
            log.warn.once(
                "Kernel: GPTQMODEL_MARLIN_USE_FP32 is disabled. Marlin will use reduced-precision reduction.")
        if self.packed_prefill:
            log.info.once(
                "Kernel: automatic packed Marlin W4A16 prefill routing is enabled; "
                "decode and unsupported shapes use ordinary Marlin, with no "
                "dense-weight cache."
            )

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(desc_act,
                                             self.group_size,
                                             is_row_parallel=False):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_size = self.padded_in_features // self.group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_size = self.padded_in_features // self.group_size

        # Quantized weights
        self.register_parameter(
            "qweight",
            torch.nn.Parameter(
                torch.empty(
                    self.in_features // self.pack_factor,
                    self.out_features,
                    dtype=torch.int32,
                ),
                requires_grad=False
            ),
        )

        # Activation order
        self.register_parameter(
            "g_idx",
            torch.nn.Parameter(data=torch.empty(
                self.in_features,
                dtype=torch.int32,
            ), requires_grad=False),
        )

        # Scales
        self.register_parameter(
            "scales",
            torch.nn.Parameter(
                torch.empty(
                    scales_and_zp_size,
                    self.out_features,
                    dtype=self.compute_dtype,
                ),
                requires_grad=False
            ),
        )

        # Quantized zero-points
        self.register_parameter(
            "qzeros",
            torch.nn.Parameter(
                torch.empty(
                    scales_and_zp_size,
                    self.out_features // self.pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=self.compute_dtype))
        else:
            self.bias = None

        self.is_lm_head = False
        if kwargs.get("name") is not None and kwargs.get("lm_head_name") is not None:
            self.is_lm_head = kwargs["name"] == kwargs["lm_head_name"]

        if (self.bits, sym) not in self.TYPE_MAP:
            raise ValueError("Unsupported quantization config: "
                             f"bits={self.bits}, sym={sym}")

        self.weight_type = self.TYPE_MAP[(self.bits, sym)]

        # auto-optimize on post init
        # self.optimize()

    # def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
    #     if self.optimized:
    #         return
    #
    #     # compile dequantize
    #     self.forward = torch_compile(self.forward, backend=backend, mode=mode, fullgraph=fullgraph)
    #
    #     super().optimize()


    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        if marlin_import_exception is not None:
            return False, ImportError(marlin_import_exception)
        return True, None


    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Marlin kernel is not supported on ROCm.")

            # Directly check capabilities of all currently visible CUDA devices
            has_supported_cuda = all(
                _marlin_capability_supported(*torch.cuda.get_device_capability(i))
                for i in range(torch.cuda.device_count())
            )
            if not has_supported_cuda:
                raise NotImplementedError(
                    "Marlin kernel only supports compute capability >= 7.5."
                )

    def post_init(self):
        device = self.qweight.device

        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            self._packed_prefill_hardware = (
                properties.major,
                properties.minor,
                properties.multi_processor_count,
            )

        if not marlin_runtime_available(self.compute_dtype):
            raise ModuleNotFoundError(
                "Marlin torch.ops kernels are not properly installed. Error: "
                + marlin_runtime_error(self.compute_dtype)
            )

        self.is_k_full = marlin_is_k_full(self.desc_act, is_row_parallel=False)

        pad_k = self.padded_in_features - self.in_features
        pad_n = self.padded_out_features - self.out_features
        if pad_k or pad_n:
            qweight_pad_rows = pad_k // self.pack_factor
            padded_qweight = torch.nn.functional.pad(
                self.qweight.data,
                (0, pad_n, 0, qweight_pad_rows),
                value=0,
            )
            replace_parameter(self, "qweight", padded_qweight)
            replace_parameter(
                self,
                "scales",
                torch.nn.functional.pad(self.scales.data, (0, pad_n), value=1.0),
            )
            if self.bias is not None:
                self.bias.data = torch.nn.functional.pad(
                    self.bias.data,
                    (0, pad_n),
                    value=0.0,
                )

        # Allocate marlin workspace.
        adapter_workspace_blocks = _LORA_MEGA_KERNEL_WORKSPACE_BLOCKS
        if self.adapter is not None:
            adapter_workspace_blocks = _LORA_MEGA_KERNEL_WORKSPACE_BLOCKS_BY_RANK.get(
                getattr(self.adapter, "rank", None), adapter_workspace_blocks
            )
        self.workspace = marlin_make_workspace_new(
            device,
            min_workspace_blocks=(
                adapter_workspace_blocks if self.adapter is not None else 128
            ),
        )

        def transform_w_q(x):
            x.data = gptq_marlin_repack(x.data.contiguous(),
                                        perm=self.g_idx_sort_indices,
                                        size_k=self.padded_in_features,
                                        size_n=self.padded_out_features,
                                        num_bits=self.bits,
                                        dtype=self.compute_dtype)
            return x

        def transform_w_s(x):
            x.data = marlin_permute_scales(x.data.contiguous(),
                                           size_k=self.padded_in_features,
                                           size_n=self.padded_out_features,
                                           group_size=self.group_size)
            return x

        # Handle sorting for activation reordering if needed.
        if self.desc_act:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(getattr(self, "g_idx"))
            _transform_param(self, "g_idx", lambda _: g_idx)
            self.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(self, "g_idx", marlin_make_empty_g_idx(device))
            self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        setattr(self, "qzeros", marlin_make_empty_g_idx(device))

        _transform_param(self, "qweight", transform_w_q)
        _transform_param(self, "scales", transform_w_s)

        if hasattr(self, "bias") and self.bias is not None:
            self.bias.data = marlin_permute_bias(self.bias)

        super().post_init()
        self.lora_cuda_up_add = marlin_lora_cuda_up_add_enabled()
        self.lora_cooperative_state = None
        if self.adapter is not None:
            use_prepared_marlin = (
                self.weight_type == scalar_types.uint4b8
                and self.group_size == 128
                and self.in_features > self.group_size
                and not self.desc_act
                and self.is_k_full
                and self.fp32
                and self.bias is None
                and self.qzeros.numel() == 0
                and self.g_idx.numel() == 0
                and self.g_idx_sort_indices.numel() == 0
            )
            self.lora_cooperative_state = prepare_marlin_fused_lora(
                self.adapter,
                device=device,
                dtype=self.compute_dtype,
                in_features=self.in_features,
                out_features=self.out_features,
                use_prepared_marlin=use_prepared_marlin,
            )
            if self.lora_cooperative_state is not None:
                log.info.once(
                    "Kernel: Ampere cooperative Marlin+LoRA inference is enabled for eligible decode/small-M shapes."
                )

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "workspace") and self.workspace is not None:
            buf.append(self.workspace)
        if hasattr(self, "g_idx_sort_indices") and self.g_idx_sort_indices is not None:
            buf.append(self.g_idx_sort_indices)
        if hasattr(self, "g_idx") and self.g_idx is not None:
            buf.append(self.g_idx)
        if getattr(self, "lora_cooperative_state", None) is not None:
            lora_workspace = self.lora_cooperative_state[3]
            if lora_workspace is not None:
                buf.append(lora_workspace)
        return buf

    def _packed_prefill_contract_supported(self) -> bool:
        qzeros = getattr(self, "qzeros", None)
        return (
            self.weight_type == scalar_types.uint4b8
            and self.group_size == 128
            and not self.desc_act
            and self.is_k_full
            and qzeros is not None
            and qzeros.numel() == 0
            and self.padded_in_features == self.in_features
            and self.padded_out_features == self.out_features
        )

    def _packed_prefill_decision(self, x: torch.Tensor, *, rows: int) -> _PackedPrefillDecision:
        if not self.packed_prefill:
            return _PACKED_PREFILL_DISABLED
        if rows == 1 or (x.ndim >= 3 and x.shape[-2] == 1):
            return _PACKED_PREFILL_DECODE
        if rows < self.packed_prefill_min_rows:
            return _PackedPrefillDecision(0, "below_min_rows")
        if self.packed_prefill_config != 0:
            return _PackedPrefillDecision(self.packed_prefill_config, "manual_config")
        if not self._packed_prefill_contract_supported():
            return _PackedPrefillDecision(0, "contract_miss")

        hardware = self._packed_prefill_hardware
        if hardware is None:
            if x.device.type != "cuda":
                return _PackedPrefillDecision(0, "hardware_miss")
            properties = torch.cuda.get_device_properties(x.device)
            hardware = (
                properties.major,
                properties.minor,
                properties.multi_processor_count,
            )
            self._packed_prefill_hardware = hardware
        config, reason = _select_packed_prefill_config(
            major=hardware[0],
            minor=hardware[1],
            sms=hardware[2],
            dtype=x.dtype,
            rows=rows,
            size_k=self.in_features,
            size_n=self.out_features,
        )
        return _PackedPrefillDecision(config, reason)

    @torch.inference_mode()
    def dequantize_weight(
        self,
        dtype: Optional[torch.dtype] = None,
        max_chunk_rows: int = 1024,
    ) -> torch.Tensor:
        """Return a dense (in_features, out_features) weight tensor.

        This is implemented by feeding an identity input through the Marlin GEMM:
        each output row is the dequantized weight row for the corresponding input
        dimension.  It is exact with respect to ``self.forward`` and therefore safe
        to use as a dense fallback for grouped GEMM or for correctness probing.
        """
        param = next(iter(self.parameters()), None)
        target_dtype = dtype or self.compute_dtype or (param.dtype if param is not None else self.qweight.dtype)
        device = self.qweight.device

        k_padded = self.padded_in_features
        n_padded = self.padded_out_features
        rows = self.in_features
        out_cols = self.out_features

        if k_padded == 0 or n_padded == 0 or rows == 0:
            return torch.empty(rows, out_cols, dtype=target_dtype, device=device)

        workspace = getattr(self, "workspace", None)
        if workspace is None:
            workspace = marlin_make_workspace_new(device, min_workspace_blocks=128)

        # Process in chunks so the identity-input scratch tensor stays modest for
        # large K (e.g. 7168-dim routed latent in Kimi-K3-like shapes).
        chunks: list[torch.Tensor] = []
        c = torch.empty((max_chunk_rows, n_padded), dtype=target_dtype, device=device)

        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            cur_m = end - start

            a = torch.zeros((cur_m, k_padded), dtype=target_dtype, device=device)
            local_idx = torch.arange(cur_m, device=device)
            a[local_idx, local_idx + start] = 1.0

            c_chunk = c if cur_m == max_chunk_rows else torch.empty((cur_m, n_padded), dtype=target_dtype, device=device)

            gptq_marlin_gemm(
                a,
                c_chunk,
                self.qweight,
                None,
                self.scales,
                None,
                self.qzeros,
                self.g_idx,
                self.g_idx_sort_indices,
                workspace,
                self.weight_type,
                cur_m,
                n_padded,
                k_padded,
                self.is_k_full,
                False,
                self.fp32,
                False,
                False,
                0,
            )
            # Clone the slice so the next iteration's in-place write into `c` cannot
            # overwrite previously-collected full-chunk rows.
            chunks.append(c_chunk[:, :out_cols].clone())

        return torch.cat(chunks, dim=0)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"{self.__class__.__name__} expected input width {self.in_features}, got {x.shape[-1]}."
            )
        if self.padded_in_features != self.in_features:
            x = torch.nn.functional.pad(
                x,
                (0, self.padded_in_features - self.in_features),
                value=0.0,
            )

        input_is_2d = x.dim() == 2
        rows = x.shape[0] if input_is_2d else x.numel() // x.shape[-1]
        packed_prefill_decision = self._packed_prefill_decision(x, rows=rows)
        packed_prefill_config = packed_prefill_decision.config
        use_packed_prefill = packed_prefill_config != 0
        if self.packed_prefill_stats:
            _record_packed_prefill_route(
                decision=packed_prefill_decision,
                hardware=self._packed_prefill_hardware,
                dtype=x.dtype,
                rows=rows,
                size_k=self.in_features,
                size_n=self.out_features,
            )

        cooperative_state = getattr(self, "lora_cooperative_state", None) if self.adapter else None
        if cooperative_state is not None:
            op, lora_a, lora_b, lora_workspace, max_rows, prepared_marlin = cooperative_state
            marlin_input = x if input_is_2d or prepared_marlin else x.reshape(rows, x.shape[-1])
            if (
                0 < rows <= max_rows
                and marlin_input.is_contiguous()
                and marlin_input.dtype == self.scales.dtype
                and (self.bias is None or self.bias.dtype == marlin_input.dtype)
                and marlin_input.dtype == lora_a.dtype
                and marlin_input.device == lora_a.device
                and not torch.cuda.is_current_stream_capturing()
            ):
                if not prepared_marlin and rows > lora_workspace.shape[0]:
                    lora_workspace = torch.empty(
                        (rows, lora_a.shape[1]),
                        dtype=torch.float32,
                        device=marlin_input.device,
                    )
                    cooperative_state = op, lora_a, lora_b, lora_workspace, max_rows, prepared_marlin
                    self.lora_cooperative_state = cooperative_state
                try:
                    if prepared_marlin:
                        out = op(
                            marlin_input,
                            self.qweight,
                            self.scales,
                            self.workspace,
                            lora_a,
                            lora_b,
                            use_packed_prefill,
                            packed_prefill_config,
                        )
                    else:
                        out = op(
                            marlin_input,
                            None,
                            self.qweight,
                            self.bias,
                            self.scales,
                            None,
                            self.qzeros,
                            self.g_idx,
                            self.g_idx_sort_indices,
                            self.workspace,
                            lora_a,
                            lora_b,
                            lora_workspace,
                            self.weight_type.id,
                            rows,
                            self.out_features,
                            self.in_features,
                            self.is_k_full,
                            False,
                            self.fp32,
                            False,
                            use_packed_prefill,
                            packed_prefill_config,
                        )
                    if input_is_2d or prepared_marlin:
                        return out
                    return out.reshape(x.shape[:-1] + (self.out_features,))
                except Exception as exc:
                    log.warn.once(
                        "Integrated Marlin+LoRA inference failed at runtime; using the standard adapter path: "
                        f"{exc}"
                    )
                    self.lora_cooperative_state = None

        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # make sure scales is synced with x/input
        if x.dtype != self.scales.dtype:
            replace_parameter(self, "scales", self.scales.to(dtype=x.dtype))
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(dtype=x.dtype)

        x_2d = x if input_is_2d else x.reshape(rows, x.shape[-1])
        marlin_input = x_2d.contiguous() if self.is_lm_head else x_2d
        out_shape = x.shape[:-1] + (self.out_features,)
        out = None
        adapter_applied = False
        cooperative_state = getattr(self, "lora_cooperative_state", None) if self.adapter else None
        if cooperative_state is not None:
            op, lora_a, lora_b, lora_workspace, max_rows, prepared_marlin = cooperative_state
            can_use_cooperative = (
                rows <= max_rows
                and marlin_input.is_contiguous()
                and marlin_input.dtype == lora_a.dtype
                and marlin_input.device == lora_a.device
                # The three library kernels have less raw GPU work once launch
                # overhead is removed by graph replay.
                and not torch.cuda.is_current_stream_capturing()
            )
            if can_use_cooperative:
                if not prepared_marlin and rows > lora_workspace.shape[0]:
                    lora_workspace = torch.empty(
                        (rows, lora_a.shape[1]),
                        dtype=torch.float32,
                        device=marlin_input.device,
                    )
                    cooperative_state = op, lora_a, lora_b, lora_workspace, max_rows, prepared_marlin
                    self.lora_cooperative_state = cooperative_state
                try:
                    if prepared_marlin:
                        out = op(
                            marlin_input,
                            self.qweight,
                            self.scales,
                            self.workspace,
                            lora_a,
                            lora_b,
                            use_packed_prefill,
                            packed_prefill_config,
                        )
                    else:
                        out = op(
                            marlin_input,
                            None,
                            self.qweight,
                            self.bias,
                            self.scales,
                            None,
                            self.qzeros,
                            self.g_idx,
                            self.g_idx_sort_indices,
                            self.workspace,
                            lora_a,
                            lora_b,
                            lora_workspace,
                            self.weight_type.id,
                            rows,
                            self.out_features,
                            self.in_features,
                            self.is_k_full,
                            False,
                            self.fp32,
                            False,
                            use_packed_prefill,
                            packed_prefill_config,
                        )
                    adapter_applied = True
                except Exception as exc:
                    log.warn.once(
                        "Integrated Marlin+LoRA inference failed at runtime; using the standard adapter path: "
                        f"{exc}"
                    )
                    self.lora_cooperative_state = None

        if out is None:
            out = apply_gptq_marlin_linear(
                input=marlin_input,
                weight=self.qweight,
                weight_scale=self.scales,
                weight_zp=self.qzeros,
                g_idx=self.g_idx,
                g_idx_sort_indices=self.g_idx_sort_indices,
                workspace=self.workspace,
                wtype=self.weight_type,
                output_size_per_partition=self.padded_out_features,
                input_size_per_partition=self.padded_in_features,
                is_k_full=self.is_k_full,
                bias=self.bias,
                use_fp32_reduce=self.fp32,
                use_atomics=False, # reduces accuracy with slightly faster performance
                use_packed_prefill=use_packed_prefill,
                packed_prefill_config=packed_prefill_config,
            )

        if self.padded_out_features != self.out_features:
            out = out[:, :self.out_features]

        if self.adapter and not adapter_applied:
            if self.lora_cuda_up_add:
                fused_out = apply_marlin_fused_lora(
                    self.adapter,
                    x=x_2d,
                    out=out,
                )
                out = fused_out if fused_out is not None else self.adapter.apply(x=x_2d, out=out)
            else:
                out = self.adapter.apply(x=x_2d, out=out)

        return out if input_is_2d else out.reshape(out_shape)


# Precompute permutations for Marlin weight and scale shuffling
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


def unpack_qzeros(qzeros):
    unpacked_zeros = torch.zeros(
        (qzeros.shape[0], qzeros.shape[1] * 8),
        dtype=torch.int8,
        device=qzeros.device,
        requires_grad=False,
    )

    for col in range(unpacked_zeros.shape[1]):
        i = col % 8
        unpacked_zeros[:, col] = (qzeros[:, col // 8] >> (4 * i)) & 0xF

    return unpacked_zeros


def dequantize_qzeros(layer):
    qzeros = layer.qzeros
    unpacked_qzeros = unpack_qzeros(qzeros)
    group_size = layer.group_size
    unpacked_qzeros = unpacked_qzeros.repeat_interleave(group_size, dim=0)

    return unpacked_qzeros


__all__ = ["MarlinLinear"]
