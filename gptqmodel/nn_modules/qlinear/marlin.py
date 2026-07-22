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

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import GPTQQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.env import env_flag
from ...utils.logger import setup_logger
from ...utils.marlin import (
    _marlin_capability_supported,
    _transform_param,
    apply_gptq_marlin_linear,
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
from ...utils.eora_marlin import (
    apply_eora_marlin_fused_lora,
    eora_marlin_cuda_up_add_enabled,
    prepare_eora_marlin_fused_lora,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


# Sample process-level routing policy once when each MarlinLinear is created;
# the native dispatcher still makes the final hardware/shape decision per call.
_PACKED_PREFILL_ENV = "GPTQMODEL_MARLIN_PACKED_PREFILL"
_PACKED_PREFILL_MIN_ROWS_ENV = "GPTQMODEL_MARLIN_PACKED_PREFILL_MIN_ROWS"
_EORA_MEGA_KERNEL_WORKSPACE_BLOCKS = 192
_PACKED_PREFILL_CONFIG_ENV = "GPTQMODEL_MARLIN_PACKED_PREFILL_CONFIG"
_PACKED_PREFILL_MIN_ROWS_DEFAULT = 1024


def _packed_prefill_enabled() -> bool:
    """Enable conservative automatic packed-prefill routing by default."""
    return env_flag(_PACKED_PREFILL_ENV, default=True)


def _should_use_packed_prefill(x: torch.Tensor, min_rows: int) -> bool:
    """Route only genuine multi-token work to the packed prefill kernel."""
    if x.ndim >= 3 and x.shape[-2] == 1:
        return False
    rows = x.numel() // x.shape[-1]
    return rows >= min_rows


class MarlinLinear(GPTQQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.GPTQ_MARLIN]
    SUPPORTS_METHODS = [METHOD.GPTQ]
    SUPPORTS_FORMATS = {FORMAT.GPTQ: 90, FORMAT.GPTQ_V2: 90, FORMAT.MARLIN: 90}
    SUPPORTS_BITS = [4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

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
            scales_and_zp_size = self.in_features // self.group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_size = self.in_features // self.group_size

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

        if not marlin_runtime_available(self.compute_dtype):
            raise ModuleNotFoundError(
                "Marlin torch.ops kernels are not properly installed. Error: "
                + marlin_runtime_error(self.compute_dtype)
            )

        self.is_k_full = marlin_is_k_full(self.desc_act, is_row_parallel=False)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(
            device,
            min_workspace_blocks=(
                _EORA_MEGA_KERNEL_WORKSPACE_BLOCKS if self.adapter is not None else 128
            ),
        )

        def transform_w_q(x):
            x.data = gptq_marlin_repack(x.data.contiguous(),
                                        perm=self.g_idx_sort_indices,
                                        size_k=self.in_features,
                                        size_n=self.out_features,
                                        num_bits=self.bits,
                                        dtype=self.compute_dtype)
            return x

        def transform_w_s(x):
            x.data = marlin_permute_scales(x.data.contiguous(),
                                           size_k=self.in_features,
                                           size_n=self.out_features,
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
        self.eora_cuda_up_add = eora_marlin_cuda_up_add_enabled()
        self.eora_cooperative_state = None
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
            self.eora_cooperative_state = prepare_eora_marlin_fused_lora(
                self.adapter,
                device=device,
                dtype=self.compute_dtype,
                in_features=self.in_features,
                out_features=self.out_features,
                use_prepared_marlin=use_prepared_marlin,
            )
            if self.eora_cooperative_state is not None:
                log.info.once(
                    "Kernel: Ampere cooperative Marlin+EoRA inference is enabled for eligible decode/small-M shapes."
                )

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "workspace") and self.workspace is not None:
            buf.append(self.workspace)
        if hasattr(self, "g_idx_sort_indices") and self.g_idx_sort_indices is not None:
            buf.append(self.g_idx_sort_indices)
        if hasattr(self, "g_idx") and self.g_idx is not None:
            buf.append(self.g_idx)
        if getattr(self, "eora_cooperative_state", None) is not None:
            eora_workspace = self.eora_cooperative_state[3]
            if eora_workspace is not None:
                buf.append(eora_workspace)
        return buf

    def forward(self, x: torch.Tensor):
        cooperative_state = getattr(self, "eora_cooperative_state", None) if self.adapter else None
        if cooperative_state is not None:
            op, lora_a, lora_b, eora_workspace, max_rows, prepared_marlin = cooperative_state
            input_is_2d = x.dim() == 2
            rows = x.shape[0] if input_is_2d else x.numel() // x.shape[-1]
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
                if not prepared_marlin and rows > eora_workspace.shape[0]:
                    eora_workspace = torch.empty(
                        (rows, lora_a.shape[1]),
                        dtype=torch.float32,
                        device=marlin_input.device,
                    )
                    cooperative_state = op, lora_a, lora_b, eora_workspace, max_rows, prepared_marlin
                    self.eora_cooperative_state = cooperative_state
                use_packed_prefill = (
                    self.packed_prefill
                    and rows >= self.packed_prefill_min_rows
                    and (input_is_2d or x.dim() < 3 or x.shape[-2] != 1)
                )
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
                            self.packed_prefill_config,
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
                            eora_workspace,
                            self.weight_type.id,
                            rows,
                            self.out_features,
                            self.in_features,
                            self.is_k_full,
                            False,
                            self.fp32,
                            False,
                            use_packed_prefill,
                            self.packed_prefill_config,
                        )
                    if input_is_2d or prepared_marlin:
                        return out
                    return out.reshape(x.shape[:-1] + (self.out_features,))
                except Exception as exc:
                    log.warn.once(
                        "Integrated Marlin+EoRA inference failed at runtime; using the standard adapter path: "
                        f"{exc}"
                    )
                    self.eora_cooperative_state = None

        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # make sure scales is synced with x/input
        if x.dtype != self.scales.dtype:
            replace_parameter(self, "scales", self.scales.to(dtype=x.dtype))
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(dtype=x.dtype)

        use_packed_prefill = self.packed_prefill and _should_use_packed_prefill(
            x,
            self.packed_prefill_min_rows,
        )
        input_is_2d = x.dim() == 2
        rows = x.numel() // x.shape[-1]
        x_2d = x if input_is_2d else x.reshape(rows, x.shape[-1])
        marlin_input = x_2d.contiguous() if self.is_lm_head else x_2d
        out_shape = x.shape[:-1] + (self.out_features,)
        out = None
        adapter_applied = False
        cooperative_state = getattr(self, "eora_cooperative_state", None) if self.adapter else None
        if cooperative_state is not None:
            op, lora_a, lora_b, eora_workspace, max_rows, prepared_marlin = cooperative_state
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
                if not prepared_marlin and rows > eora_workspace.shape[0]:
                    eora_workspace = torch.empty(
                        (rows, lora_a.shape[1]),
                        dtype=torch.float32,
                        device=marlin_input.device,
                    )
                    cooperative_state = op, lora_a, lora_b, eora_workspace, max_rows, prepared_marlin
                    self.eora_cooperative_state = cooperative_state
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
                            self.packed_prefill_config,
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
                            eora_workspace,
                            self.weight_type.id,
                            rows,
                            self.out_features,
                            self.in_features,
                            self.is_k_full,
                            False,
                            self.fp32,
                            False,
                            use_packed_prefill,
                            self.packed_prefill_config,
                        )
                    adapter_applied = True
                except Exception as exc:
                    log.warn.once(
                        "Integrated Marlin+EoRA inference failed at runtime; using the standard adapter path: "
                        f"{exc}"
                    )
                    self.eora_cooperative_state = None

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
                output_size_per_partition=self.out_features,
                input_size_per_partition=self.in_features,
                is_k_full=self.is_k_full,
                bias=self.bias,
                use_fp32_reduce=self.fp32,
                use_atomics=False, # reduces accuracy with slightly faster performance
                use_packed_prefill=use_packed_prefill,
                packed_prefill_config=self.packed_prefill_config,
            )

        if self.adapter and not adapter_applied:
            if self.eora_cuda_up_add:
                fused_out = apply_eora_marlin_fused_lora(
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
