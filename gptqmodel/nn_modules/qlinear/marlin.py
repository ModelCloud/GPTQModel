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
    apply_gptq_marlin_linear_padded,
    gptq_marlin_repack,
    marlin_import_exception,
    marlin_is_tile_aligned,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_pad_dim,
    marlin_pad_qweight,
    marlin_pad_scales,
    marlin_padded_nk,
    marlin_permute_bias,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_runtime_available,
    marlin_runtime_error,
    marlin_sort_g_idx,
    replace_parameter,
)
from ...utils.marlin_scalar_type import scalar_types
from ...utils.rocm import IS_ROCM


log = setup_logger()


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
    # Tile padding is handled below; group boundaries must still divide K.
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [1]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [1]

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

        if desc_act and group_size in (-1, in_features):
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        selected_backend = kwargs.pop("backend", BACKEND.GPTQ_MARLIN)
        # Padding adds work to every forward, so automatic selection stays conservative.
        if selected_backend in (BACKEND.AUTO, BACKEND.AUTO_TRAINABLE) and not marlin_is_tile_aligned(
            out_features, in_features
        ):
            raise NotImplementedError(
                "Automatic Marlin selection keeps tile-misaligned shapes on "
                "the next compatible backend; request GPTQ_MARLIN explicitly "
                "to enable runtime tile padding."
            )

        self.compute_dtype = kwargs.get("dtype") or torch.float16
        self.fp32 = env_flag("GPTQMODEL_MARLIN_USE_FP32", default=True)

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=selected_backend,
            adapter=adapter,
            register_buffers=False, # do not register buffers in super()
            **kwargs)

        if not self.fp32:
            log.warn.once(
                "Kernel: GPTQMODEL_MARLIN_USE_FP32 is disabled. Marlin will use reduced-precision reduction.")

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
    def _validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        ok, err = super()._validate(**args)
        if not ok:
            return ok, err

        bits = args.get("bits", 4)
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        desc_act = args.get("desc_act", False)
        group_size = args.get("group_size", -1)
        if in_features is None or out_features is None:
            return True, None

        pack_factor = 32 // bits
        # Tile padding cannot repair a partially packed int32 row or column.
        if in_features % pack_factor != 0 or out_features % pack_factor != 0:
            return False, NotImplementedError(
                "Marlin packed dimensions must be divisible by "
                f"pack_factor={pack_factor}; got K={in_features}, "
                f"N={out_features}."
            )

        effective_desc_act = desc_act and group_size not in (-1, in_features)
        # Act-order indices only describe the original K dimension.
        if effective_desc_act and not marlin_is_tile_aligned(
            out_features, in_features
        ):
            return False, NotImplementedError(
                "Marlin activation-order weights require an aligned thread "
                f"tile; got K={in_features}, N={out_features}."
            )

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
        self.workspace = marlin_make_workspace_new(device)

        # GPTQModel also accepts group_size=K as channelwise quantization.
        marlin_group_size = (
            -1
            if self.requested_group_size == self.in_features
            else self.requested_group_size
        )
        # Validation keeps act-order shapes aligned; other shapes may use zero padding.
        if self.desc_act:
            padded_n, padded_k = self.out_features, self.in_features
        else:
            padded_n, padded_k = marlin_padded_nk(
                self.out_features,
                self.in_features,
                marlin_group_size,
            )
        self._marlin_tile_padding = (
            None
            if (padded_n, padded_k) == (self.out_features, self.in_features)
            else (padded_n, padded_k)
        )

        def transform_w_q(x):
            # Pad in GPTQ layout before converting to Marlin layout.
            padded = marlin_pad_qweight(
                x.data.contiguous(),
                self.out_features,
                self.in_features,
                padded_n,
                padded_k,
            )
            x.data = gptq_marlin_repack(padded,
                                        perm=self.g_idx_sort_indices,
                                        size_k=padded_k,
                                        size_n=padded_n,
                                        num_bits=self.bits,
                                        dtype=self.compute_dtype)
            return x

        def transform_w_s(x):
            padded = marlin_pad_scales(
                x.data.contiguous(),
                self.out_features,
                self.in_features,
                padded_n,
                padded_k,
                marlin_group_size,
            )
            x.data = marlin_permute_scales(
                padded,
                size_k=padded_k,
                size_n=padded_n,
                group_size=marlin_group_size,
            )
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
            self.bias.data = marlin_permute_bias(
                marlin_pad_dim(self.bias, self.out_features, padded_n)
            )

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "workspace") and self.workspace is not None:
            buf.append(self.workspace)
        if hasattr(self, "g_idx_sort_indices") and self.g_idx_sort_indices is not None:
            buf.append(self.g_idx_sort_indices)
        if hasattr(self, "g_idx") and self.g_idx is not None:
            buf.append(self.g_idx)
        return buf

    def forward(self, x: torch.Tensor):
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # make sure scales is synced with x/input
        if x.dtype != self.scales.dtype:
            replace_parameter(self, "scales", self.scales.to(dtype=x.dtype))
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(dtype=x.dtype)

        # Keep aligned layers on the original decode-sensitive call path.
        if self._marlin_tile_padding is None:
            out = apply_gptq_marlin_linear(
                input=x.contiguous() if self.is_lm_head else x,
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
            )
        else:
            out = apply_gptq_marlin_linear_padded(
                tile_padding=self._marlin_tile_padding,
                input=x.contiguous() if self.is_lm_head else x,
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
                use_atomics=False,
            )

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out


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
