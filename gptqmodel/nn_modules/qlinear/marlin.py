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
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM
from ...utils.scalar_type import scalar_types, ScalarType

marlin_import_exception = None
try:
    import gptqmodel_marlin_kernels
except ImportError as e:
    marlin_import_exception = str(e)

log = setup_logger()

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16


def set_weight_attrs(
        weight: torch.Tensor,
        weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(act_order: bool, group_size: int,
                                      is_row_parallel: bool) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_workspace_new(device: torch.device,
                              max_blocks_per_sm: int = 1) -> torch.Tensor:
    # In the new marlin kernel, we use the num of threadblocks as workspace
    # size. The num of threadblocks is sms_count * max_blocks_per_sm.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm,
                       dtype=torch.int,
                       device=device,
                       requires_grad=False)


def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"

    # update tensor shape and stride
    dst.as_strided_(src.shape, src.stride())

    # If not the same underlying storage move tensor data
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
        del src


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_parameter(mod: torch.nn.Module, name: str,
                      new: Union[torch.Tensor, torch.nn.Parameter]) -> None:
    old = getattr(mod, name)
    if type(old) is type(new) and old.dtype == new.dtype and \
            old.untyped_storage().nbytes() == new.untyped_storage().nbytes():
        # If we can just update in-place to avoid re-registering
        #   can be faster if the underlying storage is the same
        update_tensor_inplace(old, new)
    else:
        # Fallback re-register parameter, convert to Parameter if necessary
        # this not only ensures we don't register a tensor as a parameter, but
        # also ensures that all parameter subclasses get re-registered as
        # parameters for `torch.compile` compatibility
        if not isinstance(new, torch.nn.Parameter):
            new = torch.nn.Parameter(new, requires_grad=False)
        mod.register_parameter(name,
                               torch.nn.Parameter(new, requires_grad=False))


def _transform_param(layer: torch.nn.Module, name: Optional[str],
                     fn: Callable) -> None:
    if name is not None and getattr(layer, name, None) is not None:
        old_param = getattr(layer, name)
        new_param = fn(old_param)
        # replace the parameter with torch.nn.Parameter for TorchDynamo
        # compatibility
        replace_parameter(
            layer, name,
            torch.nn.Parameter(new_param.data, requires_grad=False))


def marlin_sort_g_idx(
        g_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                              requires_grad=False)


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_tensor(layer: torch.nn.Module, name: str,
                   new_t: torch.Tensor) -> None:
    # It is important to use resize_() here since it ensures
    # the same buffer is reused
    getattr(layer, name).resize_(new_t.shape)
    getattr(layer, name).copy_(new_t)
    del new_t


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


# In case there is a performance issue with Marlin, the variable below can be
# changed to False, which allows Marlin to perform global reductions in fp16
# precision (instead of fp32), and therefore, save on some memory movements.
USE_FP32_REDUCE_DEFAULT = True

# Whether to use atomicAdd reduce in gptq/awq marlin kernel. experimental
VLLM_MARLIN_USE_ATOMIC_ADD = False


def maybe_warn_marlin_atomic_add_env():
    if torch.compiler.is_dynamo_compiling():
        return
    if VLLM_MARLIN_USE_ATOMIC_ADD:
        return
    log.info_once(
        "Marlin kernel can achieve better performance for small size_n "
        "with experimental use_atomic_add feature. "
        "You can consider set environment variable "
        "VLLM_MARLIN_USE_ATOMIC_ADD to 1 if possible.")


def maybe_warn_marlin_atomic_add(device, dtype):
    if torch.compiler.is_dynamo_compiling():
        return
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        log.info_once(
            "You are running Marlin kernel with bf16 on GPUs before SM90. "
            "You can consider change to fp16 to achieve better performance "
            "if possible.")


def should_use_atomic_add_reduce(m: int, n: int, k: int, device: torch.device,
                                 dtype: torch.dtype) -> bool:
    # the performance of atomicAdd is better than global reduce
    # only when m*n is small and k is large
    if n >= 2048 or k < 2048 or device.type != "cuda":
        return False

    # disable atomicAdd reduce by default,
    # one can enable it with VLLM_MARLIN_USE_ATOMIC_ADD=1
    if not VLLM_MARLIN_USE_ATOMIC_ADD:
        maybe_warn_marlin_atomic_add_env()
        return False

    # sm8x doesn't support atomicAdd + bfloat16 natively
    device_capability = torch.cuda.get_device_capability(device)
    if device_capability[0] < 9 and dtype == torch.bfloat16:
        maybe_warn_marlin_atomic_add(device, dtype)
        return False

    return True


def apply_gptq_marlin_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        g_idx: torch.Tensor,
        g_idx_sort_indices: torch.Tensor,
        workspace: torch.Tensor,
        wtype: ScalarType,
        output_size_per_partition: int,
        input_size_per_partition: int,
        is_k_full: bool,
        bias: Optional[torch.Tensor] = None,
        use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    use_atomic_add = should_use_atomic_add_reduce(m=reshaped_x.size(0),
                                                  n=output_size_per_partition,
                                                  k=reshaped_x.size(1),
                                                  device=input.device,
                                                  dtype=input.dtype)

    output = gptqmodel_marlin_kernels.gptq_marlin_gemm(reshaped_x,
                                                       None,
                                                       weight,
                                                       bias,
                                                       weight_scale,
                                                       None,
                                                       weight_zp,
                                                       g_idx,
                                                       g_idx_sort_indices,
                                                       workspace,
                                                       wtype.id,
                                                       size_m=reshaped_x.shape[0],
                                                       size_n=output_size_per_partition,
                                                       size_k=input_size_per_partition,
                                                       is_k_full=is_k_full,
                                                       use_atomic_add=use_atomic_add,
                                                       use_fp32_reduce=use_fp32_reduce,
                                                       is_zp_float=False)

    return output.reshape(out_shape)


class MarlinQuantLinear(BaseQuantLinear):
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
            adapter: Adapter = None,
            **kwargs):
        if marlin_import_exception is not None:
            raise ValueError(
                f"Trying to use the marlin backend, but could not import the C++/CUDA dependencies with the following error: {marlin_import_exception}"
            )

        # self.original_in_features = in_features
        # self.original_out_features = out_features

        if desc_act and group_size == -1:
            # In this case, act_order == True is the same as act_order == False
            # (since we have only one group per output channel)
            desc_act = False

        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.MARLIN),
            adapter=adapter,
            register_buffers=False,
            **kwargs)

        # toggle fp32 mode depending on MARLIN or MARLIN_FP16 backend
        self.fp32 = True if self.backend in [BACKEND.MARLIN, BACKEND.AUTO] else False

        if not self.fp32:
            log.warn.once(
                "Kernel: Marlin FP16 mode is activated with reduced accuracy. Use default Marlin model for improved inference quality.")

        # Determine sharding
        if marlin_repeat_scales_on_all_ranks(desc_act,
                                             self.group_size,
                                             is_row_parallel=False):
            # By setting scale_dim == None, weight_loader will
            # repeat the scales on each GPU in TP>1 case.
            scales_and_zp_input_dim = None
            scales_and_zp_size = self.in_features // self.group_size
        else:
            # By setting scale_dim == 0, weight_loader will
            # shard the scales in TP>1 case.
            scales_and_zp_input_dim = 0
            scales_and_zp_size = self.in_features // self.group_size

        # Quantized weights
        self.register_buffer(
            "qweight",
            torch.empty(
                self.in_features // self.pack_factor,
                self.out_features,
                dtype=torch.int32,
            ),
        )

        # Activation order
        self.register_buffer(
            "g_idx",
            torch.empty(
                self.in_features,
                dtype=torch.int32,
            ),
        )

        # Scales
        self.register_buffer(
            "scales",
            torch.empty(
                scales_and_zp_size,
                self.out_features,
                dtype=torch.float16,
            ),
        )

        # Quantized zero-points
        self.register_buffer(
            "qzeros",
            torch.empty(
                scales_and_zp_size,
                self.out_features // self.pack_factor,
                dtype=torch.int32,
            ),
        )

        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=torch.float16))
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
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        print("marlin_import_exception", marlin_import_exception)
        if marlin_import_exception is not None:
            return False, ImportError(marlin_import_exception)
        return cls._validate(**args)

    @classmethod
    def validate_device(cls, device: DEVICE):
        super().validate_device(device)
        CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
        if device == DEVICE.CUDA:
            if IS_ROCM:
                raise NotImplementedError("Marlin kernel is not supported on ROCm.")

            if CUDA_VISIBLE_DEVICES is None:
                has_cuda_v8 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count()))
            else:
                has_cuda_v8 = all(
                    torch.cuda.get_device_capability(i)[0] >= 8 for i in range(len(CUDA_VISIBLE_DEVICES.split(","))))
            if not has_cuda_v8:
                raise NotImplementedError("Marlin kernel only supports compute capability >= 8.0.")

    def post_init(self):
        device = self.qweight.device

        self.is_k_full = marlin_is_k_full(self.desc_act, is_row_parallel=False)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace_new(device)

        def transform_w_q(x):
            x.data = gptqmodel_marlin_kernels.gptq_marlin_repack(x.data.contiguous(),
                                                                 perm=self.g_idx_sort_indices,
                                                                 size_k=self.in_features,
                                                                 size_n=self.out_features,
                                                                 num_bits=self.bits)
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

        super().post_init()

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "workspace") and self.workspace is not None:
            buf.append(self.workspace)
        if hasattr(self, "g_idx_sort_indices") and self.g_idx_sort_indices is not None:
            buf.append(self.g_idx_sort_indices)
        return buf

    def forward(self, x: torch.Tensor):
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # make sure scales is synced with x/input
        if x.dtype != self.scales.dtype:
            self.scales = self.scales.to(dtype=x.dtype)

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


__all__ = ["MarlinQuantLinear"]
