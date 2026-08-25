# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Adapted from vllm at https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/gptq_marlin.py

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import CPU, DEVICE, PLATFORM
from ...nn_modules.qlinear import GroupedQuantLinear
from ...quantization import FORMAT, METHOD
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.qqq import qqq_gemm, qqq_runtime_available, qqq_runtime_error
from ...utils.rocm import IS_ROCM


log = setup_logger()


_INT4_DIVISORS = (1, 16, 256, 4096, 65536, 1048576, 16777216, 268435456)


def _invert_permutation(values) -> torch.Tensor:
    inverse = [0] * len(values)
    for index, value in enumerate(values):
        inverse[int(value)] = index
    return torch.tensor(inverse, dtype=torch.long)


def _unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    packed_i64 = packed.to(torch.int64)
    packed_u32 = torch.where(packed_i64 < 0, packed_i64 + (1 << 32), packed_i64)
    divisors = torch.tensor(_INT4_DIVISORS, dtype=torch.int64, device=packed.device)
    quotients = torch.floor_divide(packed_u32.unsqueeze(-1), divisors)
    lanes = quotients - torch.floor_divide(quotients, 16) * 16
    return lanes.reshape(packed.shape[0], packed.shape[1] * 8)


def mul(
    A, B, C, D, s1, s2, s3, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16
):
    """INT8xINT4 multiply based on Marlin kernel; can be used within `torch.compile`.
    @A: `torch.int8` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int32` weight matrix of original shape `(k, n)` in the specified format; see `Layer.pack()`
    @C: `torch.int32` reduce buffer of shape `(max_par * 64, n)` in standard row-major layout
    @D: `torch.float16` out matrix of shape `(m, n)` in standard row-major layout
    @s1: `torch.float32` activation per-token quantization scales of shape `(m, 1)`
    @s2: `torch.float32` weight per-channel quantization scales of shape `(1, n)`
    @s3: `torch.float16` weight per-group quantization scales of shape `(m / groupsize, n)`, it should be empty when group_size != -1
    @workspace: `torch.int32` tensor with at least `n / 128 * max_par` entries that are all zero
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    if not qqq_runtime_available():
        raise ModuleNotFoundError("QQQ torch.ops kernels are not properly installed. Error: " + qqq_runtime_error())
    qqq_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, sms, max_par)


class QQQLinear(GroupedQuantLinear):
    SUPPORTS_BACKENDS = [BACKEND.QQQ]
    SUPPORTS_METHODS = [METHOD.QQQ]
    SUPPORTS_FORMATS = {FORMAT.QQQ: 100}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA, DEVICE.ROCM]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False

    # for transformers/optimum tests compat
    QUANT_TYPE = "qqq"

    IN_OUTPUT_FEATURES_DIVISIBLE_BY = [(64, 256), (128, 128), (128, 64), (64, 128)]

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
        register_buffers: bool = True,
        **kwargs):
        self.tile = 16
        self.max_par = 16

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
            backend=kwargs.pop("backend", BACKEND.QQQ),
            adapter=adapter,
            # QQQ does it's own buffer setup
            register_buffers=False,
            **kwargs)

        # QQQ only needs the code range, not packed GPTQ/AWQ storage metadata.
        self.maxq = (1 << self.bits) - 1

        # during quantization, we do are not loading tensors from disk so no need to preallocate buffers
        if register_buffers:
            self.register_buffer(
                "B",
                torch.empty(
                    (self.in_features // 16, self.out_features * 16 // 8), dtype=torch.int32
                ),
            )
            self.register_buffer(
                "s_channel",
                torch.empty(
                    (1, self.out_features),
                    dtype=torch.float32,
                ),
            )
            if self.group_size != self.in_features:
                self.register_buffer(
                    "s_group",
                    torch.empty(
                        (self.in_features // self.group_size, self.out_features),
                        dtype=torch.float16,
                    ),
                )
            else:
                self.register_buffer(
                    "s_group",
                    torch.tensor([], dtype=torch.float16),
                )
            # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
            self.register_buffer(
                "workspace",
                torch.zeros(self.out_features // 128 * 16, dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "reduce_buffer",
                torch.zeros((self.max_par * 16 * 4, self.out_features), dtype=torch.int),
                persistent=False,
            )
            if bias:
                self.register_buffer("bias", torch.zeros((self.out_features), dtype=torch.float16))
            else:
                self.bias = None


        (
            self._perm,
            self._perm_i,
            self._scale_perm,
            self._scale_perm_i,
            self._scale_perm_single,
            self._scale_perm_single_i,
        ) = self._get_perms()

        # auto-optimize on post init
        # self.optimize()

    def _get_perms(self):
        perm = []
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm.extend([p + 256 * j for p in perm1])

        perm = np.array(perm)
        if self.group_size == self.in_features:
            interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
        perm = perm.reshape((-1, 8))[:, interleave].ravel()
        perm = torch.from_numpy(perm)
        perm_i = _invert_permutation(perm.tolist())
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm_i = _invert_permutation(scale_perm)
        scale_perm_single = []
        for i in range(4):
            scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        scale_perm_single_i = _invert_permutation(scale_perm_single)
        return perm, perm_i, scale_perm, scale_perm_i, scale_perm_single, scale_perm_single_i

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
        if not qqq_runtime_available():
            return False, ImportError(qqq_runtime_error())
        return True, None

    @classmethod
    def validate(cls, **args) -> Tuple[bool, Optional[Exception]]:
        in_features = args.get("in_features")
        out_features = args.get("out_features")
        if in_features and out_features and not any(
                in_features % thread_k == 0 and out_features % thread_n == 0
                    for thread_k, thread_n in cls.IN_OUTPUT_FEATURES_DIVISIBLE_BY
        ):
            raise ValueError(f"{cls} not supported `infeatures`: {in_features} and `outfeatures`: {out_features}.")

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
                has_cuda_v8 = all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(len(CUDA_VISIBLE_DEVICES.split(","))))
            if not has_cuda_v8:
                raise NotImplementedError("Marlin kernel only supports compute capability >= 8.0.")

    def post_init(self):
        super().post_init()

        self.s_channel = self.s_channel.to(dtype=torch.float32)
        self.s_group = self.s_group.to(dtype=torch.float16)

    def list_buffers(self) -> List:
        buf = super().list_buffers()
        if hasattr(self, "B") and self.B is not None:
            buf.append(self.B)
        if hasattr(self, "s_channel") and self.s_channel is not None:
            buf.append(self.s_channel)
        if hasattr(self, "s_group") and self.s_group is not None:
            buf.append(self.s_group)
        if hasattr(self, "workspace") and self.workspace is not None:
            buf.append(self.workspace)
        if hasattr(self, "reduce_buffer") and self.reduce_buffer is not None:
            buf.append(self.reduce_buffer)
        return buf

    def pack(self, linear: torch.nn.Module, scales: torch.Tensor, s_extra=None):
        """Pack a fake-quantized linear layer into the Marlin representation."""
        if self.group_size != self.in_features:
            assert s_extra is not None, "s_extra is needed"
        if linear.weight.dtype != torch.float16:
            log.warn.once(
                f"""The dtype of weights is {linear.weight.dtype}, while our w4a8 GEMM's output is torch.float16.
                If you can ensure your GEMM results don't overflow torch.float16, it will still function correctly.
                Otherwise, it will yield incorrect results."""
            )

        raw_scales = scales.t()
        if self.group_size != self.in_features:
            s_extra = s_extra.reshape(1, -1).to(dtype=torch.float32)
            packed_s_group = (raw_scales / s_extra).to(dtype=torch.float16)
            packed_s_group = packed_s_group.reshape(
                (-1, len(self._scale_perm))
            )[:, self._scale_perm].reshape((-1, self.out_features)).contiguous()
            packed_s_channel = s_extra.reshape(
                (-1, len(self._scale_perm_single))
            )[:, self._scale_perm_single].reshape((-1, self.out_features)).contiguous()
        else:
            packed_s_group = None
            packed_s_channel = (
                (raw_scales / (2 ** (8 - self.bits)))
                .reshape((-1, len(self._scale_perm_single)))[:, self._scale_perm_single]
                .to(dtype=torch.float32)
                .reshape((-1, self.out_features))
                .contiguous()
            )

        input_chunk_size = min(1024, self.in_features)
        output_chunk_size = min(256, self.out_features)
        input_chunk_size -= input_chunk_size % self.tile
        output_chunk_size -= output_chunk_size % 64
        if input_chunk_size == 0 or output_chunk_size == 0:
            raise ValueError(
                f"QQQ pack requires dimensions divisible by {self.tile} and 64, "
                f"got in_features={self.in_features}, out_features={self.out_features}"
            )

        packed_weight = torch.empty(
            (self.in_features // self.tile, self.out_features * 2),
            dtype=torch.int32,
            device=CPU,
        )
        weight = linear.weight.data
        perm = self._perm.to(device=weight.device)
        for input_start in range(0, self.in_features, input_chunk_size):
            input_end = min(input_start + input_chunk_size, self.in_features)
            input_size = input_end - input_start
            if input_size % self.tile != 0:
                raise ValueError("QQQ pack input chunk is not tile aligned")

            input_indices = torch.arange(
                input_start,
                input_end,
                device=weight.device,
                dtype=torch.long,
            )
            if self.group_size != self.in_features:
                group_indices = torch.div(input_indices, self.group_size, rounding_mode="floor")
                scale_chunk = raw_scales.index_select(0, group_indices)
            else:
                scale_chunk = raw_scales.expand(input_size, -1)

            for output_start in range(0, self.out_features, output_chunk_size):
                output_end = min(output_start + output_chunk_size, self.out_features)
                output_size = output_end - output_start
                if output_size % 64 != 0:
                    raise ValueError("QQQ pack output chunk is not permutation aligned")

                weight_chunk = weight[output_start:output_end, input_start:input_end].transpose(0, 1)
                scale_chunk_view = scale_chunk[:, output_start:output_end]
                codes = torch.round(weight_chunk / scale_chunk_view).to(dtype=torch.int32)
                if self.group_size != self.in_features:
                    codes.add_((self.maxq + 1) // 2).clamp_(0, self.maxq)
                else:
                    codes.clamp_(-self.maxq, self.maxq)

                transformed = codes.reshape(
                    input_size // self.tile,
                    self.tile,
                    output_size // self.tile,
                    self.tile,
                ).permute(0, 2, 1, 3).reshape(
                    input_size // self.tile,
                    output_size * self.tile,
                )
                transformed = transformed.reshape(-1, perm.numel()).index_select(1, perm).reshape(
                    input_size // self.tile,
                    output_size * self.tile,
                )
                packed_chunk = torch.zeros(
                    (input_size // self.tile, output_size * 2),
                    dtype=torch.int32,
                    device=weight.device,
                )
                for lane in range(8):
                    packed_chunk.bitwise_or_(
                        (transformed[:, lane::8] & 0xF) << (4 * lane)
                    )

                packed_weight[
                    input_start // self.tile:input_end // self.tile,
                    output_start * 2:output_end * 2,
                ].copy_(packed_chunk.to(device=CPU))

        self.register_buffer("B", packed_weight)
        if self.group_size != self.in_features:
            self.register_buffer("s_group", packed_s_group)
            self.register_buffer("s_channel", packed_s_channel)
        else:
            self.register_buffer("s_channel", packed_s_channel)
        if linear.bias is not None and self.bias is not None:
            self.register_buffer("bias", linear.bias.data.to(self.bias.device).to(torch.float16))

    # activation int8 quantization
    def dynamic_quant(self, x: torch.Tensor):
        quant_scale = x.abs().max(dim=-1, keepdim=True)[0].div(127.0).to(torch.float32)
        x = (x / quant_scale).round().clamp(-128, 127).to(torch.int8)
        return x, quant_scale

    def forward(self, A):
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if A.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=A.dtype, device=A.device)

        A_dtype = A.dtype
        # qqq is float16 kernel only
        if A.dtype != torch.float16:
            A = A.to(dtype=torch.float16)

        out_shape = A.shape[:-1] + (self.out_features,)
        A = A.reshape(-1, A.shape[-1]) # .to(dtype=torch.float16)
        quant_A, s1 = self.dynamic_quant(A)
        D = torch.empty(A.shape[0], self.out_features, dtype=A.dtype, device=A.device)
        mul(
            quant_A, # A
            self.B, # B
            self.reduce_buffer, # C
            D, # D
            s1, # s1
            self.s_channel, # s2
            self.s_group, # s3
            self.workspace,
            max_par=self.max_par,
        )

        # TODO: check if we should reshape at end
        D = D.reshape(out_shape)

        if self.bias is not None:
            D.add_(self.bias)

        if self.adapter:
            D = self.adapter.apply(x=A, out=D)

        return D.to(dtype=A_dtype)


class QQQTorchLinear(QQQLinear):
    SUPPORTS_BACKENDS = [BACKEND.QQQ_TORCH]
    SUPPORTS_METHODS = [METHOD.QQQ]
    SUPPORTS_FORMATS = {FORMAT.QQQ: 90}
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_DEVICES = [DEVICE.ALL]
    SUPPORTS_PLATFORM = [PLATFORM.ALL]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]
    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    REQUIRES_FORMAT_V2 = False
    QUANT_TYPE = "qqq"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("backend", BACKEND.QQQ_TORCH)
        super().__init__(*args, **kwargs)

    @classmethod
    def validate_once(cls) -> Tuple[bool, Optional[Exception]]:
        return True, None

    def _unpack_weight_codes(self) -> torch.Tensor:
        unpacked = _unpack_uint4(self.B)
        perm_i = self._perm_i.to(device=unpacked.device)
        unpacked = unpacked.reshape(-1, self._perm_i.numel())[:, perm_i].reshape(
            self.in_features // self.tile,
            self.out_features * self.tile,
        )
        return (
            unpacked.reshape(
                self.in_features // self.tile,
                self.out_features // self.tile,
                self.tile,
                self.tile,
            )
            .permute(0, 2, 1, 3)
            .reshape(self.in_features, self.out_features)
        )

    def _unpermute_scales(self, scales: torch.Tensor, inverse_perm: torch.Tensor) -> torch.Tensor:
        inverse_perm = inverse_perm.to(device=scales.device)
        return scales.reshape(-1, inverse_perm.numel())[:, inverse_perm].reshape(scales.shape)

    def _dequantize_weight_for_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
        codes = self._unpack_weight_codes()
        s_channel = self._unpermute_scales(self.s_channel, self._scale_perm_single_i).to(torch.float32)

        if self.group_size != self.in_features:
            s_group = self._unpermute_scales(self.s_group, self._scale_perm_i).to(torch.float32)
            group_idx = torch.floor_divide(
                torch.arange(self.in_features, dtype=torch.int64, device=codes.device),
                self.group_size,
            )
            weight = (codes.to(torch.float32) - 8.0) * s_group[group_idx]
            weight = weight.round().clamp(-128, 127)
        else:
            signed = torch.where(codes >= 8, codes - 16, codes)
            weight = signed.to(torch.float32) * 16.0

        return weight, s_channel

    def forward(self, A):
        if A.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=A.dtype, device=A.device)

        A_dtype = A.dtype
        if A.dtype != torch.float16:
            A = A.to(dtype=torch.float16)

        out_shape = A.shape[:-1] + (self.out_features,)
        A = A.reshape(-1, A.shape[-1])
        quant_A, s1 = self.dynamic_quant(A)
        weight, s_channel = self._dequantize_weight_for_torch()
        accum = torch.matmul(quant_A.to(torch.float32), weight.to(device=A.device, dtype=torch.float32))
        accum = accum * s1
        accum = accum * s_channel.to(device=A.device, dtype=torch.float32)
        D = accum.to(dtype=A.dtype).reshape(out_shape)

        if self.bias is not None:
            bias = self.bias
            if bias.device != D.device or bias.dtype != D.dtype:
                bias = bias.to(device=D.device, dtype=D.dtype)
            D.add_(bias)

        if self.adapter:
            D = self.adapter.apply(x=A, out=D)

        return D.to(dtype=A_dtype)


__all__ = ["QQQLinear", "QQQTorchLinear"]
