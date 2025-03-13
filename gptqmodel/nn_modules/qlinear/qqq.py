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

qqq_import_exception = None
try:
    import gptqmodel_qqq_kernels
except ImportError as e:
    qqq_import_exception = e


from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import BaseQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from ...utils.rocm import IS_ROCM

qqq_import_exception = None
try:
    import gptqmodel_qqq_kernels
except ImportError as e:
    qqq_import_exception = e

log = setup_logger()

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
    gptqmodel_qqq_kernels.qqq_gemm(A, B, C, D, s1, s2, s3, workspace, thread_k, thread_n, sms, max_par)


class QQQQuantLinear(BaseQuantLinear):
    SUPPORTS_BITS = [4]
    SUPPORTS_GROUP_SIZE = [-1, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = False
    SUPPORTS_AUTO_PADDING = False
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [64]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [64]

    SUPPORTS_DEVICES = [DEVICE.CUDA]
    SUPPORTS_PLATFORM = [PLATFORM.LINUX]
    SUPPORTS_PACK_DTYPES = [torch.int32]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

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
        **kwargs):
        if qqq_import_exception is not None:
            raise ValueError(
                f"Trying to use the qqq backend, but could not import the C++/CUDA dependencies with the following error: {qqq_import_exception}"
            )

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
            register_buffers=False,
            **kwargs)

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
        self.wf = torch.tensor(list(range(0, 32, 4)), dtype=torch.int32).unsqueeze(0)
        if bias:
            self.register_buffer("bias", torch.zeros((self.out_features), dtype=torch.float16))
        else:
            self.bias = None
        self._perm, self._scale_perm, self._scale_perm_single = self._get_perms()

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
        scale_perm = []
        for i in range(8):
            scale_perm.extend([i + 8 * j for j in range(8)])
        scale_perm_single = []
        for i in range(4):
            scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
        return perm, scale_perm, scale_perm_single

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
        if qqq_import_exception is not None:
            return False, qqq_import_exception

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

    #def pack(self, linear: nn.Module, scales: t.Tensor, zeros: t.Tensor, g_idx: t.Tensor = None):
    def pack(self, linear: torch.nn.Module, scales: torch.Tensor, s_extra=None):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.float16`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        @s_extra: corresponding quantization scales of shape `(1, outfeatures)`
        """
        if self.group_size != self.in_features:
            assert s_extra is not None, "s_extra is needed"
        if linear.weight.dtype != torch.float16:
            log.warn.once(
                f"""The dtype of weights is {linear.weight.dtype}, while our w4a8 GEMM's output is torch.float16.
                If you can ensure your GEMM results don't overflow torch.float16, it will still function correctly.
                Otherwise, it will yield incorrect results."""
            )
        s = scales.t()
        w = linear.weight.data.t()
        if self.group_size != self.in_features:
            w = w.reshape((-1, self.group_size, self.out_features))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.group_size, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        if self.group_size != self.in_features:
            w += (self.maxq + 1) // 2
            w = torch.clamp(w, 0, self.maxq)
        else:
            w = torch.clamp(w, -self.maxq, self.maxq)
        if self.group_size != self.in_features:
            s_extra = s_extra.reshape(1, -1).to(dtype=torch.float32)
            s = (s.reshape(-1, self.out_features) / s_extra).to(dtype=torch.float16)

            w = w.reshape((self.group_size, -1, self.out_features))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.in_features, self.out_features)).contiguous()
            s = s.reshape((-1, len(self._scale_perm)))[:, self._scale_perm]
            s_extra = s_extra.reshape((-1, len(self._scale_perm_single)))[
                      :, self._scale_perm_single
                      ]
            s_extra = s_extra.reshape((-1, self.out_features)).contiguous()
        else:
            # NOTE(zhangying): div 2 ** (8 - self.bits)) to deal with right_shift in unpacking
            s = (
                (s / (2 ** (8 - self.bits)))
                .reshape((-1, len(self._scale_perm_single)))[:, self._scale_perm_single]
                .to(dtype=torch.float32)
            )
        s = s.reshape((-1, self.out_features)).contiguous()
        w = w.reshape(
            (
                self.in_features // self.tile,
                self.tile,
                self.out_features // self.tile,
                self.tile,
            )
        )
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.in_features // self.tile, self.out_features * self.tile))
        res = w
        res = res.reshape((-1, self._perm.numel()))[:, self._perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        if self.group_size != self.in_features:
            for i in range(8):
                q |= res[:, i::8] << 4 * i
        else:
            for i in range(8):
                q |= (res[:, i::8] & 0xF) << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        if self.group_size != self.in_features:
            self.s_group[:, :] = s.to(self.s_group.device)
            self.s_channel[:, :] = s_extra.to(self.s_channel.device)
        else:
            self.s_group = torch.tensor(
                [], dtype=torch.float16, device=self.s_channel.device
            )
            self.s_channel[:, :] = s.to(self.s_channel.device)
        if linear.bias is not None:
            if self.bias is not None:
                self.bias[:] = linear.bias.data.to(self.bias.device).to(torch.float16)
            else:
                self.bias = linear.bias.clone().to(torch.float16)

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


__all__ = ["QQQQuantLinear"]
