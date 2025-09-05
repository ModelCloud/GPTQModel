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


import torch

from ...adapter.adapter import Adapter, Lora
from ...models._const import DEVICE, PLATFORM
from ...nn_modules.qlinear import PackableQuantLinear
from ...utils.backend import BACKEND
from ...utils.logger import setup_logger
from packaging import version

log = setup_logger()

try:
    # TODO: triton is not compatible with free threading
    # if not has_gil_disabled():
    #     raise Exception("GIL is disabled so Triton is not (yet) compatible.")

    import triton
    import triton.language as tl
    from triton import __version__ as triton_version

    if version.parse(triton_version) < version.parse("2.0.0"):
        raise ImportError(f"triton version must be >= 2.0.0: actual = {triton_version}")
    TRITON_AVAILABLE = True
except BaseException:
    TRITON_AVAILABLE = False

TRITON_INSTALL_HINT = "Trying to use the triton backend, but it could not be imported. Please install triton by 'pip install gptqmodel[triton] --no-build-isolation'"
TRITON_XPU_INSTALL_HINT = "Trying to use the triton backend and xpu device, but it could not be imported. Please install triton by [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)"

class TritonA100QuantLinear(PackableQuantLinear):
    SUPPORTS_BITS = [2, 4, 8]
    SUPPORTS_GROUP_SIZE = [-1, 16, 32, 64, 128]
    SUPPORTS_DESC_ACT = [True, False]
    SUPPORTS_SYM = [True, False]
    SUPPORTS_SHARDS = True
    SUPPORTS_TRAINING = True
    SUPPORTS_AUTO_PADDING = True
    SUPPORTS_IN_FEATURES_DIVISIBLE_BY = [32]
    SUPPORTS_OUT_FEATURES_DIVISIBLE_BY = [32]

    SUPPORTS_DEVICES = [DEVICE.CUDA] # Intel XPU can use Triton but this has been validated
    SUPPORTS_PLATFORM = [PLATFORM.LINUX, PLATFORM.WIN32]
    SUPPORTS_PACK_DTYPES = [torch.int32, torch.int16, torch.int8]
    SUPPORTS_ADAPTERS = [Lora]

    SUPPORTS_DTYPES = [torch.float16, torch.bfloat16]

    # for transformers/optimum tests compat
    QUANT_TYPE = "triton_a100"

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
        super().__init__(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            pack_dtype=pack_dtype,
            backend=kwargs.pop("backend", BACKEND.TORCH),
            adapter=adapter,
            register_buffers=register_buffers,
            **kwargs)

        self.dequant_dtype = torch.int16 if self.bits == 8 else torch.int8

        # if self.group_size != self.in_features:
        #     self.padded_infeatures = self.in_features + (-self.in_features % self.group_size)
        # else:
        #     self.padded_infeatures = self.in_features

    def post_init(self):
        # if self.padded_infeatures != self.in_features:
        #     self.qweight.resize_(self.padded_infeatures // self.pack_dtype_bits * self.bits, self.out_features)
        #     self.qzeros.resize_(
        #         math.ceil(self.padded_infeatures / self.group_size),
        #         self.out_features // self.pack_dtype_bits * self.bits
        #     )
        #     self.scales.resize_((math.ceil(self.padded_infeatures / self.group_size), self.out_features), )
        #     self.g_idx = torch.tensor([i // self.group_size for i in range(self.padded_infeatures)], dtype=torch.int32,
        #                               device=self.g_idx.device)

        super().post_init()

        # torch benefits the most from torch.compile, enable it by default
        self.optimize()

    def optimize(self, backend: str = None, mode: str = None, fullgraph: bool = False):
        if self.optimized:
            return

        if backend is None:
            # MPS doesn't support inductor.
            backend = "inductor" if self.list_buffers()[0].device.type != "mps" else "aot_eager"

        # # compile dequantize
        # self.dequantize_weight = torch_compile(self.dequantize_weight, backend=backend, mode=mode, fullgraph=fullgraph)

        if self.adapter:
            self.adapter.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

        super().optimize()

    def forward(self, x: torch.Tensor):
        if self.training:
            return super().forward(x)

        out_shape = x.shape[:-1] + (self.out_features,)

        block_size_m = x.shape[0]
        # TODO test a100_qlinear
        out = a100_qlinear.apply(
            x.reshape(-1, x.shape[-1]),
            self.qweight,
            self.scales,
            self.qzeros,
            self.group_size,
        ).reshape(out_shape)

        if self.bias is not None:
            out.add_(self.bias)

        if self.adapter:
            out = self.adapter.apply(x=x, out=out)

        return out.to(dtype=x.dtype)

__all__ = ["TritonA100QuantLinear"]

@triton.jit()
def _a100_quantized_matmul(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                           stride_am, stride_ak,
                           stride_bk, stride_bn,
                           stride_cm, stride_cn,
                           stride_scales_g, stride_scales_n,
                           stride_zeros_g, stride_zeros_n,
                           groupsize,
                           m, n, k,
                           block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
                           group_size_m: tl.constexpr,
                           ):
    pid = tl.program_id(0)

    total_blocks_m = tl.cdiv(m, block_size_m)
    total_blocks_n = tl.cdiv(n, block_size_n)
    total_blocks_k = tl.cdiv(k, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_m = (pid_m * block_size_m + tl.arange(0, block_size_m)) % m
    offs_n = (pid_n * block_size_n + tl.arange(0, block_size_n)) % n

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_size_m), block_size_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_size_n), block_size_n)
    offs_k = tl.arange(0, block_size_k)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    output = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF  # b -> int32
        b = b * scales[None, :] - zeros[None, :]  # b -> fp16

        output += tl.dot(a, b)
        a_ptrs += stride_ak * block_size_k
        b_ptrs += (block_size_k // 8) * stride_bk

    output.to(tl.float16)
    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, output)


class a100_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros, group_size):
        m, k = a.shape
        _, n = b.shape

        # quant_groupsize = 128
        quant_groupsize = group_size
        block_size_m = 16
        block_size_n = 32  # [N = 4096 // 32] = 128 blocks
        block_size_k = 256
        group_size_m = 8
        num_warps = 4
        num_stages = 4
        total_blocks_m = triton.cdiv(m, block_size_m)
        total_blocks_n = triton.cdiv(n, block_size_n)
        total_programs = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)

        c = torch.zeros((m, n), device=b.device, dtype=torch.float16)
        k = _a100_quantized_matmul[grid](
            a, b, c, scales, zeros,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            quant_groupsize,
            m, n, k,
            block_size_m, block_size_n, block_size_k, group_size_m,
            num_warps=num_warps, num_stages=num_stages,
        )

        # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared / 1000} kB shared memory\n")
        #
        # with open('dequant_simple.txt', 'w') as f:
        #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared / 1000} kB shared memory\n", file=f)
        #     print("IR", k.asm['ttir'], file=f)
        #     print("TTGIR", k.asm['ttgir'], file=f)
        #     print("PTX", k.asm['ptx'], file=f)
        #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared / 1000} kB shared memory\n", file=f)
        #
        #     print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c

