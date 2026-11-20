# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Portions of this file are adapted from turboderp-org/exllamav3.
# Credits: TurboDerp / ExLlamaV3 contributors.

from __future__ import annotations

import torch

from ....utils.kernel import KernelConfig
from ...ext import exllamav3_ext as ext
from ...trellis import trellis_rate
from .exl3_lib.quantize import (
    had_k,
    had_n,
    preapply_had_l,
    preapply_had_r,
    reconstruct_q4nx_inner,
    reconstruct_r4nx_inner,
)


class LinearEXL3:

    quant_type: str = "exl3"
    q4nx: bool = False
    q4nx_tensor: torch.Tensor | None = None
    r4nx: bool = False
    r4nx_tensor: torch.Tensor | None = None

    def __init__(
        self,
        config: object | None,
        in_features: int,
        out_features: int,
        scale: torch.Tensor | None = None,
        su: torch.Tensor | None = None,
        sv: torch.Tensor | None = None,
        suh: torch.Tensor | None = None,
        svh: torch.Tensor | None = None,
        trellis: torch.Tensor | None = None,
        mcg: torch.Tensor | None = None,
        mul1: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        transformers_fix: bool = False,
        key: str | None = None,
        kernel: KernelConfig | None = None,
        *,
        q4nx: torch.Tensor | None = None,
        r4nx: torch.Tensor | None = None,
    ):
        assert scale is None, "scale is no longer used"
        assert su is not None or suh is not None, "either su (packed) or suh (unpacked) is required"
        assert sv is not None or svh is not None, "either sv (packed) or svh (unpacked) is required"
        assert trellis is not None, "trellis is required"
        if su is not None: assert su.dtype == torch.int16, "su is wrong datatype"
        if sv is not None: assert sv.dtype == torch.int16, "sv is wrong datatype"
        if suh is not None: assert suh.dtype == torch.half, "suh is wrong datatype"
        if svh is not None: assert svh.dtype == torch.half, "svh is wrong datatype"
        assert trellis.dtype == torch.int16, "trellis is wrong datatype"
        assert len(trellis.shape) == 3, "trellis must have dim = 3"

        self.transformers_fix = transformers_fix
        self.key = key

        # self.scale = scale.item()
        self.su = None
        self.sv = None
        self.suh = suh if suh is not None else self.unpack_bf(su)
        self.svh = svh if svh is not None else self.unpack_bf(sv)
        self.trellis = trellis
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.swap_device = None
        self.out_dtype = out_dtype

        self.mcg_tensor = mcg
        self.mul1_tensor = mul1
        self.q4nx_tensor = q4nx
        self.r4nx_tensor = r4nx
        self.mcg = self.mcg_tensor is not None
        self.mul1 = self.mul1_tensor is not None
        self.q4nx = self.q4nx_tensor is not None
        self.r4nx = self.r4nx_tensor is not None

        self.K = trellis_rate(
            trellis.shape[-1], self.mcg, self.mul1, q4nx=self.q4nx, r4nx=self.r4nx
        )

        self.bc = ext.BC_LinearEXL3(
            self.trellis,
            self.suh,
            self.svh,
            self.K,
            self.bias,
            self.mcg,
            self.mul1,
            **({"kernel": kernel} if kernel is not None else {}),
            **({"q4nx": True} if self.q4nx else {}),
            **({"r4nx": True} if self.r4nx else {}),
        )


    def unload(self):
        # The bsz == 1 Hadamard scratch is allocated per forward, so a module
        # keeps no shared inference buffer to release (#4619).
        pass


    def get_tensors(self, key: str):
        return {
            f"{key}.{subkey}": tensor.contiguous()
            for subkey, tensor in [
                ("su", self.su),
                ("sv", self.sv),
                ("suh", self.suh),
                ("svh", self.svh),
                ("trellis", self.trellis),
                ("bias", self.bias),
                ("mcg", self.mcg_tensor),
                ("mul1", self.mul1_tensor),
                ("q4nx", self.q4nx_tensor),
                ("r4nx", self.r4nx_tensor),
            ] if tensor is not None
        }


    def forward(
        self,
        x: torch.Tensor,
        params: dict,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if "ovr" in params:
            ovr = params["ovr"]
            if self.key in ovr and ovr[self.key].inner is not self:
                return ovr[self.key].forward(x, params, out_dtype)

        bsz = x.numel() // x.shape[-1]
        # The small-batch tensor-core GEMM (`bc.run`) is NVIDIA-only. On ROCm the
        # native HIP GEMV handles decode-sized batches when it accepts the shape,
        # codebook and device; larger batches (and anything it declines) take the
        # reconstruct + hgemm (hipBLAS) path.
        if torch.version.hip is not None:
            torch_mode = params.get("reconstruct", bsz > 8)
        else:
            # R4NX has no tensor-core GEMM kernel on any backend, so it always
            # decodes through reconstruct + HGEMM.
            torch_mode = self.r4nx or params.get("reconstruct", bsz > 32 and not self.q4nx)

        out_shape = x.shape[:-1] + (self.out_features,)
        result_dtype = out_dtype or self.out_dtype or torch.half
        if bsz == 0:
            # A zero-row batch has no Hadamard/HGEMM work. The native
            # had_r_128 would otherwise launch a zero-row grid, which the
            # CUDA/HIP runtime rejects.
            return torch.empty(out_shape, dtype=result_dtype, device=x.device)
        # The native hadamard/hgemm/bc kernels treat the operand as densely
        # packed rows and take `data_ptr()` directly. `view` succeeds on some
        # strided inputs (column slices, `hidden_states[:, -1:, :]`), which
        # then produced silently wrong rows, so pack before the kernels.
        x = x.reshape(-1, self.in_features).contiguous()
        include_bias = params.get("include_bias", True)

        if torch.version.hip is not None and not torch_mode:
            y = torch.empty(out_shape, dtype=result_dtype, device=x.device)
            if ext.bc_linear_exl3_run_hip_gemv(
                self.trellis,
                self.suh,
                self.svh,
                float(self.K),
                self.bias if include_bias else None,
                self.mcg,
                self.mul1,
                x,
                y,
                self.q4nx,
                self.bc.int8_gemv,
            ):
                return y
            # Shape, device or codebook not eligible: reconstruct + hgemm.
            torch_mode = True

        if torch_mode:
            y = torch.empty((x.shape[0], self.out_features), dtype=torch.float32, device=x.device)
            xh = torch.empty_like(x)
            ext.had_r_128(x, xh, self.suh, None, 1.0)
            w = self.get_inner_weight_tensor()
            ext.hgemm(xh, w, y)
            ext.had_r_128(y, y, None, self.svh, 1.0)
            if include_bias and self.bias is not None:
                y.add_(self.bias)
            return y.view(out_shape).to(result_dtype)

        y = torch.empty(out_shape, dtype=result_dtype, device=x.device)
        if include_bias:
            self.bc.run(x, y)
        else:
            self.bc.run(x, y, include_bias=False)

        return y


    def unpack_bf(self, bitfield: torch.Tensor):
        # For some reason this operation causes a GPU assert on Transformers. Running on CPU seems to fix it
        device = bitfield.device
        if self.transformers_fix:
            bitfield = bitfield.cpu()

        # TODO: Maybe custom kernel for this. Only used for full reconstruct and loading old models, not during inference
        bitfield = bitfield.view(torch.uint16).to(torch.int)
        masks = (1 << torch.arange(16)).to(bitfield.device)
        expanded = (bitfield.unsqueeze(-1) & masks) > 0
        expanded = expanded.flatten()
        expanded = torch.where(expanded, torch.tensor(-1.0, dtype = torch.float16), torch.tensor(1.0, dtype = torch.float16))
        return expanded.contiguous().to(device)


    def get_weight_tensor(self):
        # suh = self.unpack_bf(self.su).unsqueeze(1)
        suh = self.unpack_bf(self.su).unsqueeze(1) if self.su else self.suh.unsqueeze(1)
        svh = self.unpack_bf(self.sv).unsqueeze(0) if self.sv else self.svh.unsqueeze(0)
        if self.r4nx:
            w = reconstruct_r4nx_inner(self.trellis)
        elif self.q4nx:
            w = reconstruct_q4nx_inner(self.trellis)
        else:
            w = self.get_inner_weight_tensor()
        w = preapply_had_l(w, had_k)
        w *= suh
        w = preapply_had_r(w, had_n)
        w *= svh
        # w *= self.scale
        return w.half() if (self.q4nx or self.r4nx) else w


    def get_inner_weight_tensor(self):
        if self.r4nx:
            return reconstruct_r4nx_inner(self.trellis).half()
        if self.q4nx:
            return reconstruct_q4nx_inner(self.trellis).half()
        w = torch.empty((self.in_features, self.out_features), dtype = torch.half, device = self.trellis.device)
        ext.reconstruct(w, self.trellis, self.K, self.mcg, self.mul1)
        return w


    def get_bias_tensor(self) -> torch.Tensor | None:
        return self.bias


    # Swap tensors to CPU (to free some space while quantizing)
    def swap_cpu(self):
        if self.swap_device is not None:
            return
        self.swap_device = self.trellis.device
        if self.su is not None: self.su = self.su.cpu()
        if self.sv is not None: self.sv = self.sv.cpu()
        if self.suh is not None: self.suh = self.suh.cpu()
        if self.svh is not None: self.svh = self.svh.cpu()
        if self.trellis is not None: self.trellis = self.trellis.cpu()
        if self.bias is not None: self.bias = self.bias.cpu()


    def unswap_cpu(self):
        if self.swap_device is None:
            return
        if self.su is not None: self.su = self.su.to(self.swap_device)
        if self.sv is not None: self.sv = self.sv.to(self.swap_device)
        if self.suh is not None: self.suh = self.suh.to(self.swap_device)
        if self.svh is not None: self.svh = self.svh.to(self.swap_device)
        if self.trellis is not None: self.trellis = self.trellis.to(self.swap_device)
        if self.bias is not None: self.bias = self.bias.to(self.swap_device)
        self.swap_device = None


    def tp_export(self, plan, producer):
        return {
            "cls": LinearEXL3,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "suh": producer.send(self.suh),
            "svh": producer.send(self.svh),
            "trellis": producer.send(self.trellis),
            "bias": producer.send(self.bias),
            "mcg": producer.send(self.mcg_tensor),
            "mul1": producer.send(self.mul1_tensor),
            "q4nx": producer.send(self.q4nx_tensor),
            "r4nx": producer.send(self.r4nx_tensor),
            "out_dtype": self.out_dtype,
        }


    @staticmethod
    def tp_import_split(local_context, exported, plan, split):
        consumer = local_context["consumer"]
        id_suh = exported["suh"]
        id_svh = exported["svh"]
        id_trellis = exported["trellis"]
        id_bias = exported["bias"]
        mcg = consumer.recv(exported["mcg"], cuda = True)
        mul1 = consumer.recv(exported["mul1"], cuda = True)
        q4nx = consumer.recv(exported["q4nx"], cuda = True) if "q4nx" in exported else None
        r4nx = consumer.recv(exported["r4nx"], cuda = True) if "r4nx" in exported else None

        if split is not None:
            split_out, first, last = split
        else:
            split_out, first, last = True, 0, exported["out_features"]

        if split_out:
            suh = consumer.recv(id_suh, cuda = True)
            svh = consumer.recv(id_svh, cuda = True, slice_dim = 0, first = first, last = last)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 1, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True, slice_dim = 0, first = first, last = last)
            in_features = exported["in_features"]
            out_features = last - first
        else:
            suh = consumer.recv(id_suh, cuda = True, slice_dim = 0, first = first, last = last)
            svh = consumer.recv(id_svh, cuda = True)
            trellis = consumer.recv(id_trellis, cuda = True, slice_dim = 0, first = first // 16, last = last // 16)
            bias = consumer.recv(id_bias, cuda = True)
            in_features = last - first
            out_features = exported["out_features"]

        module = LinearEXL3(
            config = None,
            in_features = in_features,
            out_features = out_features,
            scale = None,
            su = None,
            sv = None,
            suh = suh,
            svh = svh,
            trellis = trellis,
            mcg = mcg,
            mul1 = mul1,
            q4nx = q4nx,
            r4nx = r4nx,
            bias = bias,
            out_dtype = exported["out_dtype"],
        )
        return module
