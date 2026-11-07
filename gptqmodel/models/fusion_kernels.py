# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Explicit native storage contracts for projection fusion.

These contracts describe output axes in finalized packed layouts. No backend is
selected or substituted here; an unknown exact kernel type remains ineligible.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class FusionKernelSpec:
    layout: str
    gptq: bool = False
    fields: tuple[str, ...] = ()
    zero_name: str | None = "qzeros"
    private_fields: tuple[str, ...] = ()
    weight_name: str = "qweight"
    scale_name: str = "scales"

    @property
    def private_mappings(self) -> tuple[str, ...]:
        """Names of private mappings present in this layout; not a fixed record."""
        mappings = []
        if self.layout == "exllamav2":
            mappings.append("q_tensors")
        return tuple(mappings)

    @property
    def cpu(self):
        return self.layout in ("cpu_aten", "cpu_avx512", "cpu_legacy")

    @property
    def extra_tensors(self):
        names = ("scales_and_zeros",) if self.layout in ("cpu_aten", "cpu_legacy") else ()
        if self.layout == "exllamav2":
            names = ("_fusion_q_perm", "_fusion_q_invperm")
        return tuple(dict.fromkeys((*names, self.weight_name, self.scale_name,
                                    *((self.zero_name,) if self.zero_name is not None else ()))))

    @property
    def axes(self) -> tuple[tuple[str, int], ...]:
        """Variable-length operand/axis collection; absent zero points stay absent."""
        if self.layout in ("allspark", "int8"):
            axes = [(self.weight_name, 0), (self.scale_name, 0)]
            if self.zero_name is not None:
                axes.append((self.zero_name, 0))
        elif self.cpu:
            axes = [("qweight", 0), ("qzeros", 1), ("scales", 1)]
        elif self.layout == "awq_gemv":
            axes = [("qweight", 0), ("qzeros", 0), ("scales", 0)]
        elif self.layout == "awq_fast":
            axes = [("qweight", 0), (self.zero_name, 1), ("scales", 1)]
        else:
            axes = [("qweight", 1), ("qzeros", 1), ("scales", 1)]
        axes.append(("bias", 0))
        return tuple(axes)

    def input_dtype(self, module):
        if self.layout == "cpu_avx512":
            return torch.bfloat16
        return getattr(module, "_cpu_transform_dtype", None) or getattr(module, self.scale_name).dtype

    def prepare(self, modules, dtype=None):
        if self.layout == "exllamav2":
            for module in modules:
                metadata = getattr(module, "q_tensors", None) or {}
                module._fusion_q_perm = metadata.get("q_perm")
                module._fusion_q_invperm = metadata.get("q_invperm")
        if not self.cpu:
            return
        for module in modules:
            if self.layout == "cpu_legacy":
                from ..looper.linear_mode import LinearMode

                if module.linear_mode is None:
                    module.transform_cpu(dtype or module.scales.dtype)
                module.linear_mode = LinearMode.INFERENCE
                self.rebind(module)
                continue
            if not getattr(module, "_cpu_transformed", False):
                target = dtype or module.scales.dtype
                if self.layout == "cpu_avx512" and not self.gptq:
                    module.transform_cpu()
                else:
                    module.transform_cpu(dtype=target)
            module.linear_mode = "inference"

    def shape_error(self, module):
        k, n = module.in_features, module.out_features
        groups = (k + module.group_size - 1) // module.group_size
        if self.cpu:
            if module.qweight.device.type != "cpu":
                return "CPU int4pack fusion requires operands materialized on CPU"
            prepared = (getattr(module.linear_mode, "value", None) == "inference"
                        if self.layout == "cpu_legacy" else getattr(module, "_cpu_transformed", False))
            if self.layout == "cpu_legacy" and not module.sym:
                return "legacy CPU int4pack fusion requires symmetric weights"
            if not prepared:
                return ""  # prepare() finalizes this source before binding its native views.
            if n % 32 or k % module.group_size or tuple(module.qweight.shape) != (n, k // 2):
                return "CPU fusion requires complete N=32 tiles in finalized int4pack storage"
            if tuple(module.scales.shape) != (groups, n) or tuple(module.qzeros.shape) != (groups, n):
                return "CPU fusion scale/zero shapes do not match finalized int4pack storage"
            if self.layout in ("cpu_aten", "cpu_legacy") and tuple(module.scales_and_zeros.shape) != (groups, n, 2):
                return "CPU fusion affine metadata shape differs from the native int4pack ABI"
        elif self.layout == "awq_gemv":
            if tuple(module.qweight.shape) != (n, k // 8) or module.scales.shape[0] != n or module.qzeros.shape[0] != n:
                return "AWQ GEMV fusion requires output rows in its native packed layout"
        elif self.layout == "awq_fast":
            if k % 64 or n % 4 or tuple(module.qweight.shape) != (n // 4, k):
                return "AWQ fast fusion requires complete interleaved N=4, K=64 tiles"
            if module.scales.shape[-1] != n or getattr(module, self.zero_name).shape != module.scales.shape:
                return "AWQ fast scale/zero metadata does not match the declared output width"
        elif self.layout == "exllamav2":
            if not module.q_handle or k % 32 or n % 32 or tuple(module.qweight.shape) != (k // 8, n):
                return "ExLlamaV2 fusion requires finalized native matrix descriptors and complete tiles"
            if module.desc_act:
                return "ExLlamaV2 fusion currently requires canonical input order"
            if tuple(module.qzeros.shape) != (groups, n // 8) or tuple(module.scales.shape) != (groups, n):
                return "ExLlamaV2 native group metadata shape differs"
        elif self.layout == "allspark":
            if (not module._repacked or n % 32 or
                    tuple(module.qweight_packed.shape) != (n, k) or module.scales_packed.numel() != n):
                return "AllSpark fusion requires finalized unpadded native N32 tiles"
            if self.zero_name is not None and module.zeros_packed.numel() != n:
                return "AllSpark native zero-point layout differs"
        elif self.layout == "int8":
            if tuple(module.int8_weight_nk.shape) != (n, k) or module.int8_channel_scale.numel() != n:
                return "CPU INT8 fusion requires native N-by-K codes and per-channel scales"
        return ""

    def combined_shape_error(self, modules, dtype=None):
        first = modules[0]
        ok, error = type(first).validate(
            bits=first.bits, group_size=first.requested_group_size, sym=first.sym,
            desc_act=first.desc_act, pack_dtype=first.pack_dtype, dtype=dtype,
            device=getattr(first, self.weight_name).device, in_features=first.in_features,
            out_features=sum(module.out_features for module in modules),
        )
        return "" if ok else str(error)

    def combine(self, modules):
        first = modules[0]
        fused = copy.copy(first)
        fused._parameters = first._parameters.copy()
        fused._buffers = first._buffers.copy()
        fused._modules = first._modules.copy()
        fused._forward_hooks = first._forward_hooks.copy()
        fused._forward_pre_hooks = first._forward_pre_hooks.copy()
        fused.__dict__.pop("forward", None)
        fused.__dict__.pop("clear_weight_cache", None)
        # A compiled bound method copied from a source would decode that
        # source's width and operands rather than the combined native matrix.
        fused.__dict__.pop("dequantize_weight", None)
        if hasattr(first, "_weight_metadata"):
            fused._weight_metadata = type(first._weight_metadata)(fused)
        fused.out_features = sum(module.out_features for module in modules)
        for name, axis in self.axes:
            tensor = getattr(first, name, None)
            if tensor is None:
                continue
            combined = torch.cat([getattr(module, name) for module in modules], dim=axis)
            if name in fused._parameters:
                combined = torch.nn.Parameter(combined, requires_grad=False)
            setattr(fused, name, combined)
        if self.layout == "exllamav2":
            from ..utils.exllamav2 import exllamav2_clone_q_matrix

            fused.q_tensors = dict(getattr(first, "q_tensors", None) or {})
            fused.q_tensors.update({name: getattr(fused, name) for name in ("qweight", "qzeros", "scales")})
            fused.q_handle, _ = exllamav2_clone_q_matrix(fused, awq=not self.gptq)
            fused._fusion_native_view = False
        if self.layout in ("cpu_aten", "cpu_legacy"):
            fused.scales_and_zeros = torch.cat([module.scales_and_zeros for module in modules], dim=1)
            fused._fused_state_key = None
        if self.cpu:
            # g_idx/ret_idx are input-channel metadata and remain shared.
            fused.linear_mode = first.linear_mode
        clear = getattr(fused, "clear_weight_cache", None)
        if clear is not None:
            clear()
        self.rebind(fused)
        return fused

    def rebind(self, module):
        if self.layout == "gptq_fp8":
            module.post_init()
            return
        if self.layout == "int8":
            from ..nn_modules.qlinear.torch_int8 import Int8PackedModule

            module.int8_module = Int8PackedModule(module.int8_weight_nk, module.int8_channel_scale).eval()
            return
        if self.layout == "exllamav2":
            metadata = getattr(module, "q_tensors", None)
            if metadata is not None:
                module.q_tensors = {**metadata, **{name: getattr(module, name) for name in ("qweight", "qzeros", "scales")}}
            if module.qweight.is_contiguous():
                from ..utils.exllamav2 import exllamav2_clone_q_matrix, exllamav2_free_q_matrix

                old = module.q_handle
                module.q_handle, _ = exllamav2_clone_q_matrix(module, awq=not self.gptq)
                exllamav2_free_q_matrix(old, awq=not self.gptq)
                module._fusion_native_view = False
            else:
                module._fusion_native_view = True
            return
        if self.layout != "cpu_legacy":
            return
        from ..nn_modules.qlinear.torch_fused import Int4PackedOp, pack_scales_and_zeros

        current = getattr(module, "scales_and_zeros", None)
        dtype = torch.promote_types(module.scales.dtype, module.qzeros.dtype)
        if current is not None:
            dtype = torch.promote_types(dtype, current.dtype)
        module.scales_and_zeros = pack_scales_and_zeros(module.scales.to(dtype), module.qzeros.to(dtype))
        module.torch_fused_op = Int4PackedOp(module.qweight, module.scales_and_zeros, module.group_size).eval()

    def release(self, module):
        if self.layout == "exllamav2":
            from ..utils.exllamav2 import exllamav2_free_q_matrix

            exllamav2_free_q_matrix(module.q_handle, awq=not self.gptq)
            module.q_handle = None
            module.q_tensors = None
        for name in self.extra_tensors:
            module.__dict__.pop(name, None)


@lru_cache(maxsize=1)
def kernel_specs():
    from ..nn_modules.qlinear.allspark import AllSparkLinear
    from ..nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
    from ..nn_modules.qlinear.exllamav2 import ExllamaV2Linear
    from ..nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
    from ..nn_modules.qlinear.gemm_awq import AwqGEMMLinear
    from ..nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
    from ..nn_modules.qlinear.gemv_awq import AwqGEMVLinear, AwqGEMVTritonLinear
    from ..nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
    from ..nn_modules.qlinear.marlin import MarlinLinear
    from ..nn_modules.qlinear.marlin_awq import AwqMarlinLinear
    from ..nn_modules.qlinear.torch import TorchLinear
    from ..nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
    from ..nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
    from ..nn_modules.qlinear.torch_awq import AwqTorchLinear
    from ..nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
    from ..nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
    from ..nn_modules.qlinear.torch_fused import TorchFusedLinear
    from ..nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
    from ..nn_modules.qlinear.torch_int8 import TorchInt8Linear
    from ..nn_modules.qlinear.tritonv2 import TritonV2Linear
    from ..nn_modules.qlinear.w4a_floatx import W4AFP8Linear

    return {
        TorchLinear: FusionKernelSpec("gptq", gptq=True),
        TritonV2Linear: FusionKernelSpec("gptq", gptq=True),
        W4AFP8Linear: FusionKernelSpec("gptq_fp8", gptq=True, fields=("_require_activation_stream",)),
        AwqTorchLinear: FusionKernelSpec("awq"),
        MarlinLinear: FusionKernelSpec("marlin", gptq=True),
        AwqMarlinLinear: FusionKernelSpec("marlin"),
        TorchAtenLinear: FusionKernelSpec("cpu_aten", gptq=True,
                                        fields=("_cpu_transformed", "_cpu_transform_dtype", "linear_mode")),
        TorchAtenAwqLinear: FusionKernelSpec("cpu_aten",
                                           fields=("_cpu_transformed", "_cpu_transform_dtype", "linear_mode")),
        TorchCpuAvx512Linear: FusionKernelSpec("cpu_avx512", gptq=True,
                                             fields=("_cpu_transformed", "_cpu_transform_dtype", "linear_mode")),
        TorchCpuAvx512AwqLinear: FusionKernelSpec("cpu_avx512", fields=("_cpu_transformed", "linear_mode")),
        TorchFusedLinear: FusionKernelSpec("cpu_legacy", gptq=True, fields=("linear_mode",)),
        TorchFusedAwqLinear: FusionKernelSpec("cpu_legacy", fields=("linear_mode",)),
        AwqGEMMLinear: FusionKernelSpec("awq", fields=("fp32_accum",)),
        AwqGEMMTritonLinear: FusionKernelSpec("awq", fields=("fp32_accum", "awq_triton_schedule_mode", "awq_triton_schedule")),
        AwqGEMVLinear: FusionKernelSpec("awq_gemv", fields=("split_k_iters",)),
        AwqGEMVTritonLinear: FusionKernelSpec("awq_gemv", fields=("split_k_iters",)),
        AwqGEMVFastLinear: FusionKernelSpec("awq_fast", fields=("zeros_name",)),
        LLMAwqLinear: FusionKernelSpec("awq_fast", fields=("zeros_name",), zero_name="scaled_zeros"),
        ExllamaV2Linear: FusionKernelSpec("exllamav2", gptq=True, private_fields=("q_handle", "_fusion_native_view")),
        AwqExllamaV2Linear: FusionKernelSpec("exllamav2", private_fields=("q_handle", "_fusion_native_view")),
        AllSparkLinear: FusionKernelSpec("allspark", gptq=True, zero_name=None,
                                        weight_name="qweight_packed", scale_name="scales_packed",
                                        fields=("_hip", "_repacked")),
        AllSparkAwqLinear: FusionKernelSpec("allspark", zero_name="zeros_packed",
                                           weight_name="qweight_packed", scale_name="scales_packed",
                                           fields=("_hip", "_repacked")),
        TorchInt8Linear: FusionKernelSpec("int8", gptq=True, zero_name=None,
                                         weight_name="int8_weight_nk", scale_name="int8_channel_scale"),
    }


def kernel_spec(module):
    return kernel_specs().get(type(module))
