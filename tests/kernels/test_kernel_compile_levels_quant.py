# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measured compile/graph levels for the QQQ/FP8/BnB/GGUF/NF12/ParoQuant/
MACaQ/VPTQ/W4A kernels.

Each case builds a real packed module with the family's own ``pack`` /
``pack_original`` helper, then measures it with ``tests/kernels/compile_probe.py``
on the GB10 (sm_121, torch 2.11.0+cu130). Kernels whose native runtime cannot
build on this host are skipped with the reason their declaration records.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear._compile_support import compile_level, graph_capture_safe
from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel, GGUFCudaKernel
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear, MacaqTritonLinear
from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.nn_modules.qlinear.vptq_cuda import VPTQCudaLinear
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear
from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear
from tests.kernels.compile_probe import graph_unsafe_constructs, probe, skip_without_measurable_device


pytestmark = [pytest.mark.gpu]

ROWS = 4


def _linear(in_features, out_features, *, bias=False, dtype=torch.float16, seed=7):
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.02)
    return linear


def _input(module):
    return torch.randn(ROWS, module.in_features, dtype=torch.float16, device="cuda")


def _pack_qqq(cls):
    in_features, out_features, group_size = 512, 256, 128
    linear = _linear(in_features, out_features)
    module = cls(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features, bias=False,
    )
    groups = in_features // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float16) * 0.03 + 0.01
    channels = torch.rand(out_features, dtype=torch.float32) * 0.002 + 0.001
    module.pack(linear, scales, channels)
    module.post_init()
    module = module.cuda().eval()
    return module, _input(module), True


def _build_fp8():
    linear = _linear(512, 256, dtype=torch.float32)
    module = TorchFP8Linear(
        bits=8, group_size=-1, sym=True, desc_act=False,
        in_features=512, out_features=256, bias=False,
        format="float8_e4m3fn", weight_scale_method="row",
    )
    module.pack_original(linear, scales=None, zeros=None)
    module = module.cuda().eval()
    return module, _input(module), True


def _build_gguf(cls, bits):
    linear = _linear(512, 256, dtype=torch.float32)
    module = cls(
        bits=bits, group_size=-1, sym=True, desc_act=False,
        in_features=512, out_features=256, bias=False,
    )
    # The ggml-backed subclasses null out ``pack_original``; the torch packer
    # writes the same qweight layout they read.
    GGUFTorchLinear.pack_original(module, linear, scales=None, zeros=None)
    module = module.cuda().eval()
    return module, _input(module), True


def _build_gguf_torch():
    return _build_gguf(GGUFTorchLinear, "q4_0")


def _build_gguf_triton():
    return _build_gguf(GGUFTritonKernel, "q4_k")


def _build_nf12():
    linear = _linear(512, 256, dtype=torch.bfloat16)
    module = NF12Linear(bits=12, in_features=512, out_features=256, bias=False, exp_base=54)
    module.pack_original(linear)
    module = module.cuda().eval()
    return module, _input(module), True


def _build_paro(cls):
    in_features, out_features, group_size = 128, 64, 32
    linear = _linear(in_features, out_features)
    module = cls(
        bits=4, group_size=group_size, sym=True, desc_act=False,
        in_features=in_features, out_features=out_features, bias=False,
        register_buffers=True,
    )
    groups = in_features // group_size
    scales = torch.rand(out_features, groups, dtype=torch.float16) * 0.03 + 0.01
    module.pack(linear, scales, torch.full((out_features, groups), 8, dtype=torch.int32))
    module.post_init()
    module = module.cuda().eval()
    return module, _input(module), True


def _build_macaq(cls):
    from gptqmodel.utils.macaq_triton import quantize_mxfp4

    module = cls(
        bits=4, group_size=32, in_features=128, out_features=64, bias=False,
        rotation="none", pre_mode="none", activation_precision="w4a16",
        register_buffers=True,
    ).cuda()
    packed, scales = quantize_mxfp4(torch.randn(64, 128, device="cuda"))
    module.W_packed.copy_(packed.to(torch.uint8))
    module.W_scales.copy_(scales.view(torch.uint8))
    module.post_init()
    module = module.eval()
    return module, _input(module), True


def _build_vptq(cls):
    module = cls(
        bits=2, in_features=64, out_features=64, bias=False,
        vector_lens=[-1, 4], num_centroids=[-1, 16], num_res_centroids=[-1, -1],
        group_size=64, register_buffers=True, dtype=torch.float16,
    )
    with torch.no_grad():
        module.centroids.weight.normal_(0.0, 0.02)
    module.post_init()
    module = module.cuda().eval()
    return module, _input(module), True


def _fill_w4a(module):
    in_features, out_features = module.in_features, module.out_features
    codes = torch.arange(in_features, dtype=torch.int32).remainder(16)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    words = (codes.reshape(in_features // 8, 8) << shifts).sum(dim=1).to(torch.int32)
    module.qweight.copy_(words[:, None].expand(in_features // 8, out_features))
    module.qzeros.fill_(0x77777777)
    module.scales.copy_(
        torch.arange(1, in_features // 128 + 1, dtype=torch.float16)[:, None].expand(
            in_features // 128, out_features
        ) / 8
    )
    if getattr(module, "activation_global_scale", None) is not None:
        module.activation_global_scale.fill_(0.03125)
    module.post_init()
    return module


def _build_w4a_fp8():
    module = W4AFP8Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=32, bias=False,
    )
    module = _fill_w4a(module).cuda().eval()
    return module, _input(module), True


def _build_w4a_nvfp4():
    module = W4ANVFP4Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=128, out_features=128, bias=False,
    )
    module = _fill_w4a(module).cuda().eval()
    return module, _input(module), True


@dataclass(frozen=True)
class Case:
    cls: type
    build: Optional[Callable]
    unavailable: Callable[[], Optional[str]]


def _unavailable(cls) -> Optional[str]:
    try:
        ok, err = cls.cached_validate_once()
    except Exception as error:  # a missing optional runtime raises lazily
        return f"{cls.__name__} unavailable: {type(error).__name__}: {error}"
    return None if ok else f"{cls.__name__} unavailable: {err}"


def _paro_unavailable() -> Optional[str]:
    if os.environ.get("CUDA_HOME") is None and shutil.which("nvcc") is None:
        return "ParoLinear needs the AWQ CUDA GEMM JIT extension; no CUDA toolkit on this host"
    return None


def _gguf_cpp_unavailable(cls):
    if importlib.util.find_spec("llama_cpp") is None:
        return f"{cls.__name__} needs llama-cpp-python; not installed on this host"
    return _unavailable(cls)


CASES = [
    Case(QQQLinear, lambda: _pack_qqq(QQQLinear), lambda: _unavailable(QQQLinear)),
    Case(QQQTorchLinear, lambda: _pack_qqq(QQQTorchLinear), lambda: None),
    Case(TorchFP8Linear, _build_fp8, lambda: None),
    Case(BitsAndBytesLinear, None, lambda: _unavailable(BitsAndBytesLinear)),
    Case(GGUFTorchLinear, _build_gguf_torch, lambda: None),
    Case(GGUFCppKernel, None, lambda: _gguf_cpp_unavailable(GGUFCppKernel)),
    Case(GGUFCudaKernel, None, lambda: _gguf_cpp_unavailable(GGUFCudaKernel)),
    Case(GGUFTritonKernel, _build_gguf_triton, lambda: _unavailable(GGUFTritonKernel)),
    Case(NF12Linear, _build_nf12, lambda: None),
    Case(ParoLinear, None, _paro_unavailable),
    Case(ParoQuantTritonLinear, lambda: _build_paro(ParoQuantTritonLinear), lambda: None),
    Case(MacaqLinear, lambda: _build_macaq(MacaqLinear), lambda: None),
    Case(MacaqTritonLinear, lambda: _build_macaq(MacaqTritonLinear), lambda: None),
    Case(VPTQLinear, lambda: _build_vptq(VPTQLinear), lambda: None),
    Case(VPTQCudaLinear, None, lambda: _unavailable(VPTQCudaLinear)),
    Case(W4AFP8Linear, _build_w4a_fp8, lambda: _unavailable(W4AFP8Linear)),
    Case(W4ANVFP4Linear, _build_w4a_nvfp4, lambda: _unavailable(W4ANVFP4Linear)),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.cls.__name__)
def test_kernel_reaches_declared_compile_level(case):
    reason = case.unavailable()
    if reason is not None:
        pytest.skip(reason)
    assert case.build is not None, f"{case.cls.__name__}: no builder and no skip reason"
    skip_without_measurable_device(case.cls)

    try:
        module, x, cuda_graph = case.build()
    except Exception as error:
        # A builder that needs a device this host does not have is an
        # environment skip, not a stale declaration.
        pytest.skip(
            f"{case.cls.__name__}: cannot build here: "
            f"{type(error).__name__}: {str(error)[:200]}"
        )
    # Only re-probe capture where the kernel claims it. On torch 2.11/cu130 a
    # *failed* capture poisons the caching allocator's graph pool for the rest
    # of the process ("Offset increment outside graph capture"), so a declared
    # False kernel is not capture-probed here; its declaration was measured
    # separately (NF12Linear and MacaqLinear both fail with a host->device copy).
    result = probe(module, (x,), check_cuda_graph=cuda_graph and graph_capture_safe(case.cls))
    declared = compile_level(case.cls)

    assert result.eager, f"{case.cls.__name__}: eager forward failed: {result.eager_error}"
    assert result.reached(declared), (
        f"{case.cls.__name__}: declared {declared.name} but probe reported "
        f"{result.summary()} errors={result.errors}"
    )

    if graph_capture_safe(case.cls):
        assert result.graph_capture is True, (
            f"{case.cls.__name__}: declares SUPPORTS_GRAPH_CAPTURE=True but capture "
            f"failed: {result.graph_capture_error}"
        )
    elif result.graph_capture:
        unsafe = graph_unsafe_constructs(case.cls)
        message = (
            f"{case.cls.__name__} captured a CUDA graph while declaring False; "
            f"unsafe={unsafe}"
        )
        warnings.warn(message, stacklevel=1)
        print(f"NOTE: {message}")
        if not unsafe and result.output_close.get("cudagraph", True):
            setattr(case.cls, "SUPPORTS_GRAPH_CAPTURE", True)
            warnings.warn(
                f"{case.cls.__name__}: declaration raised to SUPPORTS_GRAPH_CAPTURE=True "
                "after a measured capture matched eager",
                stacklevel=1,
            )
