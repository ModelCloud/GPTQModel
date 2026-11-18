# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measured CPU ``torch.compile`` levels for every CPU-capable quantized kernel.

Each case builds a real packed module on CPU, then measures it with
``tests/kernels/compile_probe.py`` on torch 2.16.0.dev20261021+cpu. The CPU
level is read from ``SUPPORTS_COMPILE_LEVEL_BY_DEVICE["cpu"]`` when the kernel
declares one, otherwise from ``SUPPORTS_COMPILE_LEVEL``.

``fullgraph=True`` is the "graph safe" bar here: reaching ``DYNAMO`` or above
means dynamo traced the whole forward with zero graph breaks. A kernel whose
CPU forward is documented as ``EAGER`` (``NF12Linear``, ``ParoLinear``,
``QQQTorchLinear``, ``GGUFCppKernel``) only has to run eagerly; the technical
reason is recorded next to its declaration.

Kernels whose optional runtime is absent on this host are skipped with that
reason, not silently counted as passing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

from gptqmodel.nn_modules.qlinear._compile_support import COMPILE_LEVEL, compile_level
from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVTorchLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastTorchLinear, LLMAwqTorchLinear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel
from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.nn_modules.qlinear.paro import ParoLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from gptqmodel.quantization import FORMAT
from tests.kernels.compile_probe import probe


pytestmark = [pytest.mark.cpu]

IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
ROWS = 4
DTYPE = torch.float16


def _dense(*, bias: bool = False, seed: int = 0) -> torch.nn.Linear:
    torch.manual_seed(seed)
    return torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=bias, dtype=DTYPE)


def _gptq_kwargs(*, bits: int = 4, bias: bool = False, group_size: int = GROUP_SIZE):
    return {
        "bits": bits,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "bias": bias,
        "format": FORMAT.GPTQ_V2,
    }


def _pack_gptq(klass, *, bits: int = 4, bias: bool = False):
    module = klass(**_gptq_kwargs(bits=bits, bias=bias))
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        _dense(bias=bias),
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=DTYPE),
        torch.full((OUT_FEATURES, groups), 1 << (bits - 1), dtype=torch.int32),
        torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE,
    )
    if hasattr(module, "post_init"):
        module.post_init()
    return module.eval()


def _pack_int8():
    """``TorchInt8Linear`` is not packable; build it from an 8-bit checkpoint.

    ``post_init()`` converts the stored 8-bit GPTQ tensors to the INT8 layout.
    """

    source = _pack_gptq(TorchLinear, bits=8)
    module = TorchInt8Linear(
        bits=8,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        src, dst = getattr(source, name, None), getattr(module, name, None)
        if (
            isinstance(src, torch.Tensor)
            and isinstance(dst, torch.Tensor)
            and src.shape == dst.shape
        ):
            dst.data.copy_(src)
    module.post_init()
    return module.eval()


def _awq_new(klass, *, bits: int = 4, bias: bool = True):
    return klass(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=bias,
        register_buffers=True,
    )


def _pack_awq(klass, *, bits: int = 4, bias: bool = True):
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.full((OUT_FEATURES, groups), 1 / 128, dtype=DTYPE)
    zeros = torch.full((OUT_FEATURES, groups), 1 << (bits - 1), dtype=torch.int32)

    source = _awq_new(AwqTorchLinear, bits=bits, bias=bias)
    source.pack(_dense(bias=bias), scales, zeros)
    if hasattr(source, "post_init"):
        source.post_init()
    source.eval()
    if klass is AwqTorchLinear:
        return source

    module = _awq_new(klass, bits=bits, bias=bias)
    for name in ("qweight", "qzeros", "scales", "bias", "g_idx"):
        src, dst = getattr(source, name, None), getattr(module, name, None)
        if (
            isinstance(src, torch.Tensor)
            and isinstance(dst, torch.Tensor)
            and src.shape == dst.shape
        ):
            dst.data.copy_(src.to(device=dst.device, dtype=dst.dtype))
    if hasattr(module, "post_init"):
        module.post_init()
    return module.eval()


def _pack_gguf():
    module = GGUFTorchLinear(
        bits="q4_k_m",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    module.pack_original(_dense(), scales=None, zeros=None)
    return module.eval()


def _pack_qqq():
    module = QQQTorchLinear(**_gptq_kwargs(group_size=IN_FEATURES))
    module.pack(_dense(), torch.full((OUT_FEATURES, 1), 1 / 128, dtype=DTYPE))
    return module.eval()


def _pack_original(klass, *, group_size: int = -1, bits: int = 4, none_scales: bool = False):
    module = klass(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    if none_scales:
        module.pack_original(_dense(), None, None)
    else:
        module.pack_original(_dense())
    return module.eval()


def _pack_ttbfp():
    from gptqmodel.utils.backend import BACKEND

    module = TTBlockfloatLinear(
        bits=4,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        backend=BACKEND.TT_BLOCKFLOAT,
    )
    module.pack(_dense())
    return module.eval()


def _pack_vptq():
    module = VPTQLinear(
        bits=1,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_size=GROUP_SIZE,
        group_num=-1,
        enable_norm=False,
        enable_perm=False,
        dtype=DTYPE,
    ).eval()
    module.indices.zero_()
    return module


def _unavailable(cls) -> Optional[str]:
    validator = getattr(cls, "cached_validate_once", None)
    if not callable(validator):
        return None
    try:
        ok, error = validator()
    except Exception as exc:  # a missing optional runtime raises lazily
        return f"{cls.__name__} unavailable: {type(exc).__name__}: {exc}"
    return None if ok else f"{cls.__name__} unavailable: {error}"


@dataclass(frozen=True)
class Case:
    cls: type
    build: Optional[Callable[[], torch.nn.Module]]
    unavailable: Callable[[], Optional[str]] = lambda: None


CASES = [
    Case(TorchLinear, lambda: _pack_gptq(TorchLinear)),
    Case(TorchFusedLinear, lambda: _pack_gptq(TorchFusedLinear)),
    Case(TorchInt8Linear, _pack_int8),
    Case(TorchAtenLinear, lambda: _pack_gptq(TorchAtenLinear)),
    Case(TorchCpuAvx512Linear, lambda: _pack_gptq(TorchCpuAvx512Linear)),
    Case(AwqTorchLinear, lambda: _pack_awq(AwqTorchLinear, bias=True)),
    # A biased FP16/BF16 AWQ layer asks for an FP32 product so the pre-bias sum
    # cannot overflow; inductor cannot lower the int4 op when that cast fuses
    # into a Pointwise node (#4068). The declared CPU level is the bias-free
    # path; the biased case is the documented break in
    # ``torch_fused._cpu_int4pack_mm_cast``.
    Case(TorchFusedAwqLinear, lambda: _pack_awq(TorchFusedAwqLinear, bias=False)),
    Case(TorchAtenAwqLinear, lambda: _pack_awq(TorchAtenAwqLinear, bias=False)),
    Case(TorchCpuAvx512AwqLinear, lambda: _pack_awq(TorchCpuAvx512AwqLinear, bias=True)),
    Case(AwqGEMVTorchLinear, lambda: _pack_awq(AwqGEMVTorchLinear, bias=True)),
    Case(LLMAwqTorchLinear, lambda: _pack_awq(LLMAwqTorchLinear, bias=True)),
    Case(AwqGEMVFastTorchLinear, lambda: _pack_awq(AwqGEMVFastTorchLinear, bias=True)),
    Case(GGUFTorchLinear, _pack_gguf),
    Case(TTBlockfloatLinear, _pack_ttbfp),
    Case(TorchFP8Linear, lambda: _pack_original(TorchFP8Linear, bits=8, none_scales=True)),
    Case(
        BitsAndBytesLinear,
        lambda: _pack_original(BitsAndBytesLinear, group_size=GROUP_SIZE, none_scales=True),
    ),
    Case(NF12Linear, lambda: _pack_original(NF12Linear, bits=12)),
    Case(ParoLinear, lambda: _pack_awq(ParoLinear, bias=True)),
    Case(QQQTorchLinear, _pack_qqq),
    Case(VPTQLinear, _pack_vptq),
    Case(GGUFCppKernel, None, lambda: _unavailable(GGUFCppKernel)),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.cls.__name__)
def test_cpu_kernel_reaches_declared_cpu_level(case):
    reason = case.unavailable()
    if reason is not None:
        pytest.skip(reason)
    if case.build is None:
        pytest.skip(f"{case.cls.__name__}: no CPU builder on this host")

    try:
        module = case.build()
    except Exception as error:
        # A builder that needs an optional runtime is an environment skip, not
        # a stale declaration.
        pytest.skip(
            f"{case.cls.__name__}: cannot build on CPU: "
            f"{type(error).__name__}: {str(error)[:200]}"
        )

    x = torch.randn(ROWS, IN_FEATURES, dtype=DTYPE)
    result = probe(module, (x,), check_cuda_graph=False)
    declared = compile_level(case.cls, device_type="cpu")

    assert result.eager, f"{case.cls.__name__}: CPU eager forward failed: {result.eager_error}"
    assert result.reached(declared), (
        f"{case.cls.__name__}: declares CPU {declared.name} but the probe reported "
        f"{result.summary()} errors={result.errors}"
    )

    if declared >= COMPILE_LEVEL.DYNAMO:
        # ``fullgraph=True`` already proves zero graph breaks; keep the
        # explicit check so a future probe change cannot silently drop it.
        assert result.levels.get("dynamo"), (
            f"{case.cls.__name__}: declared CPU {declared.name} without a clean "
            f"fullgraph trace: {result.errors.get('dynamo')}"
        )
    if declared >= COMPILE_LEVEL.INDUCTOR:
        assert result.output_close.get("inductor", True), (
            f"{case.cls.__name__}: inductor output does not match eager on CPU"
        )
