# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measured ``torch.compile`` / CUDA-graph levels for the AWQ kernels.

Each case builds a real packed AWQ module, then measures it with
``tests/kernels/compile_probe.py`` on the GB10 (sm_121, torch 2.11.0+cu130).
Kernels whose runtime cannot build on this host are skipped with the reason
their declaration records, instead of faking a level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pytest
import torch

from gptqmodel.nn_modules.qlinear._compile_support import (
    compile_level,
    graph_capture_safe,
)
from gptqmodel.nn_modules.qlinear.allspark_awq import AllSparkAwqLinear
from gptqmodel.nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE
from gptqmodel.nn_modules.qlinear.bitblas_awq import AWQBitBlasKernel
from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.nn_modules.qlinear.gemv_awq import (
    AwqGEMVLinear,
    AwqGEMVTorchLinear,
    AwqGEMVTritonLinear,
)
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear, LLMAwqLinear
from gptqmodel.nn_modules.qlinear.machete_awq import AwqMacheteLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.swordfish import AwqSwordfishLinear, SwordfishLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear
from gptqmodel.nn_modules.qlinear.torch_int8_awq import TorchInt8AwqLinear
from gptqmodel.quantization.awq.utils.packing_utils import AWQ_ORDER, pack_awq_columns
from tests.kernels.compile_probe import graph_unsafe_constructs, probe


pytestmark = [pytest.mark.gpu]

IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
ROWS = 4
BITS = 4


IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
ROWS = 4
BITS = 4


def _affine(weight: torch.Tensor, dtype: torch.dtype):
    """Synthetic AWQ scales/zeros in the ``[out_features, groups]`` layout."""

    groups = IN_FEATURES // GROUP_SIZE
    grouped = weight.detach().float().view(OUT_FEATURES, groups, GROUP_SIZE)
    low, high = grouped.amin(-1), grouped.amax(-1)
    scales = ((high - low).clamp_min(1e-6) / ((1 << BITS) - 1)).to(dtype)
    zeros = torch.full_like(scales, 1 << (BITS - 1), dtype=torch.int32)
    return scales, zeros


def _packed(cls, *, device: str, cuda_graph: bool):
    """Build through the kernel's own AWQ ``pack`` (GEMM/GEMV layout)."""

    torch.manual_seed(0)
    dense = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )
    scales, zeros = _affine(dense.weight, torch.float16)
    module.pack(dense, scales, zeros)
    module = module.to(device).eval()
    module.post_init()
    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.float16, device=device)
    return module, x, cuda_graph


def _checkpoint(cls, *, device: str, cuda_graph: bool):
    """Copy a real AWQ GEMM-layout checkpoint into a kernel without a packer."""

    torch.manual_seed(0)
    groups = IN_FEATURES // GROUP_SIZE
    codes = torch.randint(0, 1 << BITS, (IN_FEATURES, OUT_FEATURES), dtype=torch.int32)
    zeros = torch.full((groups, OUT_FEATURES), 1 << (BITS - 1), dtype=torch.int32)
    qweight = pack_awq_columns(codes, BITS, order=AWQ_ORDER)
    qzeros = pack_awq_columns(zeros, BITS, order=AWQ_ORDER)
    scales = (torch.rand(groups, OUT_FEATURES, dtype=torch.float32) * 0.02 + 0.001).to(
        torch.float16
    )
    bias = torch.randn(OUT_FEATURES, dtype=torch.float16) * 0.02

    module = cls(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        register_buffers=True,
    )
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.bias.copy_(bias)
    module = module.to(device).eval()
    module.post_init()
    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.float16, device=device)
    return module, x, cuda_graph


def _validate_unavailable(cls) -> Optional[str]:
    try:
        ok, error = cls.validate_once()
    except Exception as exc:  # a missing optional runtime raises lazily
        return f"{cls.__name__} unavailable: {type(exc).__name__}: {exc}"
    return None if ok else f"{cls.__name__} unavailable: {error}"


def _awq_unavailable() -> Optional[str]:
    from gptqmodel.utils.awq import awq_runtime_available, awq_runtime_error

    if not awq_runtime_available():
        return f"native AWQ JIT extension unavailable: {str(awq_runtime_error())[:240]}"
    return None


def _marlin_unavailable() -> Optional[str]:
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error

    if not marlin_runtime_available(torch.float16):
        return f"Marlin runtime unavailable: {str(marlin_runtime_error(torch.float16))[:240]}"
    return None


def _machete_unavailable() -> Optional[str]:
    from gptqmodel.utils.machete import machete_runtime_available, machete_runtime_error

    if not machete_runtime_available():
        return f"Machete runtime unavailable: {str(machete_runtime_error())[:240]}"
    return None


def _swordfish_unavailable() -> Optional[str]:
    from gptqmodel.utils.swordfish import swordfish_runtime_available, swordfish_runtime_error

    if not swordfish_runtime_available():
        return f"Swordfish runtime unavailable: {str(swordfish_runtime_error())[:240]}"
    return None


def _exllamav2_unavailable() -> Optional[str]:
    from gptqmodel.utils.exllamav2 import (
        exllamav2_awq_runtime_available,
        exllamav2_awq_runtime_error,
    )

    if not exllamav2_awq_runtime_available():
        return f"ExLlamaV2 AWQ runtime unavailable: {str(exllamav2_awq_runtime_error())[:240]}"
    return None


def _bitblas_unavailable() -> Optional[str]:
    return None if BITBLAS_AVAILABLE else "BitBLAS is not installed on this host"


@dataclass(frozen=True)
class Case:
    cls: type
    build: Optional[Callable]
    unavailable: Callable[[], Optional[str]]


CASES = [
    Case(
        AwqTorchLinear,
        lambda: _packed(AwqTorchLinear, device="cuda", cuda_graph=True),
        lambda: None,
    ),
    Case(
        AwqGEMVTritonLinear,
        lambda: _packed(AwqGEMVTritonLinear, device="cuda", cuda_graph=True),
        lambda: None,
    ),
    Case(
        AwqGEMVTorchLinear,
        lambda: _packed(AwqGEMVTorchLinear, device="cuda", cuda_graph=True),
        lambda: None,
    ),
    Case(
        AwqGEMMTritonLinear,
        lambda: _checkpoint(AwqGEMMTritonLinear, device="cuda", cuda_graph=True),
        lambda: None,
    ),
    Case(
        TorchFusedAwqLinear,
        lambda: _checkpoint(TorchFusedAwqLinear, device="cpu", cuda_graph=False),
        lambda: None,
    ),
    Case(
        TorchAtenAwqLinear,
        lambda: _checkpoint(TorchAtenAwqLinear, device="cpu", cuda_graph=False),
        lambda: None,
    ),
    Case(
        TorchCpuAvx512AwqLinear,
        None,
        lambda: _validate_unavailable(TorchCpuAvx512AwqLinear),
    ),
    Case(
        TorchInt8AwqLinear,
        None,
        lambda: "retired backend: __init__ raises NotImplementedError",
    ),
    Case(AwqGEMVLinear, None, _awq_unavailable),
    Case(AwqGEMVFastLinear, None, _awq_unavailable),
    Case(LLMAwqLinear, None, _awq_unavailable),
    Case(AwqGEMMLinear, None, _awq_unavailable),
    Case(AwqMarlinLinear, None, _marlin_unavailable),
    Case(AwqMacheteLinear, None, _machete_unavailable),
    Case(SwordfishLinear, None, _swordfish_unavailable),
    Case(AwqSwordfishLinear, None, _swordfish_unavailable),
    Case(AwqExllamaV2Linear, None, _exllamav2_unavailable),
    Case(AWQBitBlasKernel, None, _bitblas_unavailable),
    Case(AllSparkAwqLinear, None, lambda: _validate_unavailable(AllSparkAwqLinear)),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.cls.__name__)
def test_awq_kernel_reaches_declared_compile_level(case):
    reason = case.unavailable()
    if reason is not None:
        pytest.skip(reason)
    assert case.build is not None, f"{case.cls.__name__}: no builder and no skip reason"

    module, x, cuda_graph = case.build()
    result = probe(module, (x,), check_cuda_graph=cuda_graph)
    declared = compile_level(case.cls)

    assert result.eager, (
        f"{case.cls.__name__}: eager forward failed: {result.eager_error}"
    )
    assert result.reached(declared), (
        f"{case.cls.__name__}: declared {declared.name} but probe reported "
        f"{result.summary()} errors={result.errors}"
    )
    for name, close in result.output_close.items():
        assert close, f"{case.cls.__name__}: {name} output differs from eager"

    if graph_capture_safe(case.cls):
        assert result.graph_capture is True, (
            f"{case.cls.__name__}: declares SUPPORTS_GRAPH_CAPTURE=True but capture "
            f"failed: {result.graph_capture_error}"
        )
    elif result.graph_capture:
        # Record a capture the declaration did not claim; do not fail it.
        print(
            f"NOTE: {case.cls.__name__} captured a CUDA graph while declaring "
            f"False; unsafe={graph_unsafe_constructs(case.cls)}"
        )
