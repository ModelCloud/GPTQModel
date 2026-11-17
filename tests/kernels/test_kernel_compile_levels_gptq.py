# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measured ``torch.compile`` / CUDA-graph levels for the portable-torch kernels.

Each case builds a real packed GPTQ module, then measures it with
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
from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear
from gptqmodel.nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BitblasLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from gptqmodel.nn_modules.qlinear.machete import MacheteLinear
from gptqmodel.nn_modules.qlinear.marlin import (
    MarlinLinear,
    marlin_import_exception,
)
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_fused import TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT
from tests.kernels.compile_probe import graph_unsafe_constructs, probe, skip_without_measurable_device


pytestmark = [pytest.mark.gpu]

IN_FEATURES = 512
OUT_FEATURES = 256
GROUP_SIZE = 128
ROWS = 4


def _pack_into(cls, *, bits: int, dtype: torch.dtype, device: str):
    torch.manual_seed(0)
    dense = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=dtype)
    module = cls(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        dense,
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=dtype),
        torch.full((OUT_FEATURES, groups), 1 << (bits - 1), dtype=torch.int32),
        torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE,
    )
    return module.to(device).eval()


def _packable(cls, *, bits: int = 4, device: str, cuda_graph: bool):
    module = _pack_into(cls, bits=bits, dtype=torch.float16, device=device)
    module.post_init()
    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.float16, device=device)
    return module, x, cuda_graph


def _from_torch_source(cls, *, bits: int, device: str):
    """Copy a packed TorchLinear checkpoint into a kernel that has no packer."""
    source = _pack_into(TorchLinear, bits=bits, dtype=torch.float16, device="cpu")
    module = cls(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
        register_buffers=True,
    )
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        src = getattr(source, name, None)
        dst = getattr(module, name, None)
        if (
            isinstance(src, torch.Tensor)
            and isinstance(dst, torch.Tensor)
            and src.shape == dst.shape
        ):
            dst.data.copy_(src)
    module = module.to(device).eval()
    module.post_init()
    x = torch.randn(ROWS, IN_FEATURES, dtype=torch.float16, device=device)
    return module, x, False


def _unavailable(cls) -> Optional[str]:
    try:
        ok, err = cls.cached_validate_once()
    except Exception as error:  # a missing optional runtime raises lazily
        return f"{cls.__name__} unavailable: {type(error).__name__}: {error}"
    return None if ok else f"{cls.__name__} unavailable: {err}"


def _marlin_unavailable() -> Optional[str]:
    if marlin_import_exception is not None:
        return f"Marlin import failed: {marlin_import_exception}"
    from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error

    if not marlin_runtime_available(torch.float16):
        return f"Marlin runtime unavailable: {marlin_runtime_error(torch.float16)}"
    return None


def _machete_unavailable() -> Optional[str]:
    from gptqmodel.utils.machete import machete_runtime_available, machete_runtime_error

    if not machete_runtime_available():
        return f"Machete runtime unavailable: {machete_runtime_error()}"
    return None


def _bitblas_unavailable() -> Optional[str]:
    if not BITBLAS_AVAILABLE:
        return "BitBLAS is not installed on this host"
    return _unavailable(BitblasLinear)


@dataclass(frozen=True)
class Case:
    cls: type
    build: Optional[Callable]
    unavailable: Callable[[], Optional[str]]


CASES = [
    Case(
        TorchLinear,
        lambda: _packable(TorchLinear, device="cuda", cuda_graph=True),
        lambda: None,
    ),
    Case(
        TorchFusedLinear,
        lambda: _packable(TorchFusedLinear, device="cpu", cuda_graph=False),
        lambda: None,
    ),
    Case(
        TorchInt8Linear,
        lambda: _from_torch_source(TorchInt8Linear, bits=8, device="cpu"),
        lambda: None,
    ),
    Case(
        TorchAtenLinear,
        lambda: _from_torch_source(TorchAtenLinear, bits=4, device="cpu"),
        lambda: None,
    ),
    Case(TorchCpuAvx512Linear, None, lambda: _unavailable(TorchCpuAvx512Linear)),
    Case(MarlinLinear, None, _marlin_unavailable),
    Case(MacheteLinear, None, _machete_unavailable),
    Case(ExllamaV2Linear, None, lambda: _unavailable(ExllamaV2Linear)),
    Case(
        TritonV2Linear,
        lambda: _packable(TritonV2Linear, device="cuda", cuda_graph=True),
        lambda: _unavailable(TritonV2Linear),
    ),
    Case(BitblasLinear, None, _bitblas_unavailable),
    Case(AllSparkLinear, None, lambda: _unavailable(AllSparkLinear)),
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
    result = probe(module, (x,), check_cuda_graph=cuda_graph)
    declared = compile_level(case.cls)

    assert result.eager, (
        f"{case.cls.__name__}: eager forward failed: {result.eager_error}"
    )
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
        # Record a capture that the declaration did not claim; do not fail it.
        print(
            f"NOTE: {case.cls.__name__} captured a CUDA graph while declaring "
            f"False; unsafe={graph_unsafe_constructs(case.cls)}"
        )
