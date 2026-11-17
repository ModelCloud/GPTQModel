# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compile / CUDA-graph capability of the non-torch and TT block-float kernels.

MLX arrays, TTNN device tensors and XLA device tensors are not torch tensors:
their own compiler owns the graph, so no ``torch.compile`` level applies and
there is nothing to probe. ``TTBlockfloatLinear`` *is* a torch module and is
measured with ``tests/kernels/compile_probe.py`` on the local accelerator; when
its native block-float CUDA extension cannot build, the case is skipped with
that reason instead of faking a level.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from gptqmodel.nn_modules.qlinear import (
    mlx,
    tt_blockfloat,
    tt_xla,
    ttnn,
    ttnn_blockfloat,
)
from gptqmodel.nn_modules.qlinear._compile_support import (
    COMPILE_LEVEL,
    compile_level,
    graph_capture_safe,
)
from gptqmodel.utils.backend import BACKEND
from tests.kernels.compile_probe import probe


pytestmark = [pytest.mark.gpu]

IN_FEATURES = 64
OUT_FEATURES = 8
ROWS = 2


@dataclass(frozen=True)
class Case:
    cls: type
    #: Import name of the non-torch runtime, or ``None`` for a torch module.
    runtime: Optional[str]
    label: str


#: The 16 MLX checkpoint holders declared NOT_APPLICABLE below.
MLX_KERNEL_NAMES = (
    "AWQBitBLASMlxQuantLinear",
    "AWQMarlinMlxQuantLinear",
    "AwqGemvFastMlxQuantLinear",
    "AwqGemvMlxQuantLinear",
    "AwqMlxQuantLinear",
    "BitsAndBytesMlxQuantLinear",
    "FP8MlxQuantLinear",
    "GGUFMlxQuantLinear",
    "GPTQBitBLASMlxQuantLinear",
    "GPTQMarlinMlxQuantLinear",
    "LLMAwqMlxQuantLinear",
    "MacaqMlxQuantLinear",
    "MlxQuantLinear",
    "NF12MlxQuantLinear",
    "ParoMlxQuantLinear",
    "QQQMlxQuantLinear",
)


CASES = [
    *(Case(getattr(mlx, name), "mlx", "MLX") for name in MLX_KERNEL_NAMES),
    Case(ttnn.TTNNLinear, "ttnn", "TTNN"),
    Case(ttnn_blockfloat.TTNNBlockFloat8Linear, "ttnn", "TTNN"),
    Case(tt_xla.XLALinear, "torch_xla", "XLA"),
    Case(tt_blockfloat.TTBlockfloatLinear, None, "torch"),
]


def _tt_blockfloat_unavailable() -> Optional[str]:
    if not torch.cuda.is_available():
        return "no CUDA device for the Tenstorrent block-float path"
    from gptqmodel.utils import tt_blockfloat_cuda

    if not tt_blockfloat_cuda.available():
        return (
            "native Tenstorrent block-float CUDA extension cannot build on this "
            "host (no CUDA toolkit: CUDA_HOME/nvcc missing), so only the dense "
            "fallback would run; the declaration stays conservative at EAGER"
        )
    return None


def _build_tt_blockfloat():
    torch.manual_seed(0)
    source = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True)
    module = tt_blockfloat.TTBlockfloatLinear(
        bits=4,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=True,
        backend=BACKEND.TT_BLOCKFLOAT,
        dtype=torch.float32,
    ).pack(source)
    module = module.to("cuda").eval()
    x = torch.randn(ROWS, IN_FEATURES, device="cuda", dtype=torch.float32)
    return module, x


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.cls.__name__)
def test_kernel_compile_level(case):
    declared = compile_level(case.cls)

    if case.runtime is not None:
        assert declared is COMPILE_LEVEL.NOT_APPLICABLE, (
            f"{case.cls.__name__} runs on {case.label} arrays, not torch tensors"
        )
        assert graph_capture_safe(case.cls) is False, (
            f"{case.cls.__name__}: a {case.label} kernel has no torch eager "
            "forward to capture"
        )
        installed = importlib.util.find_spec(case.runtime) is not None
        detail = (
            f"{case.runtime!r} is installed, but its arrays are not torch tensors"
            if installed
            else f"{case.runtime!r} is not installed here"
        )
        pytest.skip(
            f"{case.label} ({case.cls.__name__}) is not a torch module: {detail}, "
            "so its own compiler owns the graph and no torch.compile probe applies"
        )

    assert declared is not COMPILE_LEVEL.NOT_APPLICABLE, (
        f"{case.cls.__name__} is a torch module and must declare a torch level"
    )
    reason = _tt_blockfloat_unavailable()
    if reason is not None:
        pytest.skip(reason)

    module, x = _build_tt_blockfloat()
    result = probe(module, (x,), check_cuda_graph=graph_capture_safe(case.cls))
    assert result.eager, (
        f"{case.cls.__name__}: eager forward failed: {result.eager_error}"
    )
    assert result.reached(declared), (
        f"{case.cls.__name__}: declared {declared.name} but probe reported "
        f"{result.summary()} errors={result.errors}"
    )
    if graph_capture_safe(case.cls):
        assert result.graph_capture is True, (
            f"{case.cls.__name__}: declares graph capture but capture failed: "
            f"{result.graph_capture_error}"
        )
