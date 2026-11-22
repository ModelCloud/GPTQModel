# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native RDNA4 (gfx120x) EXL3 WMMA GEMM: parity, dispatch and gating.

``quant/exl3_gemm_hip.cu`` decodes the trellis straight into the wave32 WMMA
operands on gfx1200/gfx1201 and leaves every other ROCm target on the portable
reconstruct + hgemm path.  That portable path is the oracle here.  The fused
kernel reorders the FP32 accumulation, so the comparison is a tolerance, not
bit equality.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.models.loader import _resolve_exl3_auto_backend
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.utils.arch import rocm_arch
from gptqmodel.utils.backend import BACKEND


pytestmark = pytest.mark.cuda

# Dispatch codes owned by the HIP path (`quant/exl3_gemm_hip.cuh`).
NATIVE = 93
FALLBACK = 94

# `EXL3_HIP_MAX_M` in `quant/exl3_gemm_hip.cu`: wider batches reconstruct once
# and hand the result to hipBLAS.
NATIVE_MAX_M = 512


def _require_rdna4() -> str:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("ROCm GPU required")
    arch = rocm_arch(torch.cuda.current_device())
    if arch is None or not arch.startswith("gfx120"):
        pytest.skip(f"RDNA4 (gfx120x) GPU required (found {arch or 'none'})")
    return arch


def _ext():
    from gptqmodel.exllamav3.ext import exllamav3_ext, prewarm_exllamav3_extension

    if not prewarm_exllamav3_extension():
        pytest.skip("ExLlamaV3 extension is not available")
    return exllamav3_ext


def _codebooks(bits: float) -> tuple[bool, bool]:
    """The marker pair a rate implies, matching `trellis_rate`."""
    return bits in (2.0, 4.0, 6.0, 8.0), bits in (1.5, 2.5, 3.5)


def _tensors(k: int, n: int, bits: float, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    words = int(round(16 * bits))
    trellis = torch.randint(
        -32768, 32768, (k // 16, n // 16, words),
        generator=generator, dtype=torch.int16,
    ).cuda()
    suh = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    svh = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, suh, svh


def _input(m: int, k: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn((m, k), generator=generator) * 0.1).half().cuda()


def _run(ext, trellis, suh, svh, bits, x, y, *, q4nx: bool = False) -> int:
    mcg, mul1 = _codebooks(bits)
    return int(torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured(
        trellis, suh, svh, float(bits), None, mcg, mul1, x, y,
        False, 0, False, q4nx,
    ))


def _reference(ext, x, trellis, suh, svh, bits) -> torch.Tensor:
    """The portable reconstruct + hgemm path the fused kernel replaces."""
    mcg, mul1 = _codebooks(bits)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    weight = torch.empty((x.shape[1], svh.numel()), dtype=torch.half, device=x.device)
    if float(bits) % 1:
        torch.ops.gptqmodel_exllamav3.reconstruct_frac(weight, trellis, int(bits))
    else:
        ext.reconstruct(weight, trellis, int(bits), mcg, mul1)
    y = torch.empty((x.shape[0], svh.numel()), dtype=torch.float32, device=x.device)
    ext.hgemm(xh, weight, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def _relative_error(result: torch.Tensor, reference: torch.Tensor) -> float:
    scale = reference.abs().max().clamp_min(1e-6)
    return float(((result - reference).abs().max() / scale).item())


@pytest.mark.parametrize(
    ("bits", "m", "k", "n"),
    [
        (2.0, 16, 2048, 2048),
        (2.0, 64, 4096, 4096),
        (3.0, 7, 2048, 4096),
        (5.0, 13, 2048, 2048),
        (8.0, 32, 2048, 2048),
        (2.5, 16, 2048, 2048),
        (1.5, 16, 2048, 2048),
    ],
)
def test_native_parity_matches_reconstruct(bits, m, k, n):
    _require_rdna4()
    ext = _ext()
    trellis, suh, svh = _tensors(k, n, bits)
    x = _input(m, k)
    reference = _reference(ext, x, trellis, suh, svh, bits)

    y = torch.empty((m, n), dtype=torch.float32, device="cuda")
    assert _run(ext, trellis, suh, svh, bits, x, y) == NATIVE
    assert _relative_error(y, reference) < 1e-3


def test_fp16_destination_still_uses_native_kernel():
    # `bc_linear_exl3_run_configured` widens a half destination to an FP32
    # accumulator, so the fp16 GEMM shape reaches the native kernel too and the
    # epilogue copies back once.
    _require_rdna4()
    ext = _ext()
    trellis, suh, svh = _tensors(2048, 2048, 2.0)
    x = _input(16, 2048)
    reference = _reference(ext, x, trellis, suh, svh, 2.0)

    y = torch.empty((16, 2048), dtype=torch.half, device="cuda")
    assert _run(ext, trellis, suh, svh, 2.0, x, y) == NATIVE
    assert _relative_error(y.float(), reference) < 5e-3


def test_wide_batch_falls_back_to_reconstruct():
    _require_rdna4()
    ext = _ext()
    m = NATIVE_MAX_M + 512
    trellis, suh, svh = _tensors(2048, 2048, 2.0)
    x = _input(m, 2048)
    reference = _reference(ext, x, trellis, suh, svh, 2.0)

    y = torch.empty((m, 2048), dtype=torch.float32, device="cuda")
    assert _run(ext, trellis, suh, svh, 2.0, x, y) == FALLBACK
    assert _relative_error(y, reference) < 1e-3


def test_q4nx_raises_with_portable_hint():
    _require_rdna4()
    ext = _ext()
    # 3.0 bpw is a Q4NX rate (step bits 12); the trellis data is never read.
    trellis, suh, svh = _tensors(2048, 2048, 3.0)
    x = _input(16, 2048)
    y = torch.empty((16, 2048), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="Q4NX inference is not available on ROCm"):
        _run(ext, trellis, suh, svh, 3.0, x, y, q4nx=True)


def test_capability_probe_matches_the_device():
    from gptqmodel.exllamav3.ext import exl3_native_hip_gemm_available

    arch = _require_rdna4()
    assert arch.startswith("gfx120")
    assert exl3_native_hip_gemm_available(torch.cuda.current_device())


def test_auto_backend_selects_native_on_rdna4(monkeypatch):
    _require_rdna4()
    # #5836: a covered device with a usable extension is the AUTO default.
    assert _resolve_exl3_auto_backend(torch.device("cuda:0")) == BACKEND.EXL3_EXLLAMA_V3

    # A host whose build has no HIP ops keeps the portable kernel.
    monkeypatch.setattr(
        "gptqmodel.exllamav3.ext.exl3_native_hip_gemm_usable",
        lambda _device=None: False,
    )
    assert _resolve_exl3_auto_backend(torch.device("cuda:0")) == BACKEND.EXL3_TORCH


def test_native_backend_accepts_rdna4():
    _require_rdna4()
    ExllamaV3Linear.validate_device(torch.device("cuda:0"))


def _linear_module(bits: float, k: int, n: int):
    from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3

    trellis, suh, svh = _tensors(k, n, bits)
    mcg, mul1 = _codebooks(bits)
    return LinearEXL3(
        config=None,
        in_features=k,
        out_features=n,
        scale=None,
        suh=suh,
        svh=svh,
        trellis=trellis,
        mcg=torch.zeros((), dtype=torch.int32, device="cuda") if mcg else None,
        mul1=torch.zeros((), dtype=torch.int32, device="cuda") if mul1 else None,
        bias=None,
        out_dtype=torch.float32,
        transformers_fix=True,
    )


def test_linear_forward_uses_the_native_kernel(monkeypatch):
    _require_rdna4()
    ext = _ext()
    from gptqmodel.exllamav3.modules.quant import exl3 as exl3_module

    module = _linear_module(2.0, 2048, 2048)
    x = _input(16, 2048)
    reference = _reference(ext, x, module.trellis, module.suh, module.svh, 2.0)

    # The portable branch reconstructs through the facade; the native branch
    # calls the C++ dispatcher directly, so this must not fire.
    def _portable_path(*_args, **_kwargs):
        raise AssertionError("portable reconstruct path must not run")

    monkeypatch.setattr(exl3_module.ext, "reconstruct", _portable_path)
    result = module.forward(x, {}, out_dtype=torch.float32)
    assert result.dtype == torch.float32
    assert _relative_error(result, reference) < 1e-3


def test_linear_forward_reconstruct_override_uses_the_portable_path(monkeypatch):
    _require_rdna4()
    _ext()
    from gptqmodel.exllamav3.modules.quant import exl3 as exl3_module

    module = _linear_module(2.0, 2048, 2048)
    x = _input(16, 2048)

    def _portable_path(*_args, **_kwargs):
        raise AssertionError("portable reconstruct path requested")

    monkeypatch.setattr(exl3_module.ext, "reconstruct", _portable_path)
    with pytest.raises(AssertionError, match="portable reconstruct path requested"):
        module.forward(x, {"reconstruct": True}, out_dtype=torch.float32)


def test_exllama_v3_linear_forward_uses_the_native_kernel(monkeypatch):
    """The user-facing module, including `validate_device` and `_ensure_inner`."""
    _require_rdna4()
    ext = _ext()
    from gptqmodel.exllamav3.modules.quant import exl3 as exl3_module

    trellis, suh, svh = _tensors(2048, 2048, 2.0)
    module = ExllamaV3Linear(
        in_features=2048,
        out_features=2048,
        name="test",
        tensors={
            "trellis": trellis,
            "suh": suh,
            "svh": svh,
            "mcg": torch.zeros((), dtype=torch.int32, device="cuda"),
        },
    )
    x = _input(16, 2048)
    reference = _reference(ext, x, trellis, suh, svh, 2.0)

    def _portable_path(*_args, **_kwargs):
        raise AssertionError("portable reconstruct path must not run")

    monkeypatch.setattr(exl3_module.ext, "reconstruct", _portable_path)
    result = module.forward(x)
    assert result.dtype == torch.float16
    assert _relative_error(result.float(), reference) < 5e-3
