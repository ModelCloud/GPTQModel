# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Source-level checks for JIT kernels migrated to the torch stable ABI.

The stable ABI ("AnyTorch") lets one compiled artifact run on every newer
torch release. These tests assert that a migrated loader declares the ABI
floor and that its source graph only includes stable torch headers, without
requiring a GPU or a compiler.
"""

from __future__ import annotations

import re
from pathlib import Path

from gptqmodel.exllamav3 import ext as exllamav3
from gptqmodel.quantization import gptq_cuda
from gptqmodel.utils import (
    allspark,
    allspark_hip,
    awq,
    cpp,
    exllamav2,
    hadamard,
    machete,
    marlin,
    nf12_hip,
    paroquant,
    qqq,
    rco_cpp,
    swordfish,
    tt_blockfloat_cuda,
    tt_blockfloat_hip,
    vptq_cuda,
)


def _assert_stable(loader, sources=None, include_paths=None) -> None:
    assert loader.torch_stable_abi_target == cpp._TORCH_STABLE_ABI_TARGET
    if sources is None:
        sources = loader._resolve_sequence(loader.sources)
    if include_paths is None:
        include_paths = loader._resolved_extra_include_paths()
    python_dependent, stable, non_stable_headers = loader._source_abi_details(
        sources, include_paths
    )
    assert stable, non_stable_headers
    assert not python_dependent
    assert non_stable_headers == ()
    if loader.requires_cuda:
        assert "-DUSE_CUDA" in loader._resolved_extra_cflags()


def test_stable_abi_floor_is_torch_2_10():
    assert cpp._TORCH_STABLE_ABI_TARGET == (2, 10)
    assert (
        cpp.torch_stable_abi_target_define(*cpp._TORCH_STABLE_ABI_TARGET)
        == "-DTORCH_TARGET_VERSION=0x020A000000000000"
    )


def test_pack_block_cpu_uses_stable_abi():
    _assert_stable(cpp._pack_block_extension())


def test_floatx_cpu_uses_stable_abi():
    _assert_stable(cpp._floatx_cpu_extension())


def test_hadamard_uses_stable_abi():
    _assert_stable(hadamard._HADAMARD_TORCH_OPS_EXTENSION)


def test_qqq_uses_stable_abi():
    _assert_stable(qqq._QQQ_TORCH_OPS_EXTENSION)


def test_awq_uses_stable_abi():
    _assert_stable(awq._AWQ_TORCH_OPS_EXTENSION)


def test_paroquant_uses_stable_abi():
    _assert_stable(paroquant._PAROQUANT_ROTATION_EXTENSION)


def test_exllamav2_gptq_uses_stable_abi():
    _assert_stable(exllamav2._EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION)


def test_exllamav2_awq_uses_stable_abi():
    _assert_stable(exllamav2._EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION)


def test_exllamav3_uses_stable_abi():
    _assert_stable(exllamav3._EXLLAMAV3_TORCH_OPS_EXTENSION)


def test_gptq_block_uses_stable_abi():
    _assert_stable(gptq_cuda._EXTENSION)


def test_machete_uses_stable_abi():
    _assert_stable(machete._MACHETE_TORCH_OPS_EXTENSION)


def test_allspark_uses_stable_abi():
    _assert_stable(allspark._ALLSPARK_EXTENSION)


def test_allspark_hip_uses_stable_abi():
    _assert_stable(allspark_hip._EXTENSION)


def test_nf12_hip_uses_stable_abi():
    # The loader stages its sources into the build cache, so point the check at
    # the shipped binding instead of triggering that staging. The include path
    # only has to resolve the compat header for this read-only scan.
    root = Path(__file__).resolve().parents[1] / "gptqmodel_ext" / "nf12_hip"
    _assert_stable(
        nf12_hip._EXTENSION,
        sources=[str(root / "nf12_torch.cpp")],
        include_paths=[cpp.torch_stable_compat_include_path()],
    )


def test_tt_blockfloat_cuda_uses_stable_abi():
    _assert_stable(tt_blockfloat_cuda._EXTENSION)


def test_tt_blockfloat_hip_uses_stable_abi():
    root = Path(__file__).resolve().parents[1] / "gptqmodel_ext" / "tt_blockfloat_hip"
    _assert_stable(
        tt_blockfloat_hip._EXTENSION,
        sources=[str(root / "tt_blockfloat_hip_torch.cpp")],
        include_paths=[cpp.torch_stable_compat_include_path()],
    )


def test_vptq_cuda_uses_stable_abi():
    _assert_stable(vptq_cuda._VPTQ_TORCH_OPS_EXTENSION)


def test_marlin_fp16_core_uses_stable_abi():
    _assert_stable(marlin._MARLIN_FP16_CORE_EXTENSION)


def test_marlin_bf16_core_uses_stable_abi():
    _assert_stable(marlin._MARLIN_BF16_CORE_EXTENSION)


def test_marlin_adapter_is_intentionally_torch_specific():
    """The public GEMM name lives in a thin per-torch adapter.

    `Tensor::record_stream` is not part of the stable surface, so the adapter
    is rebuilt for the installed torch while the heavy core stays on the
    stable ABI. It must load its core first and never claim the stable floor.
    """

    for adapter, core in (
        (marlin._MARLIN_FP16_TORCH_OPS_EXTENSION, marlin._MARLIN_FP16_CORE_EXTENSION),
        (marlin._MARLIN_BF16_TORCH_OPS_EXTENSION, marlin._MARLIN_BF16_CORE_EXTENSION),
    ):
        assert adapter.torch_stable_abi_target is None
        assert adapter.with_cuda is True
        assert adapter._core_extension is core
        assert core.torch_stable_abi_target == cpp._TORCH_STABLE_ABI_TARGET


def test_marlin_core_cache_key_ignores_torch_version(monkeypatch):
    """The heavy core must reuse one artifact across torch releases."""

    core = marlin._MARLIN_FP16_CORE_EXTENSION
    monkeypatch.setattr(cpp.torch, "__version__", "2.10.0+cu128")
    fingerprint = core._cache_fingerprint()
    monkeypatch.setattr(cpp.torch, "__version__", "2.15.0+cu132")
    assert core._cache_fingerprint() == fingerprint


def test_marlin_adapter_cache_key_tracks_torch_version(monkeypatch):
    """The adapter uses the unstable ABI, so it rebuilds per torch release."""

    adapter = marlin._MARLIN_FP16_TORCH_OPS_EXTENSION
    monkeypatch.setattr(cpp.torch, "__version__", "2.10.0+cu128")
    fingerprint = adapter._cache_fingerprint()
    monkeypatch.setattr(cpp.torch, "__version__", "2.15.0+cu132")
    assert adapter._cache_fingerprint() != fingerprint


def test_swordfish_uses_stable_abi():
    _assert_stable(swordfish._SWORDFISH_TORCH_OPS_EXTENSION)


def test_rco_dp_uses_stable_abi():
    _assert_stable(rco_cpp.rco_dp_extension())


def test_int4_qzero_group_scaled_cpu_uses_stable_abi():
    _assert_stable(cpp._int4_qzero_group_scaled_cpu_extension())


def test_every_jit_loader_declares_the_stable_abi():
    """Any product JIT loader must opt into the stable ABI."""

    root = Path(__file__).resolve().parents[1] / "gptqmodel"
    missing = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "TorchOpsJitExtension(" not in text:
            continue
        relative = path.relative_to(root.parent).as_posix()
        for match in re.finditer(r"TorchOpsJitExtension\(", text):
            depth = 0
            index = match.end() - 1
            while index < len(text):
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            if "torch_stable_abi_target" not in text[match.end():index]:
                missing.append(relative)
                break
    assert not missing, f"JIT loaders missing torch_stable_abi_target: {missing}"


def test_nf12_torch_build_targets_the_stable_abi():
    """The vendored nf12 CMake build pins the same 2.10 stable-ABI floor."""

    source = Path(__file__).resolve().parents[1] / "gptqmodel_ext" / "nf12"
    cmake = (source / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "TORCH_TARGET_VERSION=0x020A000000000000" in cmake
    assert "USE_CUDA" in cmake

    ops = (source / "csrc" / "torch" / "ops.cpp").read_text(encoding="utf-8")
    assert "#include <torch/csrc/stable/library.h>" in ops
    assert "STABLE_TORCH_LIBRARY" in ops


def test_nf12_jit_cache_key_ignores_torch_version(monkeypatch):
    """A stable-ABI nf12 artifact must survive a torch upgrade."""

    from gptqmodel_nf12 import _jit

    source = _jit._source_root()
    monkeypatch.setattr(_jit.torch, "__version__", "2.10.0+cu128")
    fingerprint = _jit._cache_key(source, [80])
    monkeypatch.setattr(_jit.torch, "__version__", "2.15.0.dev20260928+cu130")

    assert _jit._cache_key(source, [80]) == fingerprint


def test_stable_abi_empty_uses_the_direct_allocation_shim():
    """`at::empty` must not pay the boxed aten::empty dispatcher per call.

    Every kernel materialises sentinel and output buffers on each invocation,
    so routing allocation through the boxed dispatcher made small (decode
    shaped) launches measurably slower. The direct AOTI shim keeps the same
    stable-ABI surface without that overhead; options it cannot represent
    (non-strided layout, pinned memory, an unset dtype) still fall back.
    """

    header = (
        Path(__file__).resolve().parents[1]
        / "gptqmodel_ext"
        / "torch_stable_compat.h"
    ).read_text(encoding="utf-8")
    start = header.index("inline Tensor empty(")
    end = header.index("inline Tensor zeros(", start)
    body = header[start:end]
    assert "aoti_torch_empty_strided(" in body
    assert "torch::stable::empty(" in body
