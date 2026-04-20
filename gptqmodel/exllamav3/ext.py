# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Portions of this file are adapted from turboderp-org/exllamav3.
# Credits: TurboDerp / ExLlamaV3 contributors.

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from ..utils.cpp import (
    TorchOpsJitExtension,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)
from .util.arch_list import maybe_set_arch_list_env


extension_name = "gptqmodel_exllamav3_ops"
verbose = str(os.environ.get("GPTQMODEL_EXT_VERBOSE", "")).strip().lower() not in {"", "0", "false", "off", "no"}
ext_debug = str(os.environ.get("GPTQMODEL_EXT_DEBUG", "")).strip().lower() in {"1", "true", "on", "yes"}
windows = os.name == "nt"


def _find_msvc() -> str | None:
    program_files_x64 = os.environ["ProgramW6432"]
    program_files_x86 = os.environ["ProgramFiles(x86)"]
    msvc_dirs = [
        a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\\Tools\\MSVC\\"
        for b in ["2022", "2019", "2017"]
        for a in [program_files_x64, program_files_x86]
        for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
    ]

    for msvc_dir in msvc_dirs:
        if not os.path.exists(msvc_dir):
            continue
        versions = sorted(os.listdir(msvc_dir), reverse=True)
        for version in versions:
            compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
            if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                return compiler_dir
    return None


def _ensure_windows_compiler() -> None:
    if not windows:
        return

    import subprocess

    try:
        subprocess.check_output(["where", "/Q", "cl"])
    except subprocess.CalledProcessError:
        cl_path = _find_msvc()
        if cl_path:
            if verbose:
                print(" -- Injected compiler path:", cl_path)
            os.environ["path"] += ";" + cl_path
        else:
            print(" !! Unable to find cl.exe; EXL3 compilation will probably fail", file=sys.stderr)


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "exllamav3"


def _source_files() -> list[str]:
    sources_dir = _source_root()
    source_files = [
        "bindings.cpp",
        "hadamard.cpp",
        "hgemm.cu",
        "libtorch/linear.cpp",
        "quant/comp_units/exl3_comp_unit_1.cu",
        "quant/comp_units/exl3_comp_unit_2.cu",
        "quant/comp_units/exl3_comp_unit_3.cu",
        "quant/comp_units/exl3_comp_unit_4.cu",
        "quant/comp_units/exl3_comp_unit_5.cu",
        "quant/comp_units/exl3_comp_unit_6.cu",
        "quant/comp_units/exl3_comp_unit_7.cu",
        "quant/comp_units/exl3_comp_unit_8.cu",
        "quant/exl3_devctx.cu",
        "quant/exl3_gemm.cu",
        "quant/exl3_kernel_map.cu",
        "quant/hadamard.cu",
        "quant/pack.cu",
        "quant/quantize.cu",
        "quant/reconstruct.cu",
        "quant/util.cu",
    ]
    return [str((sources_dir / path).resolve()) for path in source_files]


def _exllamav3_required_cuda_headers() -> tuple[str, ...]:
    return ("cusparse.h",)


def _exllamav3_include_paths() -> list[str]:
    return cuda_include_paths_with_fallback(
        [str(_source_root())],
        required_header_names=_exllamav3_required_cuda_headers(),
    )


def _extra_cflags() -> list[str]:
    if windows:
        flags = ["/O2", "/std:c++17"]
    else:
        flags = default_jit_cflags(opt_level="O2")

    if ext_debug:
        if windows:
            flags.append("/Zi")
        else:
            flags.extend(["-ftime-report", "-DTORCH_USE_CUDA_DSA"])
    return flags


def _extra_cuda_cflags() -> list[str]:
    flags = default_jit_cuda_cflags(
        opt_level="O2",
        include_abi=not windows,
        include_lineinfo=True,
    )
    if torch.version.hip:
        flags.append("-DHIPBLAS_USE_HIP_HALF")
    return flags


def _extra_ldflags() -> list[str]:
    if not windows:
        return []
    flags = ["cublas.lib"]
    if sys.base_prefix != sys.prefix:
        flags.append(f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}")
    return flags


def _prepare_build_env() -> None:
    _ensure_windows_compiler()
    maybe_set_arch_list_env()


# Shared singleton so EXL3 quantization and inference both reuse the same
# torch.ops cache and first-use build policy.
_EXLLAMAV3_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=extension_name,
    namespace="gptqmodel_exllamav3",
    required_ops=(
        "had_paley",
        "had_paley2",
        "quantize_tiles",
        "pack_trellis",
        "unpack_trellis",
        "pack_signs",
        "reconstruct",
        "had_r_128",
        "hgemm",
        "bc_linear_exl3_run",
    ),
    sources=_source_files,
    build_root_env="GPTQMODEL_EXLLAMAV3_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("exllamav3"),
    display_name="ExLlamaV3",
    extra_cflags=_extra_cflags,
    extra_cuda_cflags=_extra_cuda_cflags,
    extra_include_paths=_exllamav3_include_paths,
    extra_ldflags=_extra_ldflags,
    force_rebuild_env="GPTQMODEL_EXLLAMAV3_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def exllamav3_runtime_available() -> bool:
    _prepare_build_env()
    return _extension_api().is_available("exllamav3")


def exllamav3_runtime_error() -> str:
    extension_api = _extension_api()
    _prepare_build_env()
    if extension_api.is_available("exllamav3"):
        return ""
    return (
        extension_api.error("exllamav3")
        or "ExLlamaV3 CUDA runtime unavailable."
    )


def prewarm_exllamav3_extension() -> bool:
    _prepare_build_env()
    return _extension_api().load(name="exllamav3")["exllamav3"]


def _runtime_op(name: str):
    _prepare_build_env()
    return _extension_api().op("exllamav3", name)


class _BCLinearEXL3:
    """Preserve the old binding surface while dispatching through torch.ops."""

    def __init__(
        self,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
        K: int,
        bias: Optional[torch.Tensor],
        mcg: bool,
        mul1: bool,
        xh: torch.Tensor,
    ):
        if not exllamav3_runtime_available():
            raise ModuleNotFoundError("ExLlamaV3 torch.ops kernels are not properly installed. Error: " + exllamav3_runtime_error())
        self.trellis = trellis
        self.suh = suh
        self.svh = svh
        self.K = int(K)
        self.bias = bias
        self.mcg = bool(mcg)
        self.mul1 = bool(mul1)
        self.xh = xh

    def run(self, x: torch.Tensor, y: torch.Tensor) -> None:
        _runtime_op("bc_linear_exl3_run")(
            self.trellis,
            self.suh,
            self.svh,
            self.K,
            self.bias,
            self.mcg,
            self.mul1,
            self.xh,
            x,
            y,
        )


class _ExllamaV3TorchOpsFacade:
    """Facade that mirrors the old pybind module API over torch.ops."""

    BC_LinearEXL3 = _BCLinearEXL3

    def had_paley(self, h: torch.Tensor) -> None:
        _runtime_op("had_paley")(h)

    def had_paley2(self, h: torch.Tensor) -> None:
        _runtime_op("had_paley2")(h)

    def quantize_tiles(
        self,
        input_tiles: torch.Tensor,
        output_tiles: torch.Tensor,
        output_indices: torch.Tensor,
        temp_costs: torch.Tensor,
        temp_edges: torch.Tensor,
        K: int,
        mcg: bool,
        mul1: bool,
    ) -> None:
        _runtime_op("quantize_tiles")(
            input_tiles,
            output_tiles,
            output_indices,
            temp_costs,
            temp_edges,
            int(K),
            bool(mcg),
            bool(mul1),
        )

    def pack_trellis(self, packed: torch.Tensor, unpacked: torch.Tensor, K: int) -> None:
        _runtime_op("pack_trellis")(packed, unpacked, int(K))

    def unpack_trellis(self, unpacked: torch.Tensor, packed: torch.Tensor, K: int) -> None:
        _runtime_op("unpack_trellis")(unpacked, packed, int(K))

    def pack_signs(self, packed: torch.Tensor, unpacked: torch.Tensor) -> None:
        _runtime_op("pack_signs")(packed, unpacked)

    def reconstruct(self, unpacked: torch.Tensor, packed: torch.Tensor, K: int, mcg: bool, mul1: bool) -> None:
        _runtime_op("reconstruct")(unpacked, packed, int(K), bool(mcg), bool(mul1))

    def had_r_128(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        pre_scale: Optional[torch.Tensor],
        post_scale: Optional[torch.Tensor],
        scale: float,
    ) -> None:
        _runtime_op("had_r_128")(input, output, pre_scale, post_scale, float(scale))

    def hgemm(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
        _runtime_op("hgemm")(a, b, c)


exllamav3_ext = _ExllamaV3TorchOpsFacade()


__all__ = [
    "exllamav3_ext",
    "exllamav3_runtime_available",
    "exllamav3_runtime_error",
    "prewarm_exllamav3_extension",
]
