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
from torch.utils.cpp_extension import load

from ..utils._extension_loader import load_extension_module
from .util.arch_list import maybe_set_arch_list_env


extension_name = "gptqmodel_exllamav3_kernels"
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


def _build_directory() -> str | None:
    build_root = os.environ.get("GPTQMODEL_EXT_BUILD")
    if not build_root:
        return None
    return str(Path(build_root) / extension_name)


def _load_prebuilt():
    try:
        return load_extension_module(extension_name, package="gptqmodel")
    except ImportError:
        return None


def _load_jit():
    _ensure_windows_compiler()
    maybe_set_arch_list_env()

    extra_cflags = ["/O2", "/std:c++17"] if windows else ["-O3", "-std=c++17"]
    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    ]

    if ext_debug:
        if windows:
            extra_cflags.append("/Zi")
        else:
            extra_cflags.extend(["-ftime-report", "-DTORCH_USE_CUDA_DSA"])

    if torch.version.hip:
        extra_cuda_cflags.append("-DHIPBLAS_USE_HIP_HALF")

    extra_ldflags: list[str] = []
    if windows:
        extra_ldflags.append("cublas.lib")
        if sys.base_prefix != sys.prefix:
            extra_ldflags.append(f"/LIBPATH:{os.path.join(sys.base_prefix, 'libs')}")

    sources_dir = _source_root()
    return load(
        name=extension_name,
        sources=_source_files(),
        extra_include_paths=[str(sources_dir)],
        build_directory=_build_directory(),
        verbose=verbose,
        extra_ldflags=extra_ldflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
    )


exllamav3_ext = _load_prebuilt()
if exllamav3_ext is None:
    exllamav3_ext = _load_jit()
