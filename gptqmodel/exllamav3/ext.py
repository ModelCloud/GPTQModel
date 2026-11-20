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
from typing import Optional

import torch

from ..utils.cpp import (
    _TORCH_STABLE_ABI_TARGET,
    TorchOpsJitExtension,
    _jit_cxx_standard,
    cuda_include_paths_with_fallback,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)
from ..utils.kernel import KernelConfig, resolve_kernel_config
from .util.arch_list import scoped_arch_list_env


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
        "bf16_scale.cu",
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
        "quant/comp_units/exl3_comp_unit_frac.cu",
        "quant/exl3_devctx.cu",
        "quant/exl3_gemm.cu",
        "quant/exl3_kernel_map.cu",
        "quant/frac.cu",
        "quant/hadamard.cu",
        "quant/pack.cu",
        "quant/quantize.cu",
        "quant/reconstruct.cu",
        "quant/util.cu",
    ]
    if torch.version.hip is None:
        source_files += [
            "moe_coop_bindings.cpp",
            "quant/exl3_gemv.cu",
            "quant/exl3_gemv_int8.cu",
            "quant/exl3_moe_coop.cu",
            "quant/q4nx_quant.cu",
            "quant/comp_units/exl3_gemv_half_inst.cu",
        ]
        units = sources_dir / "quant/comp_units"
        for pattern in ("exl3_gemv_int8_inst_*.cu", "exl3_moe_coop_inst_*.cu"):
            source_files += [str(path.relative_to(sources_dir)) for path in sorted(units.glob(pattern))]
    else:
        # The native small-batch GEMV (decode-sized batches). The int8, MoE and
        # Q4NX quantizer kernels in this directory are CUDA-only for now.
        source_files += ["quant/exl3_gemv.cu"]
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
        flags = ["/O2", f"/std:{_jit_cxx_standard()}"]
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
        "reconstruct_frac",
        "quantize_tiles_frac",
        "pack_trellis_frac",
        "unpack_trellis_frac",
        "had_r_128",
        "hgemm",
        "bc_linear_exl3_run",
        "bc_linear_exl3_run_configured",
        "bc_linear_exl3_run_hip_gemv",
    ) + (("exl3_moe_coop", "quantize_tiles_q4nx", "pack_trellis_q4nx") if torch.version.hip is None else ()),
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
    build_env_scope=scoped_arch_list_env,
    python_abi_dependent=False,
    stage_hip_sources=True,
    torch_stable_abi_target=_TORCH_STABLE_ABI_TARGET,
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


def _linear_bitrate(value: int | float) -> int | float:
    rate = float(value)
    if rate.is_integer() and 1 <= rate <= 8:
        return int(rate)
    if rate in (1.5, 2.5, 3.5):
        return rate
    raise ValueError(f"Unsupported EXL3 bitrate {value!r}")


class _BCLinearEXL3:
    """Preserve the old binding surface while dispatching through torch.ops."""

    def __init__(
        self,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
        K: int | float,
        bias: Optional[torch.Tensor],
        mcg: bool,
        mul1: bool,
        *,
        kernel: KernelConfig | None = None,
        q4nx: bool = False,
    ):
        if not exllamav3_runtime_available():
            raise ModuleNotFoundError("ExLlamaV3 torch.ops kernels are not properly installed. Error: " + exllamav3_runtime_error())
        self.trellis = trellis
        self.suh = suh
        self.svh = svh
        self.K = _linear_bitrate(K)
        self.bias = bias
        self.mcg = bool(mcg)
        self.mul1 = bool(mul1)
        self.q4nx = q4nx
        policy = resolve_kernel_config(kernel)
        self.fp16_gemv = policy.exl3_fp16_gemv
        self.int8_gemv = policy.exl3_int8_gemv
        self.fused_gemm = policy.exl3_fused_gemm
        self._configured = not isinstance(self.K, int) or self.fp16_gemv or self.int8_gemv or self.fused_gemm

    def run(self, x: torch.Tensor, y: torch.Tensor, *, include_bias: bool = True) -> None:
        bias = self.bias if include_bias else None
        args = (self.trellis, self.suh, self.svh, self.K, bias, self.mcg, self.mul1, x, y)
        if self.q4nx:
            gemv = self.fp16_gemv or not self.fused_gemm
            _runtime_op("bc_linear_exl3_run_configured")(*args, gemv, 0, self.fused_gemm, q4nx=True)
        elif self._configured:
            _runtime_op("bc_linear_exl3_run_configured")(*args, self.fp16_gemv, self.int8_gemv, self.fused_gemm)
        else:
            _runtime_op("bc_linear_exl3_run")(*args)


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

    def quantize_tiles_q4nx(self, tiles, output, suffixes, kv, scale_search) -> None:
        _runtime_op("quantize_tiles_q4nx")(tiles, output, suffixes, kv, scale_search)

    def pack_trellis_q4nx(self, packed: torch.Tensor, suffixes: torch.Tensor, kv: int) -> None:
        _runtime_op("pack_trellis_q4nx")(packed, suffixes, kv)

    def unpack_trellis(self, unpacked: torch.Tensor, packed: torch.Tensor, K: int) -> None:
        _runtime_op("unpack_trellis")(unpacked, packed, int(K))

    def pack_trellis_frac(self, packed: torch.Tensor, unpacked: torch.Tensor, KA: int, mask: int) -> None:
        _runtime_op("pack_trellis_frac")(packed, unpacked, KA, mask)

    def unpack_trellis_frac(self, unpacked: torch.Tensor, packed: torch.Tensor, KA: int, mask: int) -> None:
        _runtime_op("unpack_trellis_frac")(unpacked, packed, KA, mask)

    def quantize_tiles_frac(
        self,
        input_tiles: torch.Tensor,
        output_tiles: torch.Tensor,
        output_indices: torch.Tensor,
        temp_costs: torch.Tensor,
        temp_edges: torch.Tensor,
        KA: int,
        mask: int,
    ) -> None:
        _runtime_op("quantize_tiles_frac")(input_tiles, output_tiles, output_indices, temp_costs, temp_edges, KA, mask)

    def pack_signs(self, packed: torch.Tensor, unpacked: torch.Tensor) -> None:
        _runtime_op("pack_signs")(packed, unpacked)

    def reconstruct(self, unpacked: torch.Tensor, packed: torch.Tensor, K: int | float, mcg: bool, mul1: bool) -> None:
        bitrate = _linear_bitrate(K)
        if isinstance(bitrate, int):
            _runtime_op("reconstruct")(unpacked, packed, bitrate, bool(mcg), bool(mul1))
        else:
            if mcg or not mul1:
                raise ValueError("EXL3 half-integer bitrates require mul1")
            _runtime_op("reconstruct_frac")(unpacked, packed, int(bitrate))

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

    def bc_linear_exl3_run_hip_gemv(
        self,
        trellis: torch.Tensor,
        suh: torch.Tensor,
        svh: torch.Tensor,
        K: int | float,
        bias: Optional[torch.Tensor],
        mcg: bool,
        mul1: bool,
        x: torch.Tensor,
        y: torch.Tensor,
        q4nx: bool = False,
    ) -> int:
        return _runtime_op("bc_linear_exl3_run_hip_gemv")(
            trellis, suh, svh, float(K), bias, bool(mcg), bool(mul1), x, y, bool(q4nx)
        )

    def exl3_moe_coop(self, *args, **kwargs) -> None:
        if torch.version.hip is not None:
            raise NotImplementedError("EXL3 routed decode requires NVIDIA CUDA")
        _runtime_op("exl3_moe_coop")(*args, **kwargs)


exllamav3_ext = _ExllamaV3TorchOpsFacade()


__all__ = [
    "exllamav3_ext",
    "exllamav3_runtime_available",
    "exllamav3_runtime_error",
    "prewarm_exllamav3_extension",
]
