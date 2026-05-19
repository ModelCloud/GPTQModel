#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OP_IR = REPO_ROOT / "gptqmodel_ext" / "cannoe" / "op_ir" / "cannoe_w4a16_matmul.json"
OVERLAY_ROOT = REPO_ROOT / "gptqmodel_ext" / "cannoe" / "ascendc"
DEFAULT_CANN_ROOTS = (
    Path("/usr/local/Ascend/cann"),
    Path("/usr/local/Ascend/cann-9.0.0-beta.2"),
    Path("/usr/local/Ascend/ascend-toolkit/latest"),
    Path("/usr/local/Ascend/ascend-toolkit"),
    Path("/usr/local/Ascend/cann-8.5.1"),
)
CANNOE_PUBLIC_STRATEGIES: dict[str, tuple[str, ...]] = {
    "manual": (),
    "staged-vector": (
        "experimental_staged_dequant",
        "experimental_cann9_vector_dequant",
    ),
    "vecout-local-a": (
        "experimental_vecout_local_a",
    ),
    "tscm-direct-multik": (
        "experimental_tscm_direct_multik",
    ),
    "tscm-direct-local-a": (
        "experimental_tscm_local_a",
    ),
    "tscm-direct-local-a-serial-k": (
        "experimental_tscm_local_a",
        "experimental_tscm_serial_k",
    ),
    "tscm-iterate-getc-diagnostic": (
        "experimental_tscm_iterate_getc_diagnostic",
    ),
    "aic-tscm-handoff": (
        "experimental_aic_tscm_handoff",
    ),
    "aic-tscm-zero-b-diagnostic": (
        "experimental_aic_tscm_zero_b_diagnostic",
    ),
    "aic-tscm-path-diagnostic": (
        "experimental_aic_tscm_path_diagnostic",
    ),
    "aic-staged-gm-visibility-diagnostic": (
        "experimental_aic_staged_gm_visibility_diagnostic",
    ),
    "aic-tscm-index-diagnostic": (
        "experimental_aic_tscm_index_diagnostic",
    ),
    "aic-tscm-syncall-diagnostic": (
        "experimental_aic_tscm_syncall_diagnostic",
    ),
    "aic-tscm-unsafe-runtime": (
        "experimental_aic_tscm_unsafe_runtime",
    ),
    "mixed-matmul-reg-diagnostic": (
        "experimental_mixed_matmul_reg_diagnostic",
    ),
}


def _apply_strategy_flags(args: argparse.Namespace) -> None:
    strategy = getattr(args, "strategy", "manual")
    if strategy not in CANNOE_PUBLIC_STRATEGIES:
        raise ValueError(f"Unknown Cannoe strategy: {strategy}")
    for flag in CANNOE_PUBLIC_STRATEGIES[strategy]:
        setattr(args, flag, True)


def _resolve_experimental_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _apply_strategy_flags(args)
    if args.experimental_vecout_local_a:
        args.experimental_vecout_runtime_handoff = True
    if args.experimental_vecout_tile_cast_dequant:
        args.experimental_vecout_local_a = True
        args.experimental_vecout_runtime_handoff = True
    if args.experimental_vecout_tile_fill_diagnostic:
        args.experimental_vecout_tile_cast_dequant = True
        args.experimental_vecout_local_a = True
        args.experimental_vecout_runtime_handoff = True
    if args.experimental_vecout_cast_scratch_probe:
        args.experimental_vecout_tile_cast_dequant = True
        args.experimental_vecout_local_a = True
        args.experimental_vecout_runtime_handoff = True
    if args.experimental_vecout_inplace_cast_dequant:
        args.experimental_vecout_local_a = True
        args.experimental_vecout_runtime_handoff = True
    if args.experimental_int4_lane_diagnostic:
        args.experimental_staged_dequant = True
        args.experimental_cann9_vector_dequant = True
    if args.experimental_mixed_entry_diagnostic:
        args.experimental_mixed_launch = True
    if args.experimental_mixed_matmul_reg_diagnostic:
        args.experimental_cube_consumer = True
        args.experimental_mixed_launch = True
    if args.experimental_vecout_runtime_handoff:
        args.experimental_staged_dequant = True
        args.experimental_cann9_vector_dequant = True
        args.experimental_vecout_consumer = True
        args.experimental_mixed_launch = True
    if args.experimental_tscm_direct_multik:
        args.experimental_tscm_direct_dequant = True
    if args.experimental_tscm_local_a:
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
    if args.experimental_tscm_iterate_getc_diagnostic:
        args.experimental_tscm_local_a = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
    if args.experimental_tscm_serial_k:
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
    if args.experimental_aic_tscm_handoff:
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_aic_tscm_zero_b_diagnostic:
        args.experimental_aic_tscm_handoff = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_aic_tscm_path_diagnostic:
        args.experimental_aic_tscm_handoff = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_aic_staged_gm_visibility_diagnostic:
        args.experimental_aic_tscm_handoff = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_aic_tscm_index_diagnostic:
        args.experimental_aic_tscm_handoff = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_aic_tscm_syncall_diagnostic:
        args.experimental_aic_tscm_handoff = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_aic_tscm_ping_diagnostic:
        args.experimental_aic_tscm_handoff = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_aic_tscm_unsafe_runtime:
        args.experimental_aic_tscm_handoff = True
        args.experimental_tscm_direct_multik = True
        args.experimental_tscm_direct_dequant = True
        args.experimental_tscm_tbuf_handoff = True
    if args.experimental_tscm_direct_dequant:
        args.experimental_staged_dequant = True
        args.experimental_cann9_vector_dequant = True
        args.experimental_tscm_runtime_handoff = True
    if args.experimental_tscm_tbuf_handoff:
        args.experimental_staged_dequant = True
        args.experimental_tscm_runtime_handoff = True
    if args.experimental_tscm_runtime_handoff:
        args.experimental_tscm_consumer = True
        args.experimental_mixed_launch = True
    if args.experimental_cann9_vector_dequant and not args.experimental_staged_dequant:
        parser.error("--experimental-cann9-vector-dequant requires --experimental-staged-dequant")
    if args.experimental_tscm_consumer and not args.experimental_staged_dequant:
        parser.error("--experimental-tscm-consumer requires --experimental-staged-dequant")
    if args.experimental_vecout_runtime_handoff and (
        args.experimental_tscm_consumer
        or args.experimental_tscm_runtime_handoff
        or args.experimental_tscm_direct_dequant
        or args.experimental_tscm_direct_multik
        or args.experimental_tscm_tbuf_handoff
        or args.experimental_aic_tscm_handoff
        or args.experimental_aic_tscm_zero_b_diagnostic
        or args.experimental_aic_tscm_path_diagnostic
        or args.experimental_aic_staged_gm_visibility_diagnostic
        or args.experimental_aic_tscm_index_diagnostic
        or args.experimental_aic_tscm_syncall_diagnostic
        or args.experimental_aic_tscm_ping_diagnostic
        or args.experimental_aic_tscm_unsafe_runtime
    ):
        parser.error("--experimental-vecout-runtime-handoff cannot be combined with TSCM runtime handoff flags")
    if args.experimental_vecout_consumer and args.experimental_tscm_consumer:
        parser.error("--experimental-vecout-consumer and --experimental-tscm-consumer are mutually exclusive")
    if args.experimental_vecout_inplace_cast_dequant and args.experimental_vecout_tile_cast_dequant:
        parser.error("--experimental-vecout-inplace-cast-dequant and --experimental-vecout-tile-cast-dequant are mutually exclusive")
    if args.experimental_vecout_consumer:
        args.experimental_cube_consumer = True
    if args.experimental_tscm_consumer:
        args.experimental_cube_consumer = True
    if args.experimental_mixed_aiv_baseline:
        if args.experimental_cube_consumer:
            parser.error("--experimental-mixed-aiv-baseline cannot be combined with --experimental-cube-consumer")
        args.experimental_mixed_launch = True
    if args.experimental_mixed_entry_diagnostic and args.experimental_cube_consumer:
        parser.error("--experimental-mixed-entry-diagnostic cannot be combined with --experimental-cube-consumer")
    if args.experimental_mixed_entry_diagnostic and args.experimental_mixed_matmul_reg_diagnostic:
        parser.error("--experimental-mixed-entry-diagnostic and --experimental-mixed-matmul-reg-diagnostic are mutually exclusive")


def _find_cann_root() -> Path | None:
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_AICPU_PATH", "BASE_LIBS_PATH"):
        raw = os.environ.get(env_name)
        if raw:
            path = Path(raw)
            if path.exists():
                return path
    for path in DEFAULT_CANN_ROOTS:
        if path.exists():
            return path
    return None


def _find_msopgen(explicit: str | None) -> str:
    if explicit:
        return explicit

    env_msopgen = os.environ.get("MSOPGEN")
    if env_msopgen:
        return env_msopgen

    cann_root = _find_cann_root()
    candidates: list[Path] = []
    if cann_root is not None:
        candidates.extend(
            [
                cann_root / "bin" / "msopgen",
                cann_root / "python" / "site-packages" / "bin" / "msopgen",
                cann_root / "tools" / "msopgen" / "bin" / "msopgen",
            ]
        )
    candidates.extend(Path(p) / "msopgen" for p in os.environ.get("PATH", "").split(os.pathsep) if p)

    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)

    resolved = shutil.which("msopgen")
    if resolved:
        return resolved
    raise FileNotFoundError("Could not find msopgen. Set MSOPGEN or ASCEND_HOME_PATH.")


def _command_for_entrypoint(entrypoint: str) -> list[str]:
    path = Path(entrypoint)
    if not path.exists():
        return [entrypoint]
    try:
        with path.open("rb") as handle:
            prefix = handle.read(256)
    except OSError:
        return [entrypoint]
    if prefix.startswith(b"#!") and b"python" in prefix.splitlines()[0].lower():
        return [sys.executable, str(path)]
    return [entrypoint]


def _cann_env(cann_root: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if cann_root is None:
        return env
    env.setdefault("ASCEND_HOME_PATH", str(cann_root))
    python_paths = [
        cann_root / "python" / "site-packages",
        cann_root / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe",
    ]
    existing = [part for part in env.get("PYTHONPATH", "").split(os.pathsep) if part]
    for path in reversed(python_paths):
        path_text = str(path)
        if path.exists() and path_text not in existing:
            existing.insert(0, path_text)
    if existing:
        env["PYTHONPATH"] = os.pathsep.join(existing)
    return env


def _copy_overlay(output: Path) -> None:
    for rel in (
        Path("op_host") / "cannoe_w4_a16_matmul.cpp",
        Path("op_host") / "cannoe_w4_a16_matmul_tiling.h",
        Path("op_host") / "cannoe_w4_a16_matmul_tiling_key.h",
        Path("op_kernel") / "cannoe_w4_a16_matmul.cpp",
    ):
        src = OVERLAY_ROOT / rel
        dst = output / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(
        OVERLAY_ROOT / "op_host" / "cannoe_w4_a16_matmul_tiling_key.h",
        output / "op_kernel" / "cannoe_w4_a16_matmul_tiling_key.h",
    )


def _force_compute_unit(output: Path, compute_unit: str) -> None:
    presets_path = output / "CMakePresets.json"
    if not presets_path.exists():
        raise FileNotFoundError(presets_path)
    presets = json.loads(presets_path.read_text())
    for preset in presets.get("configurePresets", []):
        cache = preset.setdefault("cacheVariables", {})
        ascend_compute_unit = cache.get("ASCEND_COMPUTE_UNIT")
        if isinstance(ascend_compute_unit, dict):
            ascend_compute_unit["value"] = compute_unit
        else:
            cache["ASCEND_COMPUTE_UNIT"] = {"type": "STRING", "value": compute_unit}
    presets_path.write_text(json.dumps(presets, indent=4) + "\n")


def _enable_kernel_define(output: Path, define: str) -> None:
    cmake_path = output / "op_kernel" / "CMakeLists.txt"
    if not cmake_path.exists():
        raise FileNotFoundError(cmake_path)
    text = cmake_path.read_text()
    option = f"-D{define}=1"
    lines = text.splitlines(keepends=True)

    cmake9_match = re.search(r"^npu_op_kernel_sources\(\s*(\S+)", text, flags=re.MULTILINE)
    if cmake9_match is not None:
        target_name = cmake9_match.group(1)
        cmake9_prefix = "npu_op_kernel_options("
        for index, line in enumerate(lines):
            if line.startswith(cmake9_prefix):
                if option in line:
                    return
                lines[index] = line.replace(")\n", f" {option})\n", 1)
                cmake_path.write_text("".join(lines))
                return
        marker = cmake9_match.group(0)
        text = text.replace(
            marker,
            f"npu_op_kernel_options({target_name} ALL OPTIONS {option})\n\n{marker}",
            1,
        )
        cmake_path.write_text(text)
        return

    experimental_prefix = "add_ops_compile_options(ALL OPTIONS -DCANNOE_EXPERIMENTAL_"
    for index, line in enumerate(lines):
        if line.startswith(experimental_prefix):
            if option in line:
                return
            lines[index] = line.replace(")\n", f" {option})\n", 1)
            cmake_path.write_text("".join(lines))
            return
    if option in text:
        return
    marker = "add_kernels_compile()\n"
    if marker in text:
        text = text.replace(marker, f"add_ops_compile_options(ALL OPTIONS {option})\n{marker}", 1)
        cmake_path.write_text(text)
        return
    raise RuntimeError(f"Could not find a kernel compile marker in {cmake_path}.")


def _enable_host_define(output: Path, define: str) -> None:
    cmake_path = output / "op_host" / "CMakeLists.txt"
    if not cmake_path.exists():
        raise FileNotFoundError(cmake_path)
    text = cmake_path.read_text()
    option = f"-D{define}=1"
    if option in text:
        return
    prefix = "add_compile_options("
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith(prefix):
            lines[index] = line.replace(")\n", f" {option})\n", 1)
            cmake_path.write_text("".join(lines))
            return
    marker = "aux_source_directory("
    if marker not in text:
        raise RuntimeError(f"Could not find a host compile marker in {cmake_path}.")
    text = text.replace(marker, f"add_compile_options({option})\n\n{marker}", 1)
    cmake_path.write_text(text)


def _run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and build the Cannoe Ascend C custom op.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/cannoe_w4a16_op"))
    parser.add_argument("--msopgen", default=None)
    parser.add_argument("--cann-root", type=Path, default=None)
    parser.add_argument("--target", default=None, help="Optional generated build.sh target, for example install.")
    parser.add_argument("--compute-unit", default="ascend910b", help="ASCEND_COMPUTE_UNIT for generated CMake.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before generating.")
    parser.add_argument("--no-build", action="store_true", help="Only run msopgen and overlay repo sources.")
    parser.add_argument(
        "--strategy",
        choices=tuple(CANNOE_PUBLIC_STRATEGIES),
        default="manual",
        help="Named Cannoe public-API strategy that expands to the required experimental flags.",
    )
    parser.add_argument(
        "--experimental-staged-dequant",
        action="store_true",
        help="Compile guarded staged-dequant workspace hooks for Cannoe fused-kernel bring-up.",
    )
    parser.add_argument(
        "--experimental-cube-consumer",
        action="store_true",
        help="Compile guarded Matmul/Cube consumer registration for Cannoe fused-kernel bring-up.",
    )
    parser.add_argument(
        "--experimental-mixed-launch",
        action="store_true",
        help=(
            "Compile the guarded Cannoe kernel as a MIX_AIC_1_2 AIC/AIV launch. "
            "This implies --experimental-cube-consumer unless --experimental-mixed-aiv-baseline is used."
        ),
    )
    parser.add_argument(
        "--experimental-mixed-aiv-baseline",
        action="store_true",
        help=(
            "Compile a MIX_AIC_1_2 launch that keeps only the AIV scalar visible-output path. "
            "This is an isolation probe and does not register the CANN Matmul/KFC consumer."
        ),
    )
    parser.add_argument(
        "--experimental-mixed-entry-diagnostic",
        action="store_true",
        help=(
            "Compile a minimal MIX_AIC_1_2 entry probe that writes tiling/path markers and returns. "
            "This isolates repeat-launch lifecycle from staging workspace and Matmul/KFC registration."
        ),
    )
    parser.add_argument(
        "--experimental-mixed-matmul-reg-diagnostic",
        action="store_true",
        help=(
            "Compile a MIX_AIC_1_2 probe that clears Matmul/KFC workspace, registers the CANN Matmul "
            "object, writes path markers, and returns before SetTensor/IterateAll. This isolates "
            "Matmul registration lifecycle from the live local-tile handoff."
        ),
    )
    parser.add_argument(
        "--experimental-cann9-vector-dequant",
        action="store_true",
        help=(
            "Compile the guarded CANN 9 public c_api asc_int42half UB dequant producer. "
            "Requires --experimental-staged-dequant."
        ),
    )
    parser.add_argument(
        "--experimental-vecout-consumer",
        action="store_true",
        help=(
            "Compile the guarded Matmul consumer probe with B_TYPE at TPosition::VECOUT. "
            "This implies --experimental-cube-consumer."
        ),
    )
    parser.add_argument(
        "--experimental-vecout-runtime-handoff",
        action="store_true",
        help=(
            "Compile the guarded mixed AIV/AIC runtime probe that dequantizes INT4 B tiles into "
            "UB/VECOUT and hands them to Matmul without materializing a full FP16 weight tile. "
            "This implies --experimental-staged-dequant, --experimental-cann9-vector-dequant, "
            "--experimental-vecout-consumer, and --experimental-mixed-launch."
        ),
    )
    parser.add_argument(
        "--experimental-vecout-local-a",
        action="store_true",
        help=(
            "Compile the guarded VecOut runtime variant that stages the M x baseK activation tile "
            "into UB/VECOUT so Cube can process multiple rows per call. This implies "
            "--experimental-vecout-runtime-handoff."
        ),
    )
    parser.add_argument(
        "--experimental-int4-lane-diagnostic",
        action="store_true",
        help=(
            "Compile an opt-in diagnostic path that writes CANN int4b_t vector-cast lanes and the "
            "current scalar unpack lanes into y for packed-layout bring-up. This implies "
            "--experimental-staged-dequant and --experimental-cann9-vector-dequant."
        ),
    )
    parser.add_argument(
        "--experimental-vecout-tile-cast-dequant",
        action="store_true",
        help=(
            "Compile the guarded VecOut local-A handoff variant that vector-decodes each packed "
            "INT4 B tile with AscendC Cast<half, int4b_t> before scaling and handing the tile to Cube. "
            "This implies --experimental-vecout-local-a."
        ),
    )
    parser.add_argument(
        "--experimental-vecout-tile-fill-diagnostic",
        action="store_true",
        help=(
            "Compile an opt-in diagnostic that compares scalar B-tile fill with Cast-based B-tile fill "
            "before the Matmul handoff. This implies --experimental-vecout-tile-cast-dequant."
        ),
    )
    parser.add_argument(
        "--experimental-vecout-cast-scratch-probe",
        action="store_true",
        help=(
            "Compile a VecOut local-A probe that executes Cast into scratch, discards it, then "
            "fills the Cube B tile through the scalar path. This implies --experimental-vecout-tile-cast-dequant."
        ),
    )
    parser.add_argument(
        "--experimental-vecout-inplace-cast-dequant",
        action="store_true",
        help=(
            "Compile the guarded VecOut local-A handoff variant that Cast-decodes INT4 directly "
            "into the VECOUT B tile, then applies scale/offset in place before Cube consumes it. "
            "This implies --experimental-vecout-local-a."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-consumer",
        action="store_true",
        help=(
            "Compile the guarded Matmul consumer probe with B_TYPE at TPosition::TSCM/CubeFormat::NZ "
            "and the staged GM-to-TSCM tile handoff helper. This implies --experimental-cube-consumer "
            "and requires --experimental-staged-dequant."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-runtime-handoff",
        action="store_true",
        help=(
            "Compile the guarded mixed AIV/AIC runtime probe that stages a single-K B tile, "
            "copies it to TSCM/NZ, and hands it to Matmul. This implies --experimental-tscm-consumer "
            "and --experimental-mixed-launch."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-direct-dequant",
        action="store_true",
        help=(
            "Compile the guarded TSCM runtime probe that writes the vector-dequantized B tile into UB "
            "and copies UB directly to TSCM/NZ, bypassing the staged FP16 GM tile. This implies "
            "--experimental-staged-dequant, --experimental-cann9-vector-dequant, and "
            "--experimental-tscm-runtime-handoff."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-direct-multik",
        action="store_true",
        help=(
            "Compile the guarded TSCM direct-dequant runtime probe for K that is an integer multiple "
            "of base_k. This iterates K tiles and accumulates into the output. It implies "
            "--experimental-tscm-direct-dequant."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-local-a",
        action="store_true",
        help=(
            "Compile the TSCM direct multi-K probe with the live A tile staged into VECOUT before "
            "Matmul. This preserves the correct row stride for M>1 while keeping B in TSCM/NZ."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-iterate-getc-diagnostic",
        action="store_true",
        help=(
            "Compile the TSCM direct local-A diagnostic that calls Matmul Iterate(enPartialSum) "
            "for each K tile and emits one final GetTensorC. This probes CO1 partial accumulation "
            "without writing partial C tiles through GM after every base_k tile."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-serial-k",
        action="store_true",
        help=(
            "Compile the TSCM direct multi-K diagnostic that waits for each Cube K tile before filling and "
            "loading the next B tile. This isolates async TSCM B-slot lifetime from partial-C accumulation."
        ),
    )
    parser.add_argument(
        "--experimental-tscm-tbuf-handoff",
        action="store_true",
        help=(
            "Compile the guarded TSCM handoff variant that uses a raw TBuf<TPosition::TSCM> L1 buffer "
            "instead of the queue-style TSCM object. This follows CANN transformer-kernel TSCM full-load "
            "patterns and implies --experimental-tscm-runtime-handoff."
        ),
    )
    parser.add_argument(
        "--experimental-aic-tscm-handoff",
        action="store_true",
        help=(
            "Compile the guarded explicit AIV producer / AIC MatmulImpl consumer probe. The AIV side "
            "dequantizes one INT4 B tile into UB, copies that tile into TSCM/NZ, signals AIC, and the "
            "AIC side copies only the live A tile into TSCM before Cube compute. This implies "
            "--experimental-tscm-direct-multik and remains a bring-up path."
        ),
    )
    parser.add_argument(
        "--experimental-aic-tscm-zero-b-diagnostic",
        action="store_true",
        help=(
            "Compile the AIC/TSCM handoff probe with the AIV producer writing zero B tiles instead of "
            "dequantized INT4 values. This isolates cross-core TSCM/Cube lifetime faults from dequant "
            "and implies --experimental-aic-tscm-handoff."
        ),
    )
    parser.add_argument(
        "--experimental-aic-tscm-path-diagnostic",
        action="store_true",
        help=(
            "Compile the AIC/TSCM handoff probe as a path marker that writes selected tiling fields into "
            "the output and returns before Matmul. This verifies whether the guarded mixed-launch branch "
            "is actually reached."
        ),
    )
    parser.add_argument(
        "--experimental-aic-staged-gm-visibility-diagnostic",
        action="store_true",
        help=(
            "Compile the AIC/TSCM bring-up probe that has AIV entries write marker values into the "
            "bounded GM workspace, SyncAll, then has AIC entries read the staged markers. This verifies "
            "the mixed-core workspace visibility lifecycle before another L1/TSCM handoff."
        ),
    )
    parser.add_argument(
        "--experimental-aic-tscm-index-diagnostic",
        action="store_true",
        help=(
            "Compile the AIC/TSCM handoff probe as a no-wait mixed-entry index marker. "
            "This reports AIC/AIV block and sub-block numbering before cross-core handoff tuning."
        ),
    )
    parser.add_argument(
        "--experimental-aic-tscm-syncall-diagnostic",
        action="store_true",
        help=(
            "Compile the AIC/TSCM handoff probe as a CANN SyncAll<false>() mixed-core marker. "
            "This validates CANN's built-in AIC/AIV rendezvous before manual tile handoff tuning."
        ),
    )
    parser.add_argument(
        "--experimental-aic-tscm-ping-diagnostic",
        action="store_true",
        help=(
            "Compile the AIC/TSCM handoff probe as a minimal cross-core flag ping. "
            "This isolates CrossCoreSetFlag/WaitFlag from TSCM data movement and Matmul."
        ),
    )
    parser.add_argument(
        "--experimental-aic-tscm-unsafe-runtime",
        action="store_true",
        help=(
            "Allow the guarded AIC/TSCM handoff to run for isolated crash/debug probes. Normal "
            "aic-tscm-handoff builds compile the code but keep staged runtime selection disabled because "
            "current wide-N and multi-K probes hit AIC MPU faults."
        ),
    )
    args = parser.parse_args()
    _resolve_experimental_flags(args, parser)

    output = args.output.resolve()
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    cann_root = args.cann_root or _find_cann_root()
    msopgen = _find_msopgen(args.msopgen)
    env = _cann_env(cann_root)
    _run(
        _command_for_entrypoint(msopgen)
        + [
            "gen",
            "-i",
            str(OP_IR),
            "-f",
            "pytorch",
            "-c",
            "ai_core-ascend910b",
            "-lan",
            "cpp",
            "-out",
            str(output),
        ],
        env=env,
    )
    _copy_overlay(output)
    _force_compute_unit(output, args.compute_unit)
    if args.experimental_staged_dequant:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_STAGED_DEQUANT")
    if args.experimental_cube_consumer or (
        args.experimental_mixed_launch
        and not args.experimental_mixed_aiv_baseline
        and not args.experimental_mixed_entry_diagnostic
    ):
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_CUBE_CONSUMER")
    if args.experimental_mixed_launch:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_MIXED_LAUNCH")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_MIXED_LAUNCH")
    if args.experimental_mixed_aiv_baseline:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_MIXED_AIV_BASELINE")
    if args.experimental_mixed_entry_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_MIXED_ENTRY_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_MIXED_ENTRY_DIAGNOSTIC")
    if args.experimental_mixed_matmul_reg_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_MIXED_MATMUL_REG_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_MIXED_MATMUL_REG_DIAGNOSTIC")
    if args.experimental_cann9_vector_dequant:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_CANN9_VECTOR_DEQUANT")
    if args.experimental_vecout_consumer:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_VECOUT_CONSUMER")
    if args.experimental_vecout_runtime_handoff:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF")
    if args.experimental_vecout_local_a:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_VECOUT_LOCAL_A")
    if args.experimental_vecout_tile_cast_dequant:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_VECOUT_TILE_CAST_DEQUANT")
    if args.experimental_vecout_tile_fill_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_VECOUT_TILE_FILL_DIAGNOSTIC")
    if args.experimental_vecout_cast_scratch_probe:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_VECOUT_CAST_SCRATCH_PROBE")
    if args.experimental_vecout_inplace_cast_dequant:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_VECOUT_INPLACE_CAST_DEQUANT")
    if args.experimental_int4_lane_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_INT4_LANE_DIAGNOSTIC")
    if args.experimental_tscm_consumer:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_CONSUMER")
    if args.experimental_tscm_runtime_handoff:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    if args.experimental_tscm_direct_dequant:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    if args.experimental_tscm_direct_multik:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_DIRECT_MULTIK")
    if args.experimental_tscm_local_a:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_LOCAL_A")
    if args.experimental_tscm_iterate_getc_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_ITERATE_GETC_DIAGNOSTIC")
    if args.experimental_tscm_serial_k:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_SERIAL_K")
    if args.experimental_tscm_tbuf_handoff:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_TSCM_TBUF_HANDOFF")
    if args.experimental_aic_tscm_handoff:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_HANDOFF")
    if args.experimental_aic_tscm_zero_b_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_ZERO_B_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_ZERO_B_DIAGNOSTIC")
    if args.experimental_aic_tscm_path_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_PATH_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_PATH_DIAGNOSTIC")
    if args.experimental_aic_staged_gm_visibility_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_STAGED_GM_VISIBILITY_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_STAGED_GM_VISIBILITY_DIAGNOSTIC")
    if args.experimental_aic_tscm_index_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_INDEX_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_INDEX_DIAGNOSTIC")
    if args.experimental_aic_tscm_syncall_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_SYNCALL_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_SYNCALL_DIAGNOSTIC")
    if args.experimental_aic_tscm_ping_diagnostic:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_PING_DIAGNOSTIC")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_PING_DIAGNOSTIC")
    if args.experimental_aic_tscm_unsafe_runtime:
        _enable_kernel_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME")
        _enable_host_define(output, "CANNOE_EXPERIMENTAL_AIC_TSCM_UNSAFE_RUNTIME")

    if args.no_build:
        print(f"Generated project with Cannoe overlay at {output}")
        return 0

    if shutil.which("cmake") is None:
        raise FileNotFoundError("CMake is required by the msopgen build.sh but was not found on PATH.")

    build_cmd = ["bash", str(output / "build.sh")]
    if args.target:
        build_cmd.append(args.target)
    _run(build_cmd, cwd=output, env=env)
    print(f"Built Cannoe Ascend C project at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
