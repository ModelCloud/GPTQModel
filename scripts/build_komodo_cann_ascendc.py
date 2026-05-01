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
OP_IR = REPO_ROOT / "gptqmodel_ext" / "komodo_cann" / "op_ir" / "komodo_cann_w4a16_matmul.json"
OVERLAY_ROOT = REPO_ROOT / "gptqmodel_ext" / "komodo_cann" / "ascendc"
DEFAULT_CANN_ROOTS = (
    Path("/usr/local/Ascend/cann"),
    Path("/usr/local/Ascend/cann-9.0.0-beta.2"),
    Path("/usr/local/Ascend/ascend-toolkit/latest"),
    Path("/usr/local/Ascend/ascend-toolkit"),
    Path("/usr/local/Ascend/cann-8.5.1"),
)


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
        Path("op_host") / "komodo_cann_w4_a16_matmul.cpp",
        Path("op_host") / "komodo_cann_w4_a16_matmul_tiling.h",
        Path("op_host") / "komodo_cann_w4_a16_matmul_tiling_key.h",
        Path("op_kernel") / "komodo_cann_w4_a16_matmul.cpp",
    ):
        src = OVERLAY_ROOT / rel
        dst = output / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(
        OVERLAY_ROOT / "op_host" / "komodo_cann_w4_a16_matmul_tiling_key.h",
        output / "op_kernel" / "komodo_cann_w4_a16_matmul_tiling_key.h",
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

    experimental_prefix = "add_ops_compile_options(ALL OPTIONS -DKOMODO_CANN_EXPERIMENTAL_"
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
    parser = argparse.ArgumentParser(description="Generate and build the Komodo-CANN Ascend C custom op.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/komodo_cann_w4a16_op"))
    parser.add_argument("--msopgen", default=None)
    parser.add_argument("--cann-root", type=Path, default=None)
    parser.add_argument("--target", default=None, help="Optional generated build.sh target, for example install.")
    parser.add_argument("--compute-unit", default="ascend910b", help="ASCEND_COMPUTE_UNIT for generated CMake.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before generating.")
    parser.add_argument("--no-build", action="store_true", help="Only run msopgen and overlay repo sources.")
    parser.add_argument(
        "--experimental-staged-dequant",
        action="store_true",
        help="Compile guarded staged-dequant workspace hooks for Komodo-CANN fused-kernel bring-up.",
    )
    parser.add_argument(
        "--experimental-cube-consumer",
        action="store_true",
        help="Compile guarded Matmul/Cube consumer registration for Komodo-CANN fused-kernel bring-up.",
    )
    parser.add_argument(
        "--experimental-mixed-launch",
        action="store_true",
        help=(
            "Compile the guarded Komodo-CANN kernel as a MIX_AIC_1_2 AIC/AIV launch. "
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
    args = parser.parse_args()
    if args.experimental_vecout_runtime_handoff:
        args.experimental_staged_dequant = True
        args.experimental_cann9_vector_dequant = True
        args.experimental_vecout_consumer = True
        args.experimental_mixed_launch = True
    if args.experimental_tscm_direct_multik:
        args.experimental_tscm_direct_dequant = True
    if args.experimental_tscm_direct_dequant:
        args.experimental_staged_dequant = True
        args.experimental_cann9_vector_dequant = True
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
    ):
        parser.error("--experimental-vecout-runtime-handoff cannot be combined with TSCM runtime handoff flags")
    if args.experimental_vecout_consumer and args.experimental_tscm_consumer:
        parser.error("--experimental-vecout-consumer and --experimental-tscm-consumer are mutually exclusive")
    if args.experimental_vecout_consumer:
        args.experimental_cube_consumer = True
    if args.experimental_tscm_consumer:
        args.experimental_cube_consumer = True
    if args.experimental_mixed_aiv_baseline:
        if args.experimental_cube_consumer:
            parser.error("--experimental-mixed-aiv-baseline cannot be combined with --experimental-cube-consumer")
        args.experimental_mixed_launch = True

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
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_STAGED_DEQUANT")
    if args.experimental_cube_consumer or (args.experimental_mixed_launch and not args.experimental_mixed_aiv_baseline):
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_CUBE_CONSUMER")
    if args.experimental_mixed_launch:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH")
        _enable_host_define(output, "KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH")
    if args.experimental_mixed_aiv_baseline:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_MIXED_AIV_BASELINE")
    if args.experimental_cann9_vector_dequant:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_CANN9_VECTOR_DEQUANT")
    if args.experimental_vecout_consumer:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_VECOUT_CONSUMER")
    if args.experimental_vecout_runtime_handoff:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_VECOUT_RUNTIME_HANDOFF")
    if args.experimental_tscm_consumer:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_TSCM_CONSUMER")
    if args.experimental_tscm_runtime_handoff:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_TSCM_RUNTIME_HANDOFF")
    if args.experimental_tscm_direct_dequant:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_DEQUANT")
    if args.experimental_tscm_direct_multik:
        _enable_kernel_define(output, "KOMODO_CANN_EXPERIMENTAL_TSCM_DIRECT_MULTIK")

    if args.no_build:
        print(f"Generated project with Komodo-CANN overlay at {output}")
        return 0

    if shutil.which("cmake") is None:
        raise FileNotFoundError("CMake is required by the msopgen build.sh but was not found on PATH.")

    build_cmd = ["bash", str(output / "build.sh")]
    if args.target:
        build_cmd.append(args.target)
    _run(build_cmd, cwd=output, env=env)
    print(f"Built Komodo-CANN Ascend C project at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
