#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Verify native window ABI on a bound real module and export a ZML execution fixture."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from gpu_idle_preflight import add_gpu_idle_preflight_args, bootstrap_gpu_idle_preflight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    idle = bootstrap_gpu_idle_preflight()
    parser = argparse.ArgumentParser(description=__doc__)
    add_gpu_idle_preflight_args(parser)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--zml-m",
        type=int,
        default=33,
        choices=(1, 33, 128, 2048, 8192),
        help="row count used by the exported StableHLO/ZML fixture",
    )
    parser.add_argument(
        "--max-recovery-overhead-percent",
        type=float,
        default=None,
        help=(
            "optional hard gate copied into the fixture manifest; the ZML "
            "tuner rejects rank8 geometries above this matched off/on cost"
        ),
    )
    parser.add_argument(
        "--quality-mode",
        choices=("fast", "balanced", "quality"),
        default="fast",
        help="arithmetic-quality policy used for the generated off/on fixture",
    )
    args = parser.parse_args()
    import torch

    from gptqmodel.quantization.qvq_rank8 import (
        P32WindowConfig,
        load_window_package,
        prepare_rank8,
    )
    from gptqmodel.utils.qvq_cuda import _pgc16_levels
    from gptqmodel.utils.qvq_window_abi import (
        WindowConfig,
        native_window_library,
        native_window_linear,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    source_package = torch.load(args.package, weights_only=True)
    layer = load_window_package(source_package, device="cuda")
    # Bind the disposable ZML fixture to the same immutable package identity
    # used by native artifact loading.  The fixture stores raw buffers for
    # execution, but its manifest must still prevent a tuning report from
    # being reused with a different window/factor payload.
    from gptqmodel.quantization.qvq_rank8 import (
        _window_artifact_binding_digest,
        export_window_package,
    )

    artifact_package = export_window_package(layer)
    artifact_entries = {}
    for name, value in artifact_package["tensors"].items():
        tensor = value.detach().cpu().contiguous()
        data = tensor.view(torch.uint8).numpy().tobytes()
        artifact_entries[name] = {
            "dtype": str(tensor.dtype).split(".")[-1],
            "shape": list(tensor.shape),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    artifact_payload_sha256 = _window_artifact_binding_digest(
        artifact_package["metadata"], artifact_entries
    )
    recovery_budget = args.max_recovery_overhead_percent
    if recovery_budget is None:
        tuning = getattr(layer, "_p32_window_tuning", None)
        if isinstance(tuning, dict):
            candidate_budget = tuning.get("max_recovery_overhead_percent")
            if candidate_budget is not None:
                recovery_budget = float(candidate_budget)
    if recovery_budget is not None and (
        recovery_budget < 0 or not math.isfinite(recovery_budget)
    ):
        parser.error("--max-recovery-overhead-percent must be finite and non-negative")
    rows = torch.load(args.activations, weights_only=True)["audit_1"].cuda().half()
    report = {"scope": "Real-factor/captured-activation native ABI equivalence; no fitting or model-quality claim",
              "quality_mode": args.quality_mode,
              "preflight": idle.as_dict(), "cases": []}
    args.fixture.mkdir(parents=True, exist_ok=True)
    files = {}

    def save(name, value):
        data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        (args.fixture / (name + ".bin")).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with torch.no_grad():
        m_values = (1, 33, 128, 2048, 8192) if args.zml_m == 8192 else (1, 33, 128, 2048)
        for m in m_values:
            x = rows[torch.arange(m, device="cuda") % rows.shape[0]].contiguous()
            for algorithm in ("hopper_m16", "hopper_direct_decode_mma"):
                for enabled in (False, True):
                    config = P32WindowConfig(
                        algorithm=algorithm, block_m=64 if algorithm == "hopper_direct_decode_mma" else 0,
                        block_n=64 if algorithm == "hopper_direct_decode_mma" else 0,
                        recovery_mode="on" if enabled else "off",
                        quality_mode=args.quality_mode,
                    )
                    prepare_rank8(layer, config)
                    expected = layer(x)
                    actual = native_window_linear(layer, x, config)
                    delta = (actual.double() - expected.double()).abs()
                    row = {"m": m, "k": layer.in_features, "n": layer.out_features,
                           "algorithm": algorithm, "rank8_enabled": enabled,
                           "mae": delta.mean().item(), "max": delta.max().item(),
                           "finite": bool(torch.isfinite(actual).all()), "exact": torch.equal(actual, expected)}
                    report["cases"].append(row)
                    print(row, flush=True)
                    if not row["finite"] or row["mae"] > 2e-3 or row["max"] > 0.046875:
                        raise ValueError("native window ABI exceeds local numerical gate")
                    if m == args.zml_m and algorithm == "hopper_direct_decode_mma":
                        save("expected_on" if enabled else "expected_off", expected)
                        save("x", x)
        window, banks, alt = layer._prepare_amd_p32_metadata(rows.device)
        for name, value in {
            "window": window, "banks": banks, "levels": _pgc16_levels(rows.device, layer.codebook_version),
            "su": layer._cached_cast("SU", torch.float16), "sv": layer._cached_cast("SV", torch.float16),
            "bias": layer._cached_cast("bias", torch.float16), "rank8_a": layer.rank8_A, "rank8_b": layer.rank8_B,
        }.items():
            save(name, value if value is not None else torch.empty(0, dtype=torch.float16))
    import ctypes

    manifest = {
        "artifact_payload_sha256": artifact_payload_sha256,
        "quality_mode": args.quality_mode,
        # Preserve the complete acceptance contract alongside the raw fixture
        # buffers.  The standalone ZML verifier must not tune or replay rank8
        # factors whose audit/signature metadata was dropped during export.
        "recovery": artifact_package["recovery"],
        "config": {"abi_version": 3, "struct_bytes": ctypes.sizeof(WindowConfig), "m": args.zml_m,
                   "k": layer.in_features, "n": layer.out_features, "transition_bits": round(2 * layer.bits),
                   "bank_alt_id": alt, "algorithm": 2, "block_m": 64, "block_n": 64,
                   "input_hadamard": int(layer.input_hadamard), "output_hadamard": int(layer.output_hadamard)},
        "libraries": [next(path for path in torch.ops.loaded_libraries if Path(path).name == name)
                      for name in ("gptqmodel_qvq_cuda_ops.so", "gptqmodel_qvq_wgmma_ops.so")]
                     + [native_window_library()._name],
        "files": files,
    }
    if recovery_budget is not None:
        manifest["max_recovery_overhead_percent"] = recovery_budget
    (args.fixture / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report["fixture"] = manifest
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
