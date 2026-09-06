#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Verify native window ABI on a bound real module and export a ZML execution fixture."""

import argparse
import hashlib
import json
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
    layer = load_window_package(torch.load(args.package, weights_only=True), device="cuda")
    rows = torch.load(args.activations, weights_only=True)["audit_1"].cuda().half()
    report = {"scope": "Real-factor/captured-activation native ABI equivalence; no fitting or model-quality claim",
              "preflight": idle.as_dict(), "cases": []}
    args.fixture.mkdir(parents=True, exist_ok=True)
    files = {}

    def save(name, value):
        data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        (args.fixture / (name + ".bin")).write_bytes(data)
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    with torch.no_grad():
        for m in (1, 33, 128, 2048):
            x = rows[torch.arange(m, device="cuda") % rows.shape[0]].contiguous()
            for algorithm in ("hopper_m16", "hopper_direct_decode_mma"):
                for enabled in (False, True):
                    config = P32WindowConfig(
                        algorithm=algorithm, block_m=64 if algorithm == "hopper_direct_decode_mma" else 0,
                        block_n=64 if algorithm == "hopper_direct_decode_mma" else 0,
                        recovery_mode="on" if enabled else "off",
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
                    if m == 33 and algorithm == "hopper_direct_decode_mma":
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
        "config": {"abi_version": 3, "struct_bytes": ctypes.sizeof(WindowConfig), "m": 33,
                   "k": layer.in_features, "n": layer.out_features, "transition_bits": round(2 * layer.bits),
                   "bank_alt_id": alt, "algorithm": 2, "block_m": 64, "block_n": 64,
                   "input_hadamard": int(layer.input_hadamard), "output_hadamard": int(layer.output_hadamard)},
        "libraries": [next(path for path in torch.ops.loaded_libraries if Path(path).name == name)
                      for name in ("gptqmodel_qvq_cuda_ops.so", "gptqmodel_qvq_wgmma_ops.so")]
                     + [native_window_library()._name],
        "files": files,
    }
    (args.fixture / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report["fixture"] = manifest
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
