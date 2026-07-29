# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the MXFP4 CPU AVX-512 VNNI int8 path.

Compares the VNNI kernel against the FP8 reference kernel, the torch baseline,
and the dense FP32 reference across several shapes (including N not divisible
by the 16-column tile).  Set GPTQMODEL_MXFP4_DISABLE_VNNI=1 to exercise the
scalar fallback that runs on CPUs without avx512_vnni.

    /opt/.devin/venv/bin/python scripts/check_mxfp4_vnni_correctness.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_mxfp4_cpu_kernel import (
    _mxfp4_extension,
    dense_reference,
    make_activation,
    quantize_mxfp4,
    torch_baseline,
)

SHAPES = [
    (1, 1536, 18432),
    (8, 1536, 18432),
    (3, 512, 576),
    (8, 128, 77),  # N not a multiple of the 16-column tile
    (2, 64, 16),
]


def main() -> None:
    ext = _mxfp4_extension()
    disabled = os.environ.get("GPTQMODEL_MXFP4_DISABLE_VNNI", "0") == "1"
    print(f"VNNI path {'DISABLED (scalar fallback)' if disabled else 'ENABLED'}")

    failures = 0
    for M, K, N in SHAPES:
        torch.manual_seed(1234)
        weight = torch.randn(N, K, dtype=torch.float32)
        qweight, scales = quantize_mxfp4(weight)
        qpack, spack = ext.mxfp4_prepack_vnni(qweight, scales)

        # The prepack is a permutation, not a dequant cache: the footprint must
        # match MXFP4 exactly, apart from padding N up to the 16-column tile.
        n_pad = -(-N // 16) * 16
        expected_bytes = n_pad * (K // 2) + n_pad * (K // 32)
        packed_bytes = qpack.numel() + spack.numel()
        assert packed_bytes == expected_bytes, (packed_bytes, expected_bytes)

        x = make_activation(M, K, torch.float8_e4m3fn, seed=99)
        ref = dense_reference(x, weight)
        ref_max = ref.abs().max().item() + 1e-8

        out_base = torch_baseline(x, qweight, scales).to(torch.float32)
        out_fp8 = ext.mxfp4_linear_cpu(x, qweight, scales, 0).to(torch.float32)
        out_vnni = ext.mxfp4_linear_cpu_vnni(x, qpack, spack, N, 0).to(torch.float32)

        assert out_vnni.shape == (M, N), out_vnni.shape
        err_dense = (out_vnni - ref).abs().max().item()
        err_base = (out_vnni - out_base).abs().max().item()
        err_fp8 = (out_vnni - out_fp8).abs().max().item()
        base_dense = (out_base - ref).abs().max().item()

        # Outputs are FP8-E4M3, whose relative ULP is 2^-3.  Count elements that
        # differ from the torch baseline by more than one output ULP.
        denom = out_base.abs().clamp_min(ref_max * 1e-3)
        rel = (out_vnni - out_base).abs() / denom
        over_ulp = (rel > 0.13).float().mean().item()
        rms = ((out_vnni - ref) ** 2).mean().sqrt().item()
        rms_base = ((out_base - ref) ** 2).mean().sqrt().item()

        # The VNNI path re-quantizes activations to int8, so bit-exactness with
        # the FP8 reference is not expected; require the RMS error against the
        # dense FP32 reference to stay within 5% of the torch baseline, and only
        # a small tail of outputs to move by more than one output ULP.
        ok = over_ulp < 0.05 and rms <= rms_base * 1.05 + 1e-6
        failures += 0 if ok else 1
        print(
            f"  M={M:<3} K={K:<5} N={N:<6} max_vs_dense={err_dense:.4e} "
            f"(baseline {base_dense:.4e}) max_vs_baseline={err_base:.4e} "
            f"vs_fp8_kernel={err_fp8:.4e} frac>1ulp={over_ulp:.4f} "
            f"rms={rms:.4e} (baseline {rms_base:.4e}) -> {'OK' if ok else 'FAIL'}"
        )

    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
