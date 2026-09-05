# Full dispatch ceiling after upstream integration

Merge revision: 400970d5, integrating origin/main 83c8fc33 into the WIP.
Pre-merge production comparison: 7a2c3e14. Overall target baseline remains
c89459e3; this diagnostic does not change the baseline or narrow the goal.

## Complete paired diagnostic

`direct_dispatch_400970d5_full.json` contains all 364 cases, all seven Qwen3.8-27B
shape groups, M1 through4096, and rates2/2.5/3/3.5. Strict idle and per-timing
process checks passed. Warmup20, iterations50, alternating baseline/candidate
CUDA-event timings. Raw candidate bypasses layer guards only for the352 eligible
folded cases; the12 unsupported large gate/up cases retain the public fallback.
No unguarded production path is enabled.

All364 outputs are bitwise identical to the pre-merge production comparison.
352 pass canonical max-absolute0.002; the12 unchanged gate/up fallback cases
remain exact-baseline-equal but above the canonical threshold. All applicable
existing graph/stream/ownership checks passed. Synthetic legal packed fixtures
prove kernel equivalence only, not model quality.

Overall diagnostic geometric mean: 1.076523901x. Only1/364 reaches1.5x against
current production even after bypassing dispatch. This is not the production
count against c89459e3, which remains the previously recorded36/364. Do not
multiply separate-run averages to manufacture an overall result.

Per-M geometric means over four rates, raw operator versus current production:

| Shape | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_q_gate | 1.020 | 1.008 | 0.999 | 1.002 | 0.999 | 0.999 | 1.000 | 0.997 | 1.006 | 1.023 | 1.003 | 1.024 | 1.002 |
| full_kv | 1.304 | 1.364 | 1.344 | 1.332 | 1.341 | 1.319 | 1.338 | 1.349 | 1.378 | 1.045 | 0.999 | 0.999 | 1.000 |
| attn_out | 1.291 | 1.236 | 1.313 | 1.278 | 1.119 | 1.118 | 1.009 | 1.004 | 1.001 | 1.000 | 0.999 | 1.001 | 1.000 |
| linear_qkv | 1.353 | 1.096 | 1.123 | 1.001 | 1.002 | 1.002 | 0.999 | 0.999 | 1.001 | 1.001 | 1.003 | 1.003 | 1.002 |
| linear_z | 1.299 | 1.278 | 1.402 | 1.374 | 1.366 | 1.359 | 0.999 | 0.999 | 1.000 | 1.000 | 1.004 | 1.004 | 1.001 |
| mlp_gate_up | 1.013 | 1.011 | 1.016 | 1.013 | 1.011 | 0.998 | 0.993 | 1.000 | 0.998 | 1.001 | 1.000 | 1.002 | 0.999 |
| mlp_down | 1.081 | 1.085 | 1.042 | 1.067 | 1.065 | 1.015 | 1.006 | 1.002 | 0.999 | 1.003 | 1.002 | 1.002 | 1.001 |

## Decision and backend lookup

Guard cleanup cannot deliver the full target. There is still useful host
headroom for small full-KV and linear-Z cases, but full-Q, larger linear-QKV,
and gate/up need GPU-side work. Prioritize a same-type FP16 FlyDSL HGEMM trial
for high-only folded paths; preserve the existing FP32 residual/composite
fallbacks. Test full-Q and linear-QKV over M8..4096 first, then expand to all
eligible shapes and all364 before any promotion. M1/2/4 still need separate
small-M routing. A targeted pilot will not satisfy the overall target.

Installed FlyDSL is discoverable at
`/opt/python314/lib/python3.14/site-packages/flydsl/__init__.py`; Primus-Turbo
is not installed. Installed AITER remains7440ef72503e1c3fadc5be85a5c74eb7c9c34841,
and authenticated upstream main lookup returned636098e5a462abfb2900efe623751e7a612b09e3.
The installed default `bf16_tuned_gemm.csv` has zero matching FP16 entries
for the seven target K/N pairs at gfx950,256CUs. This describes that file,
not every possible user override or every backend capability.

The inspected `flydsl_hgemm` accepts A[M,K],B[N,K], same-type output and explicit
M/N/K tile, split-K and pipeline settings. It rejects a supplied FP32 output
for FP16 input. Do not replace residual GEMMs by rounded FP16 products or call
FP32 split-K scratch FP32 output support. Weight layout must be verified to
avoid hidden per-call contiguous copies. Correctness and executed profiling
remain required; no FlyDSL performance result exists in this phase.

The fetched Hopper work includes opt-in FP8 MLP caching/fusion. Its changed
arithmetic and MLP scope do not establish compliance for this locked FP16
single-linear contract. Reuse/layout and launch-removal ideas remain relevant,
but its headline speedups cannot be transferred to this workload.

## Merge verification / generated-code scope

The merge did not change `qvq_amd.py`, `qlinear/qvq.py`, either experimental AMD
kernel source, or this benchmark harness. The diagnostic calls the same
operator with the same tile selection and mathematical expression; no GPU
instruction rewrite or new ISA/occupancy delta is claimed. Prior executed
profiles and SSA audits remain the device evidence. Upstream CUDA changes
are not claimed to be validated on this AMD device. No new GPU-code change
is introduced by this report or the accompanying test-only fix.

The combined AMD/experimental/JIT run returned1123passed,2failed in14.46s.
All1074 AMD/experimental tests passed. Both failures were newly merged
Swordfish ABI tests expecting a version error without disabling the earlier
ROCm rejection. Tests now explicitly select the non-ROCm branch, with an
additional test preserving ROCm-before-ABI rejection. The complete JIT file
then passed52tests in3.56s,14existing Python3.14 deprecation warnings.
Production Swordfish behavior was not changed. Ruff reports four existing
issues in unchanged lines of the merged JIT test file (I001, twoPIE807,
BLE001); no new lint finding is in the added lines. Whitespace checks pass.

The initial pre-merge diagnostic stopped with a detected new GPU PID and
terminal exit1. Its partial report is not used. The PID was absent and the
GPU idle on reinspection; no process was killed. The accepted full rerun
exited0 from the merged revision.

Reproduction:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID HIP_VISIBLE_DEVICES=0 python scripts/benchmark_qvq_p32_amd_butterfly.py \
  --butterfly none --folded-direct-ceiling --baseline-amd-commit 7a2c3e14 \
  --full-sweep --warmup 20 --iterations 50 \
  --output /tmp/qvq-direct-revisit-merged/report.json
```

Logs: `/tmp/qvq-direct-revisit-merged.log`, `/tmp/qvq-merged-400970d5-tests.log`,
`/tmp/qvq-merged-jit-fixed-tests.log`. Hardware/software/config/source fingerprints
are retained in the full JSON. The goal remains active and unmet.
