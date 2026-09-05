# AMD kernel backend lookup

This lookup was verified on 2026-09-05. Resolve current upstream HEADs before integration and pin the revision used in
benchmark artifacts or build scripts.

## Backend map

| Candidate | Canonical upstream and inspected revision | Best fit | First places to inspect |
| --- | --- | --- | --- |
| AITER / FlyDSL | [ROCm/aiter](https://github.com/ROCm/aiter) `456b92780c8b650c1e3e4b0fa1ca21f0d1fb363d` | gfx950 FP16/BF16, low-precision, tuned per-shape dispatch, and custom FlyDSL | `aiter/ops/flydsl/gemm_kernels.py`, `aiter/ops/flydsl/kernels/gemm_a16w16_gfx950.py`, `aiter/tuned_gemm.py`, `aiter/configs/bf16_tuned_gemm.csv`, `aiter/configs/model_configs/`, `aiter/ops/opus/` |
| Gluon frontend | [triton-lang/triton](https://github.com/triton-lang/triton) `6e7587360478ae956f050816de99cb0361bb90db` | Lower-level layout and pipeline control on the Triton compiler stack | `python/tutorials/gluon/`, `python/triton/language/extra/libdevice.py`; import from `triton.experimental.gluon` only after probing the installed build |
| gfx950 Gluon designs | [ROCm/gfx950-gluon-tutorials](https://github.com/ROCm/gfx950-gluon-tutorials) `4d7d632a320b25a789bbdb7a9ba8a8683dce2142` | MI350/MI355 GEMM layouts, LDS pipelines, wave scheduling, rocprof automation | `kernels/gemm/intra_wave/a16w16/`, `kernels/gemm/inter_wave/a16w16/`, `docs/`, `scripts/` |
| Primus-Turbo | [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo) `785b1429c85c2506672c547cc02d9b42cba7daaf` | Packaged gfx942/gfx950 GEMM, GroupedGEMM, FP8/FP4, fusion, and backend autotuning | `primus_turbo/pytorch/ops/gemm.py`, `grouped_gemm.py`, `gemm_fp8.py`, `primus_turbo/pytorch/kernels/gemm/`, `benchmark/` |
| PyTorch scaled GEMM | [pytorch/pytorch](https://github.com/pytorch/pytorch) plus the installed ROCm build | Scaled FP8/low-precision GEMM when dtype and scale layout match | `torch.ops.aten._scaled_mm.default._schema`, `torch._C._dispatch_dump_table('aten::_scaled_mm')`, local `gptqmodel/nn_modules/qlinear/fp8.py` |
| hipBLASLt / rocBLAS | [ROCm/hipBLASLt](https://github.com/ROCm/hipBLASLt), [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) | Portable dense GEMM fallback and library baseline | PyTorch `torch.mm` trace, AITER `hipb_mm`, Primus `BackendType.HIPBLASLT`, installed `/opt/rocm/lib/` |

The revisions above are lookup provenance records, not permanent dependency pins. Runtime reinspection on
2026-09-05 found `/opt/aiter-glm53-tune` at `7440ef72503e1c3fadc5be85a5c74eb7c9c34841` and upstream main at
`636098e5a462abfb2900efe623751e7a612b09e3`. At the inspected installed revision,
`flydsl_hgemm(a, b, ...)` uses A `[M,K]`, B `[N,K]`, FP16/BF16 inputs and **same-type output only**:
`_validate_hgemm_inputs` explicitly rejects `out.dtype != a.dtype`. FP32 split-K scratch is not FP32 output support.
Do not route the FP32 high/residual GEMMs through that wrapper without establishing a different supported API.
Its tuned dispatcher can select `flydsl`, `opus`, `asm`, `skinny`, `triton`, `torch`, or `hipblaslt` per shape.
Recheck these contracts against the revision actually used; a newer upstream HEAD is not evidence of its behavior.

AITER's `skinny_gemm` also allocates same-type output, even though its wrapper accepts an `otype` argument.
The inspected `wvSpltK` native dispatcher supports activation M=1..4, K divisible by 8, physical N-by-K weights,
and a runtime CU count; `LLMM1` requires M=1. Check the selected native accumulation path before benchmarking,
and reject FP32-output requests instead of silently ignoring them.

Primus-Turbo's inspected `gemm(a, b, trans_a=False, trans_b=False, out_dtype=None)` is a 2-D FP16/BF16 API whose
default implementation selects hipBLASLt; the project also exposes backend-specific tuning below the public wrapper.
Its dependency footprint is larger than a self-contained kernel, so measure cold import/build and packaging impact as
well as hot latency.

## Reproducible lookup commands

```bash
gh api repos/ROCm/aiter/commits/main --jq .sha
gh api repos/ROCm/gfx950-gluon-tutorials/commits/main --jq .sha
gh api repos/AMD-AGI/Primus-Turbo/commits/main --jq .sha
gh api repos/triton-lang/triton/commits/main --jq .sha

git clone --filter=blob:none https://github.com/ROCm/aiter.git /tmp/aiter-current
rg -n "flydsl_hgemm|libtype|hipblaslt|opus|tuned_gemm" /tmp/aiter-current/aiter

gh search code "gfx950 repo:AMD-AGI/Primus-Turbo" --limit 50
gh search code "gemm repo:ROCm/gfx950-gluon-tutorials" --limit 50
```

Probe the active PyTorch build rather than assuming scaled GEMM support:

```python
import torch

print(torch.__version__, torch.version.hip)
print(hasattr(torch, "_scaled_mm"))
if hasattr(torch.ops.aten, "_scaled_mm"):
    print(torch.ops.aten._scaled_mm.default._schema)
    print(torch._C._dispatch_dump_table("aten::_scaled_mm"))
```

## Shape-level decision record

For every candidate, retain a row with backend revision, M/N/K, dtype/layout, output dtype, preprocess/cache bytes,
launch count, p50/mean/p95, max absolute error, and rejection reason. Include all requested M values even when dispatch
uses buckets. A library win at large M does not justify routing decode-like M to it.

For QVQ compensated high-plus-residual weights, first test whether a backend can consume both terms without writing a
large intermediate. Two independent library GEMMs plus an FP32 add may still win at large M, while a single fused
Gluon/FlyDSL kernel may win when launch count or rereading A dominates. Treat changed accumulation order as a new
numerical candidate and certify it against the canonical FP32 reference.
