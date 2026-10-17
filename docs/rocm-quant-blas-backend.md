# ROCm quantize BLAS backend selection (MI350X / gfx950)

## Summary

A cold quantization run on ROCm spends most of its wall time inside hipBLASLt's
lazy Tensile-library loading, not in the quantization math. Selecting rocBLAS for
the duration of the quantization phase removes that cost.

| method | hipBLASLt cold | rocBLAS cold | delta |
|---|---|---|---|
| GPTQ | 4.12 s | 2.38 s | −1.74 s (−42%) |
| QQQ | 4.28 s | 2.49 s | −1.79 s (−42%) |
| ParoQuant | 6.45 s | 4.56 s | −1.89 s (−29%) |
| AWQ | 3.23 s | 3.24 s | neutral |

Warm runs are unchanged for every method (GPTQ 0.74 s → 0.73 s), so steady-state
throughput is unaffected.

## Where the time goes

`hipBLASLt` loads one Tensile library per (dtype, layout) pair on first use. Each
load finalizes a code object holding thousands of kernel entries:

```
hipModuleLoad(TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bljk_...gfx950.co) 839 ms
hipModuleLoad(TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Alik_Bljk_...gfx950.co) 802 ms
hipModuleLoad(TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Ailk_Bljk_...gfx950.co) 1052 ms
hipModuleLoad(TensileLibrary_SS_SS_HA_Bias_SAV_UA_Type_SS_Contraction_l_Ailk_Bjlk_...gfx950.co) 823 ms
```

Measured with an `LD_PRELOAD` shim over `hipModuleLoad`, on an idle device, with
64x64 operands: the cost is per library, not per operand size. A single
calibration pass touches four combinations (fp32 `l_Alik_Bljk`, `l_Alik_Bjlk`,
`l_Ailk_Bjlk`, fp16 `HH`), and torch additionally picks a `_CU256` variant on
larger shapes.

The same flavors through rocBLAS:

| flavor | hipBLASLt | rocBLAS |
|---|---|---|
| fp32 `l_Alik_Bljk` | 945 ms | 77 ms |
| fp32 `l_Alik_Bjlk` | 831 ms | 29 ms |
| fp32 `l_Ailk_Bjlk` | 858 ms | 43 ms |
| fp16 `HH` | 1069 ms | 1157 ms |

The loads can run concurrently across threads (four libraries load in 1.56 s
wall), but issuing them from a background thread during `load()` does not help:
they contend with the quantization work already on the device and the total
grows (4.6 s vs 4.0 s). Prewarming is not a win here.

## What was tried and did not work

- `HIPBLASLT_PRELOAD_KERNELS=1`: worse (1.2-1.5 s per flavor).
- `HIPBLASLT_USE_ROCROLLER=1`: 6316 ms of JIT for a single flavor.
- Parallel background prewarm from a helper thread: regression, see above.

## Numerics

Different BLAS libraries round differently, so this is a numerics-affecting
change, not a bit-exact refactor. On the 6-layer test model:

- 2.12% of 4-bit codes differ, max code delta 3 (of 0..15).
- The largest relative scale/zero delta is 7.0e-2.
- Both backends are deterministic: repeated runs reproduce their own state hash
  bit-for-bit.

## Implementation

`gptqmodel/utils/rocm.py` exposes `quant_blas_backend()`, a context manager that
switches torch to the rocBLAS backend for the quantization phase and restores the
previous selection afterwards. `GPTQModel.quantize()` wraps its quantize call in
it. The switch is skipped on non-ROCm hosts, when the platform lacks
`torch.backends.cuda.preferred_blas_library`, when the user already selected a
different backend, and when `GPTQMODEL_ROCM_QUANT_BLAS=hipblaslt` is set.
