# QVQ A41/R0 Phase 23: fused H100 down reduction and recovery

## Decision

Phase 23 promotes an H100-only Llama 3.2 1B down-projection path that folds
the deterministic split-16 reduction directly into exact output recovery.
The change removes one CUDA launch and the intermediate reduced FP32 tensor
without changing P32 decode, WGMMA accumulation, split ordering, Hadamard
butterfly ordering, FP16-emulation boundaries, output scaling, bias, or final
FP16 rounding.

The isolated down decode-plus-recovery site improves by **1.1135x geometric
mean** across W2--W3.5 and M1--M16. The complete MLP improves by **1.0275x
geometric mean** versus Phase 19, and all 20 cells improve.

Production executable commit: `c861b0a0`.

## Previous execution

The H100 Llama down projection has geometry

\[
M\times K\times N=M\times8192\times2048,
\qquad M\in\{1,2,4,8,16\}.
\]

Its measured split policy is 16. The WGMMA kernel writes sixteen disjoint
FP32 planes with physical shape

\[
P\in\mathbb{R}^{16\times16\times2048}.
\]

Phase 19 executed:

```text
split-16 WGMMA
    -> [16, 16, 2048] FP32 partials
ordered fixed reducer
    -> [16, 2048] FP32 inner output
output Hadamard + SV + bias
    -> [M, 2048] FP16 output
```

For each scalar output, the reducer used the exact left-to-right expression

\[
r=((((0+P_0)+P_1)+\cdots)+P_{15}).
\]

## Fused math

The Phase-23 recovery block begins with the identical expression:

```cpp
float value = 0.0f;
#pragma unroll
for (int split = 0; split < 16; ++split) {
    value += partial[split][row][column];
}
```

It then applies the established recovery operation without materializing
`r` in global memory. For the production stable-output mode:

\[
v_0=R_{16}(r),
\qquad
v_1=R_{16}\left(\frac{v_0}{R_{16}(\sqrt{N})}\right),
\]

where `R16(x)` rounds to FP16 and back to FP32 when finite, while retaining
the FP32 value if narrowing would overflow. Each ascending Hadamard butterfly
bit then performs

\[
(a,b)\mapsto
\left(R_{16}(a+b),R_{16}(a-b)\right).
\]

The epilogue remains

\[
y_j=R_{16}(h_j\cdot SV_j),
\qquad
y_j=R_{16}(y_j+b_j),
\qquad
out_j=\operatorname{fp16}_{rn}(y_j).
\]

Thus the optimization changes only where the first reduction result lives:

```text
split-16 WGMMA
    -> [16, 16, 2048] FP32 partials
fused ordered reduction + Hadamard + SV + bias + FP16 store
    -> [M, 2048] FP16 output
```

The partial workspace remains because 512 WGMMA blocks must retain their
deterministic child-local split planes. Persistent checkpoint and cache
storage is unchanged.

## Dispatch boundary

The fused path is fail-closed and requires all of the following:

- physical device name `NVIDIA H100`, compute capability 9.0;
- V2B2-P32, vector size 2, and W2/W2.5/W3/W3.5;
- Llama down geometry `K=8192`, `N=2048`;
- one through sixteen FP16 rows;
- measured ordered split count 16;
- output Hadamard enabled;
- the already-promoted H100 grouped MLP lifecycle.

All other devices, shapes, formats, training/adapters, and ordinary module
execution retain the established reduced-FP32 path. The runtime counter
`h100_fused_down_reduction_recovery_launches` proves that the new lifecycle
site fired.

## Isolated H100 result

The formal isolated artifact uses 30 warmups, 100 samples, and 50 CUDA Graph
replays per sample. Timing is by CUDA events; admission required zero MiB in
use on physical H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`. Every cell is bit-exact to the
separate ordered reducer plus recovery.

| Rate | M | MKN | Separate us | Fused us | Speedup | Better |
|---:|---:|:---|---:|---:|---:|:---:|
| W2 | 1 | 1x8192x2048 | 22.983 | 19.967 | 1.1510x | Yes |
| W2 | 2 | 2x8192x2048 | 22.254 | 19.919 | 1.1172x | Yes |
| W2 | 4 | 4x8192x2048 | 22.015 | 19.894 | 1.1066x | Yes |
| W2 | 8 | 8x8192x2048 | 22.028 | 20.042 | 1.0991x | Yes |
| W2 | 16 | 16x8192x2048 | 22.147 | 20.149 | 1.0991x | Yes |
| W2.5 | 1 | 1x8192x2048 | 22.279 | 19.934 | 1.1176x | Yes |
| W2.5 | 2 | 2x8192x2048 | 22.363 | 20.040 | 1.1159x | Yes |
| W2.5 | 4 | 4x8192x2048 | 22.328 | 19.943 | 1.1196x | Yes |
| W2.5 | 8 | 8x8192x2048 | 22.399 | 20.221 | 1.1077x | Yes |
| W2.5 | 16 | 16x8192x2048 | 22.334 | 20.289 | 1.1008x | Yes |
| W3 | 1 | 1x8192x2048 | 21.410 | 19.016 | 1.1259x | Yes |
| W3 | 2 | 2x8192x2048 | 21.446 | 19.152 | 1.1198x | Yes |
| W3 | 4 | 4x8192x2048 | 21.379 | 19.029 | 1.1235x | Yes |
| W3 | 8 | 8x8192x2048 | 21.391 | 19.321 | 1.1071x | Yes |
| W3 | 16 | 16x8192x2048 | 21.435 | 19.406 | 1.1045x | Yes |
| W3.5 | 1 | 1x8192x2048 | 22.196 | 20.105 | 1.1040x | Yes |
| W3.5 | 2 | 2x8192x2048 | 22.240 | 20.007 | 1.1116x | Yes |
| W3.5 | 4 | 4x8192x2048 | 22.424 | 20.030 | 1.1195x | Yes |
| W3.5 | 8 | 8x8192x2048 | 22.415 | 20.154 | 1.1122x | Yes |
| W3.5 | 16 | 16x8192x2048 | 22.587 | 20.393 | 1.1076x | Yes |

The isolated geometric mean is **1.1135x**; all 20 cells improve.

## Complete Llama 3.2 1B MLP

The formal full-MLP artifact uses the Phase-19 matched protocol: 30 warmups,
200 samples, and 50 CUDA Graph replays per sample. Marlin W4 and Machete W4
are freshly measured in the same process. `vs` means comparator latency
divided by QVQ latency. W4 remains a figurative dense-equivalent throughput
baseline rather than an equal-rate or equal-quality claim. `Better` is the
strict median comparison with the committed Phase-19 production artifact.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 19 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 61.618 | 1.634 | 0.476x | 0.813x | 1.0276x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 60.745 | 3.314 | 0.512x | 0.821x | 1.0274x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 60.502 | 6.655 | 0.517x | 0.831x | 1.0289x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 61.051 | 13.191 | 0.480x | 0.825x | 1.0237x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 61.412 | 26.226 | 0.529x | 0.822x | 1.0285x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 62.759 | 1.604 | 0.468x | 0.798x | 1.0264x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 61.592 | 3.269 | 0.505x | 0.810x | 1.0262x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 61.439 | 6.554 | 0.509x | 0.818x | 1.0280x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 62.031 | 12.982 | 0.473x | 0.812x | 1.0206x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 62.563 | 25.744 | 0.520x | 0.806x | 1.0253x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 59.715 | 1.686 | 0.491x | 0.839x | 1.0334x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 58.605 | 3.435 | 0.530x | 0.851x | 1.0329x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 58.583 | 6.873 | 0.534x | 0.858x | 1.0290x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 58.887 | 13.675 | 0.498x | 0.855x | 1.0249x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 59.617 | 27.016 | 0.545x | 0.846x | 1.0282x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 62.420 | 1.613 | 0.470x | 0.802x | 1.0286x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 61.482 | 3.275 | 0.506x | 0.811x | 1.0301x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 61.670 | 6.529 | 0.507x | 0.815x | 1.0251x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 61.540 | 13.086 | 0.477x | 0.818x | 1.0283x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 62.533 | 25.756 | 0.520x | 0.807x | 1.0272x | Yes |

Geometric means are **2.2950x** versus ordinary per-module QVQ, **0.5028x**
versus Marlin W4, **0.8227x** versus Machete W4, and **1.0275x** versus Phase
19. All 20 cells improve.

## Measured profiling

The tools used here are real profilers rather than source inspection alone:

- Nsight Systems 2026.4.1 verifies the CUDA Graph node sequence;
- Nsight Compute captures the promoted fused kernel through 19 hardware
  counter replay passes;
- CUDA events inside warmed graph replay provide promotion timing.

The Phase-19 graph contained nine visible QVQ MLP nodes. Its down tail was:

| Node | Nsight Systems duration |
|:--|--:|
| split-16 down WGMMA | 12.704 us |
| fixed split-16 reducer | 1.984 us |
| FP32-to-FP16 recovery | 6.368 us |

The Phase-23 graph contains eight visible QVQ MLP nodes. Its corresponding
tail is:

| Node | Nsight Systems duration |
|:--|--:|
| split-16 down WGMMA | 13.344 us |
| fused reduction/recovery | 6.688 us |

Instrumentation durations are not CUDA-event production timings. Their role
is to prove that the fixed reducer node disappeared and the fused node is
actually present.

The targeted W3/M1 Nsight Compute capture reports:

| Metric | Fused recovery |
|:--|--:|
| Grid x block | 1 x 1024 |
| Registers/thread | 32 |
| Dynamic shared memory | 8.448 KiB |
| Local spills | 0 |
| Executed warp instructions | 29,184 |
| NCU replay duration | 9.184 us |
| Eligible warps/scheduler/cycle | 1.313 |
| Active warps | 51.41% |
| DRAM throughput | 0.674% |
| Combined memory throughput | 3.645% |

The kernel remains a one-block M1 recovery transform and is not bandwidth
saturated. The Phase-23 win comes from deleting a materialization boundary,
not from accelerating the Hadamard itself.

Binary reports stay outside Git:

```text
/root/qvq-profiler-artifacts/phase23-full-mlp/
```

The Nsight CSV device label still inherits global ordinal zero's H200 name on
this host. CUDA visibility, UUID admission, 97,871 MiB memory, and 132-SM
device assertions establish that execution used the physical H100.

## Storage and graph behavior

The split partial workspace is unchanged:

\[
16\cdot16\cdot2048\cdot4=2\text{ MiB}.
\]

The removed reduced FP32 tensor is:

\[
16\cdot2048\cdot4=128\text{ KiB}.
\]

The output remains only `M x 2048` FP16. Persistent VRAM, checkpoint bytes,
canonical P32 payloads, and grouped caches are unchanged. The operator is
CUDA-Graph safe and reuses the captured graph's transient allocations.

## Validation

- 20/20 isolated rate/row cells: bit-exact and faster;
- 20/20 complete MLP cells: exact, repeatable, and faster than Phase 19;
- five focused split-recovery CUDA-Graph tests: passed;
- real Llama layer logits and cached generation: passed;
- grouped Hopper P32 and lifecycle regression suite: **65 passed**;
- compilation capped at Ninja `-j4`; NVCC split compilation remained bounded.

Artifacts:

- `artifacts/a41_phase23_h100/down_reduction_recovery_experiment.json`
- `artifacts/a41_phase23_h100/production_mlp_fused_down_recovery_vs_phase19.json`

## Next phase

The final down tail is now one tensor-core launch plus one recovery launch.
Further isolated Hadamard tuning has limited whole-MLP leverage. Phase 24
should inspect whether gate/up's grouped split-1 WGMMA can feed recovery
without materializing both complete FP32 children, while preserving child
output ordering and every recovery rounding boundary. Any design that merely
adds another shared-memory exchange or reduces the number of useful WGMMA
blocks is excluded by the rejected Phase-13 N128 experiment.
