# QVQ A41/R0 Phase 19: direct FP16 recovery store

Phase 19 removes the separate FP32-to-FP16 cast after independent grouped
output recovery. The recovery Hadamard still keeps FP32 input, FP32 shared
storage, and the exact finite-FP16 rounding emulation at every historical
boundary. Only its final output pointer changes from FP32 to FP16, applying
the same round-to-nearest conversion that the following PyTorch cast used to
perform.

The H100 production path uses the fused store for Q/K child recovery and the
fused-MLP down projection. V has a folded output axis and remains unchanged;
gate/up already use the paired FP16 recovery kernel. H200 and other devices
retain the established path until separately measured.

On the physical 132-SM NVIDIA H100, the isolated recovery boundary improves
**1.1774x** geometric mean. Complete Llama 3.2 1B MLP improves **1.0171x**
over Phase 18 and all 20 rate/M cells improve.

## Exact store math

Let `R` be the FP32 value after the exact recovery sequence:

\[
R=\operatorname{bias}\left(
   \operatorname{SV}\left(
   H_{fp16-emulated}(Y_{inner})\right)\right).
\]

Every historical finite FP16 boundary inside the transform is still applied
by:

\[
round_{finite}(v)=
\begin{cases}
FP32(FP16_{rn}(v)),&FP16_{rn}(v)\text{ is finite},\\
v,&\text{otherwise}.
\end{cases}
\]

Phase 18 then executed a second CUDA node:

\[
O_{old}=FP16_{rn}(R).
\]

Phase 19 changes only the final kernel store:

\[
O_{new}=FP16_{rn}(R).
\]

Thus `O_new` and `O_old` are bit-identical, including the overflow-rescue
case where an intermediate remains FP32 and the final cast becomes FP16
infinity. The kernel template now separates input/shared scalar type from the
output scalar type:

```text
qvq_hadamard_kernel<InputScalar, OutputScalar, PadTo16>
```

For this specialization, `InputScalar=float`, `OutputScalar=half`, and
`PadTo16=false`. No P32 payload, accumulator, split-K reduction, Hadamard
butterfly, normalization, SV multiply, or bias-add order changes.

## Phase-18 bottleneck trace

The first Phase-19 step was a fresh Nsight Systems 2026.4.1 trace of one W3/M1
production CUDA Graph replay after Phase 18:

| Stage | Instrumented duration |
|:--|--:|
| shared input Hadamard | 8.512 us |
| grouped gate/up P32 | 26.464 us |
| paired gate/up recovery | 4.448 us |
| SwiGLU/down precondition | 3.456 us |
| down P32 + fixed split reduction | 14.911 us |
| down recovery Hadamard | 6.080 us |
| separate FP32-to-FP16 cast | 1.376 us |

This identified the recovery/cast boundary as a removable full-operation node
without another decoder-layout experiment.

## Isolated CUDA Graph A/B

The exact-SHA A/B uses 30 warmups, 200 CUDA-event samples, and 50 CUDA Graph
replays per sample. The two paths are bit-exact.

| M | Recovery + separate cast us | Direct FP16 store us | Speedup | FP32 intermediate removed | Better |
|---:|---:|---:|---:|---:|:---:|
| 1 | 8.415 | 7.137 | 1.179x | 8 KiB | Yes |
| 2 | 8.479 | 7.261 | 1.168x | 16 KiB | Yes |
| 4 | 8.329 | 7.271 | 1.145x | 32 KiB | Yes |
| 8 | 8.618 | 7.179 | 1.200x | 64 KiB | Yes |
| 16 | 8.637 | 7.226 | 1.195x | 128 KiB | Yes |

## Nsight Systems graph delta

Matched W3/M1 graph traces before and after promotion show:

| Metric | Phase 18 | Phase 19 | Change |
|:--|--:|--:|:--|
| full MLP visible nodes | 10 | 9 | **-1 node** |
| final recovery kernel | 6.080 us, FP32 store | 6.368 us, FP16 store | slightly longer instrumented kernel |
| FP32-to-FP16 cast | 1.376 us | removed | **one launch removed** |
| sum of visible GPU node durations | 65.247 us | 63.615 us | **1.0257x** |

Raw reports remain outside Git under:

```text
/root/qvq-profiler-artifacts/phase19-full-mlp/
```

The CSV device-name field inherits the host global ordinal-zero H200 label;
CUDA visibility, application UUID/PCI assertions, and `nvidia-smi` establish
that execution used physical GPU 1, H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, PCI `00000000:44:00.0`.

## Storage

The old path retained an `MxN` FP32 recovery output while allocating its
`MxN` FP16 cast result. The new kernel allocates and writes only FP16. For
Llama down with `N=2048`, the removed transient FP32 allocation is:

\[
4M\cdot2048=8M\ \text{KiB}.
\]

That is 8, 16, 32, 64, and 128 KiB for M1, M2, M4, M8, and M16. Q/K receive
the same per-child saving according to their widths. Persistent VRAM,
checkpoints, canonical/grouped P32 payloads, and caches are unchanged.

## Complete Llama 3.2 1B MLP

The formal artifact executes production SHA `25d4c13b` with strict zero-MiB
idle admission. `vs` is comparator latency divided by QVQ latency. `Better`
is the strict median comparison with the committed Phase-18 artifact.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 18 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 63.319 | 1.590 | 0.460x | 0.779x | 1.0146x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 62.407 | 3.226 | 0.500x | 0.789x | 1.0150x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 62.250 | 6.468 | 0.504x | 0.805x | 1.0161x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 62.497 | 12.885 | 0.469x | 0.804x | 1.0194x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 63.163 | 25.499 | 0.514x | 0.798x | 1.0180x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 64.413 | 1.563 | 0.452x | 0.766x | 1.0166x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 63.204 | 3.185 | 0.494x | 0.779x | 1.0156x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 63.157 | 6.375 | 0.497x | 0.793x | 1.0184x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 63.310 | 12.720 | 0.463x | 0.794x | 1.0190x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.143 | 25.110 | 0.507x | 0.786x | 1.0162x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 61.710 | 1.631 | 0.472x | 0.799x | 1.0131x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 60.533 | 3.326 | 0.516x | 0.813x | 1.0135x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 60.282 | 6.680 | 0.521x | 0.831x | 1.0155x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 60.351 | 13.344 | 0.485x | 0.833x | 1.0213x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 61.301 | 26.274 | 0.530x | 0.822x | 1.0148x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 64.203 | 1.568 | 0.454x | 0.768x | 1.0211x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 63.335 | 3.179 | 0.493x | 0.777x | 1.0195x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 63.217 | 6.369 | 0.497x | 0.792x | 1.0186x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 63.284 | 12.725 | 0.463x | 0.794x | 1.0185x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.233 | 25.075 | 0.506x | 0.785x | 1.0166x | Yes |

All 20 cells improve, with **1.0171x** geometric mean versus Phase 18.
Geometric means are **2.2346x** versus ordinary per-module QVQ, **0.4893x**
versus Marlin W4, and **0.7952x** versus Machete W4. The current-run Machete
control was faster than in Phase 18, so its independently measured ratio moves
slightly down even though QVQ itself improves. W4 remains a figurative
dense-equivalent throughput baseline, not an equal-work or equal-quality
claim.

## Correctness and promotion gates

- Ten randomized combinations cover M1/M2/M4/M8/M16 and both FP16-emulation
  normalization orders; all are bit-exact and CUDA Graph stable.
- A dedicated late-butterfly/SV-overflow case proves the direct store equals
  the former final cast even when recovery must temporarily retain FP32 range.
- Native type/mode/padding guards pass.
- All 50 grouped runtime tests pass across every rate/M and both group types,
  including telemetry, invalidation/fallback, exact child outputs, graphs, and
  real Llama logits/cached generation.
- Formal timing uses only the physical H100, CUDA events around warmed CUDA
  Graph replay, and strict three-sample 0%-utilization/zero-MiB admission.
- Compilation remains capped at four Ninja jobs, one NVCC host thread, and one
  split-compile partition.

Artifacts:

- `artifacts/a41_phase19_h100/fp16_recovery_store_experiment.json`
- `artifacts/a41_phase19_h100/production_mlp_fp16_recovery_store_vs_phase18.json`

## Next experiment

The graph is now nine nodes and the grouped gate/up P32 kernel still owns
roughly 40% of W3/M1 time. Phase 20 should return to that kernel with matched
Nsight Compute evidence and target a representation-level decoder or
consumer-scheduling change; the remaining non-P32 boundaries are individually
too small to close the Machete gap.
