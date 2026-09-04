# QVQ A41/R0 Phase 18: direct padded shared input transform

Phase 18 removes the separate allocation, zero-fill, and valid-row copy
between A41/R0's shared input Hadamard and the grouped Hopper P32 kernel. For
`M < 16`, the exact `SU -> Hadamard` kernel now writes the final `16x2048`
input directly. The M real-row blocks execute the unchanged transform and
then cooperatively own disjoint zero-tail rows. M16 and every unsupported
device, width, or non-Hadamard group retain the established path.

On the physical 132-SM NVIDIA H100, the isolated boundary improves **1.2181x**
geometric mean for M1/M2/M4/M8. Complete Llama 3.2 1B MLP improves **1.0268x**
over Phase 17 across the 16 affected cells, and all affected cells improve.
The same production specialization also applies to canonical Llama QKV.

## Exact input math

For each real row, the shared A41/R0 transform is:

\[
T_r=H\left(X_r\odot SU_{shared}\right),\qquad 0\le r<M.
\]

The prior runtime constructed the grouped P32 input in two later operations:

\[
P=0_{16\times2048},\qquad P[0:M,:]=T.
\]

Phase 18 writes the same matrix directly:

\[
P_r=\begin{cases}
T_r,&0\le r<M,\\
0,&M\le r<16.
\end{cases}
\]

Grouped P32 and every child segment remain row-independent:

\[
Y_{i,r}=P_rQ_i.
\]

No padded row enters a valid row's Hadamard butterfly. The kernel launches
exactly M transform blocks, not sixteen. After a real block finishes its
normal output epilogue, block `b` owns padded rows:

\[
r=M+b, M+b+M, M+b+2M,\ldots <16.
\]

These row sets are disjoint, so no synchronization, atomic, or changed
rounding order is introduced. Tests require the complete `16x2048` FP16
matrix to be bit-exact to explicit zero-fill and copy.

## CUDA pipeline

Before:

```text
shared SU/Hadamard (M rows)
  -> allocate/fill 16x2048 zeros
  -> copy Mx2048 valid rows
  -> grouped P32
```

Phase 18:

```text
shared SU/Hadamard (M transformed rows + disjoint zero-tail stores)
  -> grouped P32
```

The transient grouped payload is built before transform dispatch so the
runtime's measured H100 capability gate is available on the first grouped
execution. Source/device invalidation clears that gate together with the
payload.

## Isolated CUDA Graph A/B

The exact-SHA A/B uses 30 warmups, 200 CUDA-event samples, and 50 CUDA Graph
replays per sample. Both paths return the complete padded matrix.

| M | Separate padding us | Direct padding us | Speedup | Transient bytes removed | Better |
|---:|---:|---:|---:|---:|:---:|
| 1 | 9.641 | 9.142 | 1.055x | 4 KiB | Yes |
| 2 | 9.678 | 8.008 | 1.209x | 8 KiB | Yes |
| 4 | 9.634 | 7.480 | 1.288x | 16 KiB | Yes |
| 8 | 9.668 | 7.208 | 1.341x | 32 KiB | Yes |
| 16 | 6.934 | 6.916 | 1.003x | 0 | Yes (identical control) |

M1 has only one real transform block, so that block also writes fifteen zero
rows and its gain is smaller. M16 invokes identical code in both controls;
its small difference is timing variance.

## Nsight Systems launch evidence

Nsight Systems 2026.4.1 traced one warmed M1 CUDA Graph replay with graph-node
tracing enabled. The application admits only H100 UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`, PCI `00000000:44:00.0`. Raw
reports remain outside Git under:

```text
/root/qvq-profiler-artifacts/phase18-direct-input-padding/
```

| Graph node | Separate duration | Direct duration | Change |
|:--|--:|--:|:--|
| input Hadamard | 5.984 us | 8.352 us | same one-block transform plus zero-tail stores |
| FP16 zero fill | 1.056 us | removed | one kernel removed |
| device-to-device valid-row copy | 0.959 us | removed | one memory node removed |
| visible nodes | 3 | 1 | **-2 nodes** |

Profiler instrumentation shows the expected trade: the sole kernel executes
more stores, while two graph nodes disappear. Those instrumented durations do
not include the complete graph scheduling effect and are not used as the
performance decision; the 10,000-replay CUDA-event distributions above are.
As on Phase 17, the exported Nsight CSV inherits the host global ordinal-zero
`H200` label even though CUDA visibility, the application UUID assertion, and
`nvidia-smi` identify physical GPU 1 as the H100.

## Storage

The old path retained an `Mx2048` FP16 transform output while allocating the
final `16x2048` input. Phase 18 retains only the final padded matrix. Removed
transient live bytes per grouped invocation are:

\[
2M\cdot2048=4M\ \text{KiB}.
\]

That is 4, 8, 16, and 32 KiB for M1, M2, M4, and M8. Both QKV and gate/up can
use the specialization, but their buffers are invocation-local rather than
simultaneously persistent. Checkpoints, canonical/grouped P32 payloads,
persistent caches, and persistent VRAM are unchanged.

## Complete Llama 3.2 1B MLP

The formal artifact executes production SHA `2e49f7e9` with strict zero-MiB
idle admission. `vs` is comparator latency divided by QVQ latency. `Better`
is the strict median comparison with the committed Phase-17 artifact.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 17 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 64.242 | 1.567 | 0.453x | 0.788x | 1.0118x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 63.343 | 3.178 | 0.493x | 0.795x | 1.0246x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 63.255 | 6.366 | 0.496x | 0.804x | 1.0369x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 63.709 | 12.640 | 0.460x | 0.802x | 1.0407x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 64.298 | 25.049 | 0.506x | 0.792x | 1.0029x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 65.484 | 1.537 | 0.445x | 0.773x | 1.0067x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 64.191 | 3.136 | 0.486x | 0.785x | 1.0267x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 64.321 | 6.260 | 0.488x | 0.790x | 1.0327x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 64.515 | 12.483 | 0.455x | 0.792x | 1.0390x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.183 | 24.709 | 0.499x | 0.781x | 1.0017x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 62.516 | 1.610 | 0.466x | 0.810x | 1.0084x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 61.348 | 3.282 | 0.509x | 0.821x | 1.0267x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 61.217 | 6.578 | 0.513x | 0.830x | 1.0362x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 61.638 | 13.065 | 0.476x | 0.829x | 1.0405x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 62.209 | 25.890 | 0.523x | 0.818x | 1.0024x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 65.559 | 1.535 | 0.444x | 0.773x | 1.0021x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 64.571 | 3.118 | 0.483x | 0.780x | 1.0236x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 64.393 | 6.253 | 0.488x | 0.789x | 1.0335x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 64.455 | 12.494 | 0.455x | 0.793x | 1.0408x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 65.300 | 24.665 | 0.498x | 0.779x | 1.0002x | Yes |

The affected M1-M8 cells improve **16/16** with **1.0268x** geometric mean.
All 20 measured medians improve, with **1.0218x** geometric mean; M16 is
unchanged code, so its favorable differences are run-to-run telemetry. The
all-cell geometric means are **2.1969x** versus ordinary per-module QVQ,
**0.4812x** versus Marlin W4, and **0.7961x** versus Machete W4. W4 remains a
figurative dense-equivalent throughput baseline rather than equal work or
equal quantization quality.

## Correctness and promotion gates

- Five M values prove byte-exact valid rows, exact zero tails, ordinary-launch
  repeatability, and CUDA Graph stability.
- Eleven focused native Hadamard/range tests pass on the physical H100.
- All 50 grouped runtime tests pass across W2/W2.5/W3/W3.5, both QKV and
  gate/up, and M1/M2/M4/M8/M16.
- Runtime coverage includes fallback/invalidation, telemetry, CUDA Graphs,
  exact child outputs, and real Llama logits/cached generation.
- Formal timing uses only the physical H100, CUDA events around warmed CUDA
  Graph replay, and strict three-sample 0%-utilization/zero-MiB admission.
- Compilation is capped at four Ninja jobs, one NVCC host thread, and one CUDA
  split-compile partition.

Artifacts:

- `artifacts/a41_phase18_h100/direct_input_padding_experiment.json`
- `artifacts/a41_phase18_h100/production_mlp_direct_input_padding_vs_phase17.json`

## Next experiment

Phase 19 re-profiles the full graph and fuses the final independent-recovery
FP32-to-FP16 cast into the recovery kernel's store. See
`docs/kernels/qvq_a41_r0_phase19_h100_fp16_recovery_store.md`.
