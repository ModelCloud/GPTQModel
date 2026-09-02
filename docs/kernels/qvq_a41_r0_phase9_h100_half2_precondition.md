# QVQ A41/R0 Phase 9: H100 half2 precondition high stage

Phase 9 vectorizes the last five stages of the exact N=8192 SwiGLU/down-input
Hadamard precondition on the physical H100.  Two adjacent FP16 columns now
travel through one `half2` register value.  This removes **75.7%** of the
high-stage executed instructions and improves the isolated complete
precondition by **1.131x** geometric mean.

The complete W2--W3.5 MLP improves in all 20 M/rate cells.  Geometric-mean
latency is **1.0102x** better than accepted Phase 8, with a new range of
**67.33--72.77 us**.

This is an execution-only change.  The canonical P32 payload, quantized
weights, scales, activation function, FP16 product, butterfly ordering,
rounding boundaries, operation-local workspace, and non-H100 fallbacks are
unchanged.

## Exact math

The N=8192 multiblock precondition factorizes the ascending butterfly into
eight low stages over 256-value tiles and five high stages over the 32 tile
indices.  For a fixed within-tile column, the old high stage evaluates every
pair as:

\[
s=\operatorname{fp16}_{rn}
  (\operatorname{fp32}(a)+\operatorname{fp32}(b)),
\]

\[
d=\operatorname{fp16}_{rn}
  (\operatorname{fp32}(a)-\operatorname{fp32}(b)).
\]

Adjacent within-tile columns follow identical tile-index butterflies and
have no cross-column dependency.  They can therefore occupy the two lanes of
one `half2`:

```cpp
half2 a = values[tile];
half2 b = values[peer];
values[tile] = __hadd2(a, b);
values[peer] = __hsub2(a, b);
```

Each lane performs the same FP16 add/sub and round-to-nearest operation as the
scalar sequence.  No addition is reassociated, no transform stage moves, and
the two columns remain independent.

## Ownership and layout

The input and output remain ordinary contiguous `[M, 8192]` FP16 tensors.
Since every within-tile column pair starts at an even address and both the row
stride and 256-column tile stride are multiples of two, all `half2` loads and
stores are naturally aligned.

| Property | Legacy high | Phase-9 high |
|:--|--:|--:|
| columns/thread | 1 | 2 adjacent |
| threads/block | 64 | 32 |
| blocks/row | 4 | 4 |
| warps/row | 8 | 4 |
| tile-index values/thread | 32 half | 32 half2 |
| registers/thread | 40 | 40 |
| shared memory/block | 1.024 KiB | 1.024 KiB |
| local memory/spills | 0 | 0 |

The grid deliberately keeps four blocks per row.  The candidate has fewer
warps, so promotion was conditional on CUDA-event timing rather than source
or instruction counts alone.

The experiment adds one fixed device specialization.  There is no rate, M,
dtype, bias, or shape template expansion.  Production selects it only inside
the existing physical-H100/SM90/N=8192 multiblock gate.  H200 and unsupported
widths retain the previous path.

## Matched SASS

Nsight Compute captured one M1 launch of each high stage from the same binary.
Profiler duration is shown only as supporting attribution; CUDA events define
the performance result.

| Metric/opcode | Legacy | half2 | Change |
|:--|--:|--:|--:|
| total executed warp instructions | 3,984 | 968 | **-75.7%** |
| `HADD2` | 1,280 | 320 | -75.0% |
| `FADD` | 1,280 | 0 | -100% |
| `F2FP` | 640 | 0 | -100% |
| `HFMA2` | 0 | 320 | +320 |
| `LDG` | 256 | 128 | -50.0% |
| `STG` | 256 | 128 | -50.0% |
| `PRMT` | 136 | 0 | -100% |
| NCU duration | 4.352 us | 3.392 us | -22.1% |

The compiler uses packed `HFMA2` and `HADD2` for the two FP16 lanes and
eliminates the scalar unpack/add/repack chain.  Registers stay at 40 and no
spill appears.

As expected from halving the warps, active-warps occupancy falls from 3.63%
to 1.50%, while long-scoreboard stalls rise from 6.45 to 7.28 cycles per
issued instruction.  The instruction reduction is large enough to win despite
those scheduler costs.  This is the key distinction from the rejected
paired-round recovery experiment, which removed only 3.57% of combined
instructions and lengthened dependencies.

## Isolated precondition benchmark

Legacy and `half2` paths are compiled into the same extension and timed in the
same process.  Times are warmed CUDA Graph replays measured by CUDA events:
30 warmups, 100 samples, and 50 replays per sample.  Every output is bit-exact.

| M | Legacy p50 us | half2 p50 us | Speedup | Better |
|---:|---:|---:|---:|:---:|
| 1 | 4.845 | 4.261 | 1.137x | Yes |
| 2 | 5.113 | 4.456 | 1.147x | Yes |
| 4 | 5.073 | 4.620 | 1.098x | Yes |
| 8 | 5.293 | 4.625 | 1.145x | Yes |
| 16 | 5.601 | 4.960 | 1.129x | Yes |

Geometric-mean speedup is **1.131x**.

## Complete Llama 3.2 1B MLP

The formal matrix uses 20 warmups, 50 CUDA-event samples, and 20 CUDA Graph
replays per sample.  It includes both gate/up projections, recovery, SiLU,
FP16 product, down precondition, and the down projection.  `vs` is comparator
latency divided by QVQ latency, so a value below one means the W4 comparator
remains faster.  `Better` compares with accepted Phase 8.

| Rate | M | MKN (gate/up; down) | QVQ us | Effective TFLOP/s | vs Marlin W4 | vs Machete W4 | vs Phase 8 | Better |
|---:|---:|:---|---:|---:|---:|---:|---:|:---:|
| W2 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 70.006 | 1.438 | 0.427x | 0.739x | 1.0078x | Yes |
| W2 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 70.238 | 2.866 | 0.456x | 0.734x | 1.0116x | Yes |
| W2 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 71.097 | 5.663 | 0.454x | 0.722x | 1.0100x | Yes |
| W2 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 71.311 | 11.293 | 0.423x | 0.722x | 1.0161x | Yes |
| W2 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 67.327 | 23.922 | 0.490x | 0.762x | 1.0151x | Yes |
| W2.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 70.814 | 1.422 | 0.422x | 0.730x | 1.0111x | Yes |
| W2.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 71.293 | 2.824 | 0.449x | 0.723x | 1.0118x | Yes |
| W2.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 71.714 | 5.615 | 0.450x | 0.716x | 1.0154x | Yes |
| W2.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 72.554 | 11.099 | 0.415x | 0.709x | 1.0137x | Yes |
| W2.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 68.294 | 23.584 | 0.483x | 0.751x | 1.0079x | Yes |
| W3 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 71.442 | 1.409 | 0.418x | 0.724x | 1.0077x | Yes |
| W3 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 71.708 | 2.808 | 0.446x | 0.719x | 1.0016x | Yes |
| W3 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 71.846 | 5.604 | 0.449x | 0.714x | 1.0099x | Yes |
| W3 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 72.770 | 11.067 | 0.414x | 0.707x | 1.0126x | Yes |
| W3 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 68.836 | 23.398 | 0.480x | 0.745x | 1.0083x | Yes |
| W3.5 | 1 | 1x2048x8192 (x2); 1x8192x2048 | 70.915 | 1.419 | 0.422x | 0.729x | 1.0066x | Yes |
| W3.5 | 2 | 2x2048x8192 (x2); 2x8192x2048 | 71.302 | 2.824 | 0.449x | 0.723x | 1.0120x | Yes |
| W3.5 | 4 | 4x2048x8192 (x2); 4x8192x2048 | 72.123 | 5.583 | 0.447x | 0.712x | 1.0049x | Yes |
| W3.5 | 8 | 8x2048x8192 (x2); 8x8192x2048 | 71.966 | 11.190 | 0.419x | 0.715x | 1.0165x | Yes |
| W3.5 | 16 | 16x2048x8192 (x2); 16x8192x2048 | 68.546 | 23.497 | 0.482x | 0.748x | 1.0046x | Yes |

Geometric means:

- **1.0102x** versus accepted Phase 8;
- **1.999x** versus ordinary per-module QVQ;
- **0.444x** versus Marlin W4;
- **0.727x** versus Machete W4.

The W4 comparisons are figurative dense-equivalent efficiency baselines.
They do not claim equal quantization quality or equal compressed decode work.

## Correctness and telemetry

- 30 matched random cases cover M=1/2/4/8/16, three seeds, and both legacy
  and `half2` high stages, each repeated ten times with bit-exact output.
- Both paths pass CUDA Graph replay and contract guards.
- A real Llama 3.2 layer passes exact repeatability, logits within the locked
  `2e-3` dense/reference tolerance, cached generation, and token equality.
- Production telemetry records
  `h100_half2_precondition_high_launches` independently from the existing
  multiblock-precondition counter.

Artifacts and driver:

- `artifacts/a41_phase9_h100/half2_precondition_experiment.json`
- `artifacts/a41_phase9_h100/production_mlp_half2_high_vs_baselines.json`
- `scripts/benchmark_qvq_phase9_half2_precondition.py`

Nsight reports and source/raw exports are kept outside Git under
`/root/qvq-profiler-artifacts/phase9-half2-precondition/`.
