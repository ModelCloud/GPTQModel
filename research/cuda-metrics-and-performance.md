# CUDA metrics are evidence, not optimization objectives

## Core rule

A decrease in instruction count, increase in occupancy, higher bandwidth,
fewer stalls or more Tensor Core activity is **not sufficient by itself**
to establish a speedup. A metric can be predictive under a demonstrated
bottleneck model; its direction alone is not a universal predictor.

The acceptance objective is matched, reproducible latency or throughput at
the required correctness and quality. Metrics help explain the mechanism.

## Primary evidence

[Nsight Compute's Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
distinguishes resident, eligible and issued warps. A scheduler skips an issue
slot when none are eligible. Sampled warp waits are not equivalent to idle GPU
cycles, and `not_selected` can mean another ready warp was issued.

[CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/efficient_gemm.html)
describes register-heavy GEMMs with relatively low occupancy and software
pipelining to hide latency. This gives a concrete mechanism by which using
more storage for overlap can be worthwhile.

## How a metric can mislead (QVQ engineering interpretation)

| Observation | Possible benefit | Possible opposing effect |
|---|---|---|
| Fewer executed instructions | Less issue/ALU demand | Dependent lookup latency, lower ILP or more expensive opcodes |
| More registers | Reuse, larger accumulators, prefetch distance | Fewer resident warps or resource exhaustion |
| Higher occupancy | More latency-hiding opportunities | Less per-thread reuse, smaller tiles, extra traffic |
| Fewer memory bytes | Less demand on the measured memory level | Added decode/control work or worse access efficiency |
| Higher GB/s | Better delivery of useful operands | More unnecessary bytes per output |
| Fewer sampled stalls | More ready work | Different denominator, fewer samples or a moved bottleneck |
| Fewer bank conflicts | Less shared serialization | Extra permutations, replicas, barriers or global traffic |
| Higher Tensor Core utilization | Better feed or useful matrix work | Padding/redundant work or slower non-matrix phases |

Distinguish static instruction count from executed warp instructions, and
warp instructions from per-thread counts. Distinguish theoretical occupancy
from achieved occupancy and from eligible-warps statistics. Normalize
comparisons to the same completed work and inspect absolute values as well
as percentages.

## QVQ evidence and procedure

The [LUT experiment](cuda-lut-tradeoffs.md) shows why replacing arithmetic
with memory operations needs direct timing. It does not contain the counters
needed to declare instruction count the measured cause.

For each candidate, write a specific prediction: for example, removing repeated
index arithmetic should reduce issue pressure without increasing dependent
loads. Then inspect source-correlated instructions, memory traffic, resource
limits and scheduler behavior together.

Save exact binary/source versions, input shapes, dtype, launch geometry,
cache/replay settings and repeated unprofiled timings. A one-kernel microbenchmark
does not account for transforms, table setup, correction, reductions or extra
launches. Compare complete operators and prefill/decode separately.

Do not tune toward lower occupancy or higher register use either; those reverse
heuristics are just as unjustified. Keep the faster, correct design with an
explanation supported by multiple measurements.

See [Nsight workflow](nsight-profiling.md), [Roofline](roofline.md) and
[pipeline design](cuda-pipeline-throughput.md). No new benchmark was run for
this documentation.
