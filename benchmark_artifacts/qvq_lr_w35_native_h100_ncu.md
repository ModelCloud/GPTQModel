# QVQ LR W3.5 native-N8 Nsight Compute profile — H100

The profile covers the production Hopper W3.5 native-N8 kernel at the Llama
3.2 1B gate/up geometry, M=16, K=2,048, N=8,192. Physical GPU 1 was selected
by UUID `GPU-f5ea03cf-efa4-9807-7de5-b174957a1348`. Nsight Compute 2026.2.1
collected one kernel launch from commit `cb865428`.

| W | M | K | N | Executed instructions | Registers/thread | Static shared KiB | Shared-load conflicts | Shared-store conflicts | Issue active | Active warps | NCU duration us |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.5 | 16 | 2,048 | 8,192 | 16,527,872 | 92 | 46.18 | 935,250 | 131,072 | 42.55% | 11.86% | 39.87 |

The integer/logic pipeline is the most utilized pipeline at 42.1%. This makes
the scalar four-edge W3.5 plane-combination loop the next isolated
instruction-reduction target. Its four low nibbles, four middle two-bit codes,
and four top bits can be spread into four seven-bit slots with fixed masks and
shifts before they enter the existing state recurrence.

The NCU duration includes profiler replay overhead and is not the latency used
for the Marlin/Machete comparison. Benchmark decisions remain based on CUDA
Graph replay timed by CUDA events.
