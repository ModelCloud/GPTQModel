# QVQ LR direct-fragment Nsight Compute profile — H100

The profile covers the production direct-fragment kernel at Llama 3.2 1B
gate/up geometry, M=16, K=2,048, N=8,192. Physical GPU 1 was selected by UUID
`GPU-f5ea03cf-efa4-9807-7de5-b174957a1348` and was idle before collection.
Nsight Compute 2026.2.1 collected one kernel launch per rate.

| W | M | K | N | Executed instructions | Registers/thread | Static shared KiB | Shared-load conflicts | Shared-store conflicts | Issue active | Active warps | NCU duration us |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 2,048 | 8,192 | 14,197,760 | 70 | 35.42 | 2,554,185 | 0 | 40.28% | 12.09% | 39.49 |
| 2.5 | 16 | 2,048 | 8,192 | 15,086,080 | 78 | 36.96 | 2,507,503 | 0 | 34.00% | 12.10% | 47.81 |

The direct-fragment W2 kernel executes 235,520 fewer instructions than the
earlier row-major native-N8 experiment (14,433,280, a further 1.63% reduction)
and uses one fewer register per thread. Decoded-weight shared stores are now
conflict-free. The roughly 2.5 million remaining shared-load conflicts point to
the row-major activation tile loaded by `ldmatrix`, making activation row
spacing the next isolated experiment.

The NCU duration includes profiler replay overhead and is not the latency used
for the Marlin/Machete comparison. Formal latency remains the CUDA Graph replay
with internal CUDA events in the companion benchmark artifact.

## Activation-row padding follow-up

The accepted 40-half activation row spacing was profiled with the same W2.5
M=16, K=2,048, N=8,192 command and metrics.

| Variant | Executed instructions | Registers/thread | Static shared KiB | Shared-load conflicts | Shared-store conflicts | Issue active | NCU duration us | Better than last |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Direct fragment, stride 32 | 15,086,080 | 78 | 36.96 | 2,507,503 | 0 | 34.00% | 47.81 | no |
| Padded activation, stride 40 | 15,215,104 | 91 | 43.10 | 934,876 | 131,072 | 45.36% | 37.57 | yes |

The padding reduces shared-load conflicts by 62.7% and raises issue activity by
11.36 percentage points. It trades 13 registers/thread, 6.14 KiB of static
shared memory, and 0.85% more instructions for a 21.4% shorter profiled kernel.

## Vector-store follow-up

The accepted aligned `uint4` decoded-fragment store was profiled at the same
W2.5 M=16, K=2,048, N=8,192 geometry on physical GPU 1.

| Variant | Executed instructions | Registers/thread | Static shared KiB | Shared-load conflicts | Shared-store conflicts | Issue active | NCU duration us | Better than last |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Padded activation, scalar source | 15,215,104 | 91 | 43.10 | 934,876 | 131,072 | 45.36% | 37.57 | no |
| Explicit vector fragment store | 15,214,080 | 91 | 43.10 | 934,760 | 131,072 | 45.68% | 37.54 | yes |

The explicit vector form saves 1,024 executed instructions (0.007%). This is
far smaller than the source-level four-to-one store reduction because the CUDA
compiler had already combined nearly all adjacent scalar stores. The profile
therefore keeps decode arithmetic and indexed level-table reads as the next
material instruction-reduction targets. Nsight Compute duration includes
profiler overhead; the benchmark decision remains based on CUDA Graph replay
timed by CUDA events.
