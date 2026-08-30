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
