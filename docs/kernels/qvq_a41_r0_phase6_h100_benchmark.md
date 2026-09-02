# QVQ A41/R0 Phase 6 H100 split-K benchmark

The benchmark targets the physical NVIDIA H100 at PCI address
`00000000:44:00.0`; the H200 is hidden with `CUDA_VISIBLE_DEVICES=1`.  A strict
idle-device gate runs before Torch import.  Timings use warmed CUDA Graph
replays bracketed by CUDA events, excluding CPU/container launch gaps.

This first probe measures only the P32 inner decoder and matrix multiply for
Llama 3.2 1B down, (M=1,K=8192,N=2048,W3).  It is a scheduling experiment,
so full-module Marlin and Machete values are intentionally not mixed into this
table.  Production comparisons are added after runtime policy integration.

| M | K | N | Split | Blocks | Median (us) | Speedup vs split 1 | Repeatable | Max abs vs dense |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 8192 | 2048 | 1 | 32 | 72.806 | 1.000x | Yes | 0.00007010 |
| 1 | 8192 | 2048 | 2 | 64 | 38.898 | 1.872x | Yes | 0.00003386 |
| 1 | 8192 | 2048 | 4 | 128 | 22.122 | 3.291x | Yes | 0.00001621 |
| 1 | 8192 | 2048 | 8 | 256 | 17.647 | 4.126x | Yes | 0.00000763 |
| 1 | 8192 | 2048 | 16 | 512 | 16.506 | 4.411x | Yes | 0.00000429 |
| 1 | 8192 | 2048 | 32 | 1024 | 18.449 | 3.946x | Yes | 0.00000381 |

Split 16 wins this W3/M1 probe.  Split 32 regresses 11.8% from split 16,
showing that decoder parallelism has crossed the point where extra block and
reduction traffic dominate.  Every ordered split is bit-repeatable over ten
eager launches and the captured CUDA Graph result equals the eager reference.

The raw record is
`artifacts/a41_phase6_h100/llama_down_w3_m1_ordered_probe.json`.
