# QVQ P32 Hopper large-M optimization, second pass

## Scope and baseline

This pass starts from merged `origin/main` commit `8c131ac5`, after the first
native Hopper large-M implementation.  It targets the complete Llama 3.2 1B
MLP on the physical 132-SM NVIDIA H100.  Timing uses warmed CUDA Graph replay
and CUDA events.  Correctness is checked against the dense FP32 P32 Torch
oracle, and the low-level transform is also required to be bit-exact to the
existing unpaired implementation.

The representative MLP consists of two `M x 2048 x 8192` gate/up projections,
exact FP16 recovery and SwiGLU/down preconditioning, followed by one
`M x 8192 x 2048` down projection.

## Phase 1: pair large-M recovery tiles

The fused recovery/precondition kernel reconstructs the five high Hadamard
stages after the independent 256-column low transforms.  Output tile `t` and
tile `t + 16` differ only in the final high-transform bit.  The already-proven
paired kernel therefore evaluates the shared high tree once and produces both
outputs:

```text
shared high-tree prefix
        -> output tile t
        -> output tile t + 16
```

Both values retain their original FP16 recovery, SiLU, product, down scale,
and low-transform rounding boundaries.  Large-M dispatch now selects this
kernel with a `16 x M` grid instead of the unpaired `32 x M` grid.  No
checkpoint, quantization, selector, or P32 decode representation changes.

The operator remains graph-safe and uses transient workspaces allocated by
PyTorch before graph replay.  Direct tests at M32 and M512 require bit-exact
FP16 output and repeated CUDA Graph equality.

## H100 benchmark

The comparison is against the fresh merged-main artifact measured on the same
H100.  Ratios above one mean QVQ is faster than the W4 baseline.

| Weight | MLP MKN (gate/up; down) | Merged main | Phase 1 | Speedup | vs Marlin W4 | vs Machete W4 | Better than last |
|---|---|---:|---:|---:|---:|---:|---|
| W2 | 128x2048x8192; 128x8192x2048 | 132.746 us | 122.528 us | 1.083x | 0.649x | 0.595x | Yes |
| W2 | 512x2048x8192; 512x8192x2048 | 458.784 us | 416.464 us | 1.102x | 0.397x | 0.280x | Yes |
| W2 | 4096x2048x8192; 4096x8192x2048 | 3449.629 us | 3111.446 us | 1.109x | 0.446x | 0.293x | Yes |
| W2.5 | 128x2048x8192; 128x8192x2048 | 134.454 us | 124.038 us | 1.084x | 0.641x | 0.588x | Yes |
| W2.5 | 512x2048x8192; 512x8192x2048 | 461.190 us | 422.634 us | 1.091x | 0.391x | 0.276x | Yes |
| W2.5 | 4096x2048x8192; 4096x8192x2048 | 3461.939 us | 3144.467 us | 1.101x | 0.441x | 0.290x | Yes |
| W3 | 128x2048x8192; 128x8192x2048 | 133.107 us | 122.278 us | 1.089x | 0.650x | 0.597x | Yes |
| W3 | 512x2048x8192; 512x8192x2048 | 454.355 us | 412.173 us | 1.102x | 0.401x | 0.283x | Yes |
| W3 | 4096x2048x8192; 4096x8192x2048 | 3457.146 us | 3088.131 us | 1.119x | 0.449x | 0.295x | Yes |
| W3.5 | 128x2048x8192; 128x8192x2048 | 134.861 us | 124.502 us | 1.083x | 0.639x | 0.586x | Yes |
| W3.5 | 512x2048x8192; 512x8192x2048 | 475.651 us | 434.413 us | 1.095x | 0.380x | 0.268x | Yes |
| W3.5 | 4096x2048x8192; 4096x8192x2048 | 3583.834 us | 3225.254 us | 1.111x | 0.430x | 0.282x | Yes |

All twelve cells improve.  The measured range is 7.4% to 10.7% lower complete
MLP latency.

## Exact-commit NCU and SASS audit

Commit `86fa881b` was profiled at W3/M512.  Matched Nsight Systems traces show
the changed recovery/precondition stage falling from 103.871 us to 61.567 us,
a 40.7% reduction.  The gate/up and down P32 kernels remain approximately
175 us and 96 us respectively.

Nsight Compute for the paired stage reports:

| Metric | Value |
|---|---:|
| Executed warp instructions | 50,300,002 |
| NCU replay duration | 61.760 us |
| Registers per thread | 40 |
| Achieved occupancy | 71.11% |
| Compute throughput | 79.16% |
| DRAM throughput | 24.69% |
| Eligible warps per scheduler | 4.15 |

The leading executed opcode families are `HADD2` 5.44M, `FSETP` 5.15M,
`F2FP` 4.85M, `LDG` 4.59M, `FSEL` 4.59M, `FADD` 4.33M, and `IMAD` 2.13M.
Pairing is therefore a representation-level win: it removes duplicated
high-tree work and halves the grid rather than relying on a small compiler
peephole.

The next optimization target is the exact compressed P32 execution itself.
At W3/M512 the grouped gate/up kernel executes 102.9M warp instructions while
using only 6.33% of HBM bandwidth.  Its dominant opcode families are `IMAD`
25.17M, `LDS` 17.89M, `PRMT` 12.68M, `LOP3` 11.20M, and `SHF` 11.11M.  A
second recovery-only change cannot deliver the remaining cumulative 25%
target.
