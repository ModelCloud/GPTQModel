# QUICK: quantization-aware interleaving and shared-memory conflicts

## Primary reference and finding

Taesu Kim et al.,
[QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient
LLM inference, v1](https://arxiv.org/abs/2402.10076v1),
especially §3 of the [paper](https://arxiv.org/pdf/2402.10076).

The paper addresses bank conflicts in mixed-precision GEMM's
post-dequantization shared-memory write-back. It interleaves quantized weights
offline to avoid that write-back on the studied path. Its reported speedups
are scoped to the evaluated kernels, devices and models.

## Proposed QVQ application

Investigate whether the decoded representation can directly match the matrix
fragment layout, avoiding an intermediate store/load or permutation.
This relates to QVQ's direct decode-MMA work but does not prove that QUICK's
layout applies to a trellis/window representation.

Distinguish conflicts during table reads from conflicts during decoded-weight
write-back. Both use shared resources, but they are different bottlenecks and
need different address evidence.

Preserve exact decoded values and operand placement, then measure the emitted
instructions and whole operator. Eliminating a shared round trip can be useful
even if instruction count or register demand rises elsewhere.
See [LUT tradeoffs](cuda-lut-tradeoffs.md) and [metrics](cuda-metrics-and-performance.md).
No new QUICK implementation or experiment is claimed here.
