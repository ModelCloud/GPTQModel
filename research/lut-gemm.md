# LUT-GEMM: lookup-based quantized matrix computation

## Primary reference and scope

Gunho Park et al.,
[LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference
in Large-Scale Generative Language Models, v4](https://arxiv.org/abs/2206.09557v4),
ICLR 2024.

The paper proposes lookup-based quantized matrix multiplication to reduce
dequantization and computation costs, with group-wise quantization and
evaluations of generation workloads. Its author-reported gains concern that
algorithm and benchmark configuration.

## QVQ implication (proposed)

A LUT can be part of a different algorithm, not simply a substitution of one
arithmetic helper by a load. Compare the complete representation, table
construction, lookup reuse and accumulation path before transferring results.

Do not equate LUT-GEMM's computation with QVQ's PGC state-to-codebook decoder
or [FLUTE](flute.md). They answer different lookup-design questions.

This is evidence against a blanket claim that CUDA LUT methods cannot win,
not a recommendation to replace QVQ's current decoder. Start with the measured
[QVQ LUT tradeoffs](cuda-lut-tradeoffs.md), preserve quantization/accuracy
contracts, and require a matched full-operator result.
