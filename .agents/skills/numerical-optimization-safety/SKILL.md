---
name: numerical-optimization-safety
description: Audit QVQ, XLA, StableHLO, and backend optimizations that may change numerical results; require matched eager-oracle evidence before accepting them.
---

# Numerical optimization safety

Use this skill for any QVQ kernel, wrapper, fusion, autotuning, layout,
reduction, dtype, rounding, compiler, XLA, or StableHLO change that can affect a
quantized model's output. Algebraic equivalence is not numerical equivalence:
reassociation, accumulation order, split reductions, fusion, layout, autotune
selection, and implicit casts can change logits and selected tokens.

## Required baseline

Establish a reproducible unfused Transformer eager oracle before judging an
optimization. Hold constant the real model and weights, token IDs, rendered
prompts, attention mask, position IDs/KV state, dtype inputs, tokenizer, padding
direction and IDs, EOS normalization, random seed, and sampling settings.
Record the full CLI, effective configuration, dataset revision/path, run and arm
IDs, hardware, runtime/compiler versions, and QVQ/ZML/XLA/StableHLO commits.

Compare paired rows, not just aggregate accuracy. Capture the earliest available
boundaries: tokenization, QVQ input preparation, dequantized/quantized linear
output, Hadamard/epilogue, normalization, attention/KV state, residual, final
logits, and selected token. Save raw tensors or lossless hashes plus durable
stdout/stderr for every arm. For every quantization and post-quantization
evaluation, also write a Markdown or JSON record next to the artifact containing
its stored path, full CLI, effective config, dataset/checkpoint paths and
revisions, run ID, arm ID, hardware/runtime versions, QVQ/ZML/XLA/StableHLO
commits, and complete results. Do not treat an artifact without this record as a
valid comparison input.

## Optimization review

For every changed result, inspect the QVQ execution plan and internal kernel
selection as well as generated StableHLO/HLO and ZML custom-call configuration
when present. Check:

- fusion boundaries and epilogues;
- reduction tree, reassociation, per-thread accumulation and split-K/partial sums;
- accumulator/output dtypes and exact FP16/BF16 rounding boundaries;
- GEMM tile, layout, padding, masking, bank/trellis packing and transition bits;
- attention implementation, KV-cache updates, batching and CUDA-graph mode;
- autotune candidates, cache reuse, stale cache entries, and GPU-specific dispatch.

Add lightweight plan telemetry when needed: selected kernel/config, shape, dtype,
workspace, split count, backend, and cache provenance. Telemetry must not alter
the stream, synchronization, graph-capture behavior, or numerical path.

Use a one-variable intervention for each suspected cause. Report the first
divergent tensor and exact plan/instruction/configuration delta. Separate a local
kernel error gate from propagated model logits and task accuracy; they answer
different questions. Relative error or a matching top-1 token alone is never a
correctness proof.

## Acceptance policy

An optimization with no measured speed benefit in the target workload must not be
kept when it adds numerical drift. Retain the accurate implementation as the
default while investigating. A speed benefit does not waive a declared accuracy
gate: present scope, uncertainty, local mean/max drift, propagated logit/token
impact, and held-out task evidence for human review before changing criteria.

Validate repeated execution, eager versus optimized execution, and graph/capture
execution where applicable. Use real model weights and real tokenized data;
synthetic tensors are limited to algebraic, kernel, and serialization tests.

## Ownership and patching

Find the first source of the numerical change before fixing it. A generic
XLA/StableHLO behavior needs an upstreamable compiler fix, with any downstream
QVQ/ZML mitigation clearly scoped. A ZML integration/policy issue belongs in ZML;
a QVQ kernel/wrapper or quantization-contract issue belongs here. Never mask an
upstream numerical change with a tolerance or silently relabel a backend. Version
or invalidate compiler/autotune caches when selection policy changes.

Every accepted fix must include a focused regression test, before/after numeric
metrics, matched performance measurements, and complete provenance. Preserve
failed, partial, and rejected experiment records so optimized but less accurate
artifacts cannot be mistaken for reference behavior.

Use [$quantized-model-provenance](../quantized-model-provenance/SKILL.md) for all
quantization and post-quantization evaluation, [$qvq-kernel-accuracy](../qvq-kernel-accuracy/SKILL.md)
for QVQ kernel gates, and [$qvq-multi-agent-git-sync](../qvq-multi-agent-git-sync/SKILL.md)
for branch/PR lifecycle work.
