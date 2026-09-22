---
name: benchmark-oracle-identity
description: Validate benchmark and numerical-oracle arm identity before comparing tokens, tensors, quality, throughput, or memory. Use for kernel, compiler, inference, batching, KV-cache, and end-to-end A/B tests where mismatched rows, shapes, or runtime configuration could create false drift.
---

# Benchmark oracle identity

Treat comparison identity as a hard precondition, not a post-hoc explanation. A
difference between non-equivalent arms is **harness-invalid** and says nothing
about candidate accuracy or performance.

## Build and verify an arm fingerprint

Before computing any delta, record a machine-readable fingerprint for each arm:

- source revisions, dirty-state diff hash, binary/shared-library hashes, model
  and tokenizer identity;
- dataset revision and exact sample keys in execution order;
- compiled engine batch, request batch, active-lane count, row-to-lane mapping,
  padding/inactive-lane policy, prefill bucket, logical context, physical KV
  capacity/page geometry, maximum generation, stop conditions, and sampling;
- hardware, runtime/compiler versions, relevant environment, allocator and
  graph mode, autotune/cache identity, and warmup state.

Compare fingerprints before outputs. Fields outside the declared treatment must
match exactly. Print the mismatching fields and stop; never reinterpret that
result as numerical drift, quality loss, or a speed change.

## Pair rows by stable identity

Pair samples by a stable dataset key and prompt/token hash, never by an
unqualified phrase such as “row 1” or by positional index alone. Record whether
indices are zero- or one-based. Verify uniqueness, cardinality, order, rendered
prompt tokens, and initial state. Fail on missing, duplicate, reordered, or
differently tokenized samples.

Batch shape is part of the numerical oracle. B1, B64, B128, and B256 are
different experiments even when they contain the same sample. Do not compare a
B1 token stream with one lane extracted from a B128 run unless a separate
batch-invariance test first proves that contract. The same rule applies to
prefill buckets, padding, active masks, CUDA graphs, KV layout/capacity, and
continuous-batching schedules.

## Preflight controls

Before candidate attribution:

1. Run control versus control with identical fingerprints and require the
   expected repeatability.
2. Run candidate versus candidate to detect nondeterminism.
3. Compare baseline versus candidate only after both checks and the fingerprint
   match pass.
4. For a local kernel oracle, feed byte-identical captured inputs, weights, and
   state to both implementations and compare at the same operator boundary.
5. For model quality, retain per-sample outputs and paired gain/loss data in
   addition to aggregate scores.

A KV-pool exhaustion, changed generation length, or first-token divergence is a
diagnostic trigger. First re-check arm fingerprints and run the matching control;
do not assign causality to the most recent kernel change.

## Reporting

Every comparison report must state the stable sample key, batch/request geometry,
row-to-lane mapping, fingerprint match result, and control repeatability result.
Keep results from mismatched arms for diagnosis but label them
`harness-invalid`; they cannot accept or reject an optimization.

