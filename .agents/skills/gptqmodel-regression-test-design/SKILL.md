---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-regression-test-design
description: Construct regression tests for GPT-QModel bug fixes and behavior-changing refactors, especially checkpoint/offload round trips, partial-failure retries, concurrency and cache ownership, mutable config propagation, calibration replay, and numerical/layout boundaries. Require evidence that tests detect the broken behavior rather than merely exercising the fix.
---

# GPT-QModel regression-test design

## At a glance

- **What:** a regression that fails on the broken code for the intended reason, exercises the next consumer and the risky state transition, and passes with the fix.
- **Why:** recent fixes kept testing a helper or the first successful call while the bug lived in a reload, a retry, a second consumer, or another representation, and so they regressed.
- **When:** every bug fix and behaviour-changing refactor, alongside the domain skills.
- **Where:** extend the existing family suite under `tests/`. The [Nov 17 audit](references/2026-11-17-regressions.md) and [Nov 18 follow-ups](references/2026-11-18-regressions.md) link fixes to tests worth extending.
- **How:** write the invariant, prove red then green, pick the applicable transition rows in section 3, and report the evidence table in the PR.

Recent fixes repeatedly covered a helper or the first successful call while
missing its next consumer, another representation, or a partially completed
operation. The audits separate demonstrated regressions from recurring bug
families. Use these mechanisms to
design tests, not the historical implementation as a permanent specification.

## 1. Write the invariant before the patch

Record the public entry point, broken observable result, trigger, owning state,
and the transition that invalidates the old assumption. Inspect recent fixes
and tests for the same mechanism, not only the same filename or issue number.
An issue and the PR that fixes it are one event, not two recurrences.

Choose the applicable rows below; justify omissions rather than multiplying
every backend, dtype, and model into an impractical full Cartesian product.
Add a tiny deterministic unit regression and a consumer-level test when the
bug crosses a loader, looper, writer, or dispatch boundary. Prefer extending an
existing family suite to another disconnected issue-number-only test file.

## 2. Prove the test can detect the defect

1. Run the new test against the broken implementation in an isolated worktree,
   or temporarily reintroduce only the defective operation in a disposable
   checkout. Do not discard unrelated work or rewrite shared history.
2. Verify it fails at the intended behavioral assertion. Import errors, a
   missing newly added symbol, collection failures, timeouts unrelated to the
   defect, and skips are not red evidence. If a pre-fix run is infeasible,
   report why and mark detection evidence unverified.
3. Assert the trigger actually occurred: a nonzero off-block update, a second
   replay pass, a native dispatch, a partial tile, an injected failure, or
   overlapping owners. A test that never reaches its branch can pass forever.
4. Apply the fix and run the regression plus the affected family suite. Keep
   the negative/control case proving that valid behavior still works and that
   unrelated errors are not swallowed.

Build expected results independently: checkpoint tensors before transformation,
a fresh reference run, native model execution, explicit token rows, or a Torch
oracle. Never derive the expected answer using the same converter, normalizer,
cache, or quantizer under test. Random data supplements adversarial fixtures;
all-zero weights, identity scales, diagonal Hessians, and uniform masks can
remove the very effect being tested.

Mocks may isolate an external dependency or inject a failure at its boundary.
Do not mock away the mutation, conversion, replay, or dispatch under test.
Call-count/source-text assertions are supporting evidence, not substitutes for
observable tensor values, selected policy, restored state, or released owners.

**Prove the guard runs where it belongs.** When adding or moving a regression,
check both the CI path filter and the job's pytest selection. Run the affected
nodes with `pytest -rs` in the intended dependency profile. Keep a CPU policy
test runnable without optional native dependencies: scope integration imports,
fixtures and skips to the nodes that need them. For a native-path guard, use
the backend's actual runtime/build prerequisite and assert which implementation
served the call. Collection, a skipped node, or a passing fallback is not a
pass for the target path.

## 3. Test the relevant transition, not only construction

### Checkpoint layout, dtype, and model ownership

For changed serialization/load/offload paths, exercise the applicable sequence:
`pack -> forward -> save -> load -> forward -> re-save -> reload -> forward`.
Also test direct `state_dict()` copies into **fresh and already initialized**
modules, `load_state_dict(assign=False/True)`, and offload/restore where used.

- Assert the complete owned tensor-key set, storage dtype, packed values, and
  layout markers, plus forward parity. A matching marker alone cannot prove
  the words were converted; a finite output cannot prove the right weights
  were loaded. Repeated loads/saves must not double-convert qzeros.
- Deliberately mismatch model compute dtype and stored tensor dtype. Include
  packed floating weights, scales, biases, other buffers, and dense Parameters
  where supported; a buffer-only fixture misses Parameter casts. Use values
  that would change on narrowing. Compare tensor contents, not container-file
  byte identity (headers/sharding may legitimately differ).
- Include mixed quantized/dense or pruned MoE entries and a real tiny model
  requiring Defuser when changing shell reconstruction. Assert all experts'
  payloads survive and nonzero fixtures stay nonzero. Removing a required
  destination must fail before publishing a successful checkpoint.
- Prove the in-session model still forwards, saves again, and moves when those
  operations are supported. Cleanup may reclaim a resource only after its last
  live consumer releases it; assert both continued usability and eventual
  cleanup, including a failed save followed by retry.

Follow [checkpoint serialization](../gptqmodel-checkpoint-serialization/SKILL.md),
[model plans](../gptqmodel-model-plan-tests/SKILL.md), and
[lazy loading](../gptqmodel-lazy-load-smoke-testing/SKILL.md) as applicable.

### Failure, cancellation, and retry after partial progress

Inject failures before mutation, **after a completed mutating block**, and at
publication/cleanup boundaries relevant to the change. Assert the injection
fired. For Hessian/error-feedback retries use correlated inputs, a non-diagonal
Hessian (positive definite after damping), multiple blocks, nontrivial activation
order, and dead columns/static groups when applicable; confirm off-block updates are
nonzero. A diagonal Hessian plus failure at the first allocation cannot expose
reusing a partly updated weight.

Compare retry results and discrete quantization outputs with an independent
fresh run under the numerical skill's rules. Check all coupled state follows
the retry device: weights, Hessians, permutations, group quantizers, and scales.
Test a non-memory exception separately: it must propagate, not trigger an OOM
fallback. Bound retries, require monotonic progress (for example shrinking the
effective batch), and reset retry state between top-level calls.

After cancellation/error, assert hook removal, weight/device restoration,
released temporary references, completed/failed futures, and the absence of
partially published artifacts as applicable. Then run an ordinary forward or
same-key retry: checking an empty hook/cache dictionary alone misses unusable
models and waiters that remain blocked.

For the [cancelled-run contract](../../../docs/cancelled-run-contract.md), test
both calibrated (e.g. GPTQ) and weight-only (e.g. RTN) loops when the guarantee
is shared. Leave one layer completed, one in flight, and one unreached; inject
a one-shot fault at the changed phase, including the final restore if affected.
Assert the original error, then forward in **ordinary grad mode**: an
inference-mode probe can hide unrepaired tensors. Check the three layers'
placement or live offload backing, not a blanket “no meta tensors” rule. A
runnable model is required; re-quantizing that partially processed model is not.

### Concurrency, caches, and temporary storage

State the ownership scope explicitly: module, run, process, device, stream, or
caller lease. Test cold first use, warm reuse, clear/invalidate, and rebuild.
Use barriers/events to force contention; do not rely on sleeps or a loop that
merely submits work sequentially. Count builds/releases and inspect retained
references/storage after work completes, not just peak allocator reservations.

For shared writable scratch, two live streams must have exclusive storage or
a proven lease lasting until asynchronous consumption completes. An allocation
lock does not protect a buffer after the lock is released. Verify same-stream
reuse, different-stream ownership, growth/invalidation, and numerical parity
under overlapping native work. For immutable shared data, verify readiness and
lifetime instead of requiring unnecessary copies.

Exercise owner failure after publishing a future/handle but before completion;
all waiters must finish or receive the error, and a same-key retry must work.
Use bounded joins and an outer subprocess timeout so a regression cannot hang
the suite. Always clean up workers and process-global settings on test failure.
Test overlapping global guards leaving in a different order from entry.

CPU doubles establish ownership logic, not native stream or GIL=0 safety.
Follow the [GIL=0 skill](../gptqmodel-pytorch-gil0/SKILL.md) and ledger for actual
threaded qualification, and the numerical skill for persistent packed storage.

### Configuration reaches the actual consumer

Test non-default values through the public API, resolution, actual dispatch,
and serialization/reload, not only construction or `to_dict()`. Cover enum and
typed backend forms, inherited load policy versus explicit quantize override,
and the default/AUTO control where applicable. Reuse the same config for two
operations; derived runtime policy must not contaminate caller-owned settings.

Warm caches before mutating a nested typed leaf in place, replacing it, and
clearing an override. A fresh resolution must see the new content while an
already-issued immutable snapshot keeps its old value. Construct module A
under policy P1, change the process policy to P2, construct B, then forward A
again: assert each module uses its documented snapshot. Test every setting
that participates in cache/pool identity, not only the most obvious field.

Unknown options, unsupported targets, and conflicting forms need explicit
rejection tests before mutation. Do not silently accept an option the consumer
ignores. Select runtime behavior through config/API arguments, never retired
environment switches.

For loader placement, feed the same checkpoint and explicit map through native
load and the supported Transformers bridge. Include integer ordinal `0` (valid
but falsey), a nonzero control, and an absent target when exercising map
resolution. For mixed CPU/accelerator maps, check AUTO selects an offload-safe
kernel without silently overriding an explicit backend. Inspect the **packed
buffer's actual device** and run a first forward against a reference; a
plausible device map or correct logits alone can hide a copy on every call.

### Calibration representations and repeated replay

Represent the same explicit token rows as each affected supported input form:
row lists, row/column mappings (including non-`dict` Mapping/column objects),
tensors, and one-shot generators. Compare exact prepared IDs, masks, row count,
and row order through the real consumer. A generator test must reach all passes
that consume it (native plus EoRA/GPTAQ when enabled); do not replace it with a
list in the test. Keep re-iterable streams lazy where that is the contract.

Include empty/malformed rows, left/right padding, concatenation, the final
partial batch, and unequal per-sample masks/positions where relevant. Boolean
keep masks and additive causal masks have different semantics: an all-zero
additive mask is not an all-padding boolean mask. Test BOS-present and BOS-less
tokenizers, not the assumption that position zero is always a special token.
Run consecutive batches/layers/parallel branches with distinct context and
assert no stale mask or KV state leaks between them.

For packed token replay, make row **identity** observable, not just row count.
For example, use a padded grid `[[11, 12, PAD], [21, 22, 23]]`: kept row 2
must carry token 21, not PAD. Give each hidden row a distinct marker and assert
its token-aligned kwargs at the replay consumer against hand-listed pairs.
Include excluded sinks when applicable and an all-kept control. Do not use the
production row-index helper to build the expected pairs.

Follow [tokenizer ownership](../gptqmodel-tokenizer-normalization/SKILL.md) for
reusable normalization and [AWQ safety](../gptqmodel-awq-model-onboarding/SKILL.md)
for native replay, routing, and source/consumer qualification. Do not duplicate
those architecture gates with tree-generated fake models.

### Arithmetic and tensor geometry

Follow the [numerical oracle skill](../gptqmodel-torch-oracle-numerics/SKILL.md);
its tolerances, packed-code rules, and hardware requirements remain unchanged.
Cross the **changed mechanism** with supported FP16/BF16/FP32 and dispatch paths,
including outer autocast and inference mode when they affect it.

- Use cancellation (large product plus opposing bias), range endpoints,
  underflow, and nonfinite inputs where supported. Check wide accumulation and
  bias before one output cast; assigning into FP32 cannot undo earlier rounding.
- Test contiguous, row-strided, and column-strided inputs, batched prefill and
  single-token calls. Assert fixture strides before calling the public wrapper;
  `.reshape()` can preserve the bad stride, especially at batch size one.
- Cover zero/one rows and dimensions immediately below/at/above the actual tile,
  group, or alignment boundary, plus ragged final tiles. Assert unsupported
  geometry is rejected or selects a supported fallback before native dispatch.
- Assert output shape/dtype, oracle error, native/fallback selection, and retained
  storage. Compare arithmetic error separately from required output rounding.

Audit sibling backends and all call sites of a changed primitive. A portable
fix does not repair a fused kernel, and a default-path test does not qualify
explicit selection. Add cases for affected siblings or record why they differ.

## 4. Make coverage executable and report it honestly

Put hardware-independent contract tests outside module-wide GPU skip markers;
keep native-device tests separately marked with precise prerequisites. Ensure
the intended CI job collects and runs the new node, not just that the file
exists. No blanket exception-to-skip wrappers, unused test methods, opt-in env
gates, or silently empty parameter sets. Do not widen tolerances or refresh
expected outputs solely to make a failing regression green.

Run from `tests/` when local helpers require it, for example:

```bash
cd tests
python -m pytest test_dynamic_config_invalidation_4475.py -q -ra
```

The PR's validation table must name: invariant/trigger, test node, red evidence,
green command/result, and hardware/dependency skips. State which sibling paths
remain unverified. Mocks, CPU runs, and skipped GPU tests must never be reported
as native qualification. Documentation-only updates need link/frontmatter and
evidence checks, not claims that these new runtime requirements already pass.

## 5. Separate test gaps from structural contract failures

Escalate beyond another isolated test when fixes repeatedly transfer ownership
between consumers, duplicate representation conversion or retry preparation,
or depend on undocumented ordering of mutation, publication, and cleanup.
A missing stride case or boundary fixture alone does not justify a redesign.

Describe the violated invariant, current owner(s), and recurrence evidence.
Propose one explicit ownership/transition contract, a bounded migration using
existing extension points, and conformance tests for every affected consumer.
Distinguish already-landed fixes from proposed architecture and unverified bug
reports. Preserve storage, compatibility, memory, and concurrency constraints;
do not substitute a broad rewrite or dense weight cache for a precise contract.

When the user requests an issue for structural findings, publish the proposal
with evidence links and acceptance criteria, and append `@Qubitium`. Otherwise
report the recommendation for approval rather than opening an unsolicited issue.
The [2026-11-17 contract proposal](https://github.com/ModelCloud/GPTQModel/issues/4615)
illustrates persistence ownership, transactional attempts, asynchronous scratch
lifetime, and immutable execution policy; it is not implemented behavior.
