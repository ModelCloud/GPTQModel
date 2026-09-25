---
name: qvq-zml-kernel-contract
description: Synchronize QVQ hardware-optimized kernels with ZML/StableHLO/XLA admission, autotuning, and runtime dispatch; catch silent dense fallbacks. Use when QVQ or ZML changes a CUDA family, ABI, shape specialization, tuning control, pin, custom call, or inference integration.
---

# QVQ-ZML Kernel Contract

Treat a QVQ optimization as integrated only after the pinned ZML path can select and execute it. A compiled kernel, a passing Torch benchmark, or a fresh dependency pin is not sufficient.

## Establish the two source-of-truth revisions

1. Fetch both repositories and work from fresh upstream branches. QVQ uses `origin/main`; ZML uses `origin/master`.
2. Read ZML's immutable QVQ revision in `third_party/qvq/repo.bzl`.
3. Compare the pinned QVQ payload with current QVQ, including the raw ABI header/source, CUDA families, P32 subfiles, and Bazel metadata. A merge commit may differ while the relevant payload is identical.
4. Run `scripts/audit_qvq_zml_contract.py --qvq-root <qvq> --zml-root <zml>` before editing and after validation.
5. Never validate a ZML integration only through `--override_repository`; the final gate must build and run against the immutable remote pin.
6. Keep this skill, its audit script, and its matrix template byte-identical in QVQ and ZML; update both copies in paired PRs.

Use the repository's git/PR workflow skill for branch and push mechanics. This skill governs the cross-repository technical contract.

## Recurring dispatch failures to prevent

| Observed failure | Required gate before a full-suite benchmark |
| --- | --- |
| Fast CUDA specialization existed but ZML did not select it | Match the pinned ABI's exact shape/algorithm predicate to ZML policy and inspect a production-shaped native launch. |
| ZML selected the algorithm but XLA rejected the composite and emitted dense TF32 | Check the actual XLA rewriter guard and optimized HLO for each projection; count native versus fallback launches. |
| Rewrite lived in a plausible wrapper bypassed by model `forward` | Trace the real model call graph and prove the expected production Step changed before timing 1,209 rows. |
| Autotune/cache change retuned an unrelated decode fusion | Match compiled executable, cache, decode HLO/kernel selection, warmup, and batch/row identity across control and candidate. |
| Shape policy overrode an explicit `WithConfig`/autotune request | Keep automatic policy at the model boundary; test that explicit geometry remains intact. |
| Diagnostic environment switch became necessary to reach production kernel | Select by architecture/ABI/shape capability and expose the selected route and fallback reason in telemetry. |
| Correct logits were mistaken for optimized-path proof | Require projection-scoped dispatch evidence; numerical parity alone is insufficient. |

## Build a bidirectional coverage matrix

For every changed or exported QVQ family, record one row using [references/coverage-matrix.md](references/coverage-matrix.md). Include:

- ABI version, symbol, algorithm ID, exact CUDA architecture/family target and compute-capability admission, dtype/layout, and output accumulation type.
- Exact M/K/N and transition-bit domain; grouped versus single projection; split, BM/BN/BK, stage, warp, and workspace constraints.
- Public tuning controls and which values are aliases rather than independent axes.
- CUDA graph capture/replay behavior and cache-key fields.
- ZML loader, policy/admission function, StableHLO attributes, XLA validation, workspace calculation, and runtime dispatch proof.

Every row must end in exactly one disposition:

- `wired`: reachable through the pinned production path and proven at runtime.
- `intentionally-unwired`: has an owner, concrete reason, regression test, and tracking issue/PR.
- `not-applicable`: cannot be consumed by ZML, with a technical reason.

Missing or implicit disposition is a failure. Perform the reverse check too: every nonzero algorithm or specialized geometry emitted by ZML must exist in the pinned QVQ ABI with identical constraints.

## Prove the production call path before the expensive model gate

Trace the *actual model entry point* for every optimized projection and phase, including rank-correction on/off and grouped/single paths. Record the chain from model `forward` through its configured QVQ call, composite/lowering, XLA admission, FFI symbol, and native kernel. Do not assume an optimization placed in a plausible public wrapper is reachable: Llama's QVQ projection called `p32WindowMatmulWithConfig` directly, so an M1920 rewrite in `p32WindowMatmul` compiled and passed tests but never executed in the full model.

Before running the full quality/throughput suite, compile one **production-shaped** Step with the real pinned model and compare candidate versus matched control:

- State the expected per-module/per-layer launch-count and M/K/N/rate/algorithm changes, then inspect optimized HLO and projection-scoped runtime kernel/counter evidence. Fail the path gate if the expected change is absent, even when outputs are exact and the candidate builds.
- Verify gate/down, Q/K/V, rank correction, and small-M decode separately when their routes differ. A total QVQ launch count cannot prove that the changed projection used its intended kernel.
- Check the caller-side shape predicate, ZML composite attributes, XLA rewriter guard, pinned QVQ raw-ABI predicate, and selected native symbol for the same tuple. Include a near-miss shape that retains its documented fallback.
- Apply automatic shape rewrites at the model-policy boundary, not in an explicit `WithConfig`/autotune entry point: a production fix must not silently override a caller's requested algorithm or geometry. Test both the automatic route and an explicit-config near miss.
- Hold XLA/PJRT build, autotune cache, compiled decode algorithms, graph mode, and warmup identity fixed. A prefill-only source change may independently retune decode: the M1920 investigation observed `64×128` versus `128×128` Triton decode tiles and misleading token drift until the executable policy was matched.
- Read the runner-ready telemetry before scoring: assert the intended attention backend/page size, rank-correction phases, prefill bucket, KV pool, allocator, CPU/GPU affinity, and compiled batch. A server command that omits `ZML_LLAMA_ATTENTION=fa2`, for example, starts a valid but incomparable Triton run.
- If a local kernel wins but full-model timing is flat, check this route proof *before* treating the mechanism as slow. Conversely, HLO attributes or selected Zig configuration alone are not proof that the native launch occurred.

For an exact shape-specialized path, production selection should follow device/ABI/shape capability automatically. Keep diagnostic environment switches out of the promoted dispatch policy unless they are an intentional, user-visible control. Record the selected path and the reason for fallback in readiness/telemetry so a future benchmark can verify it without inferring from throughput.

## Propagate QVQ changes through every layer

When QVQ changes, inspect and update all affected layers in the same workstream:

1. QVQ implementation and framework-neutral ABI.
2. QVQ Bazel exports and sparse-checkout reachability.
3. ZML QVQ pin and Zig ABI struct/symbol loader.
4. ZML policy and shape admission in `zml/qvq.zig`.
5. StableHLO custom-call attributes and XLA parsing/validation/workspace logic in `third_party/xla/qvq-p32-xla-integration.patch`.
6. Compilation/autotune cache identity. Add every field that can change generated code or dispatch, including compute capability/architecture-family target, layout/swizzle/vectorization, pipeline stages, cluster/CTA geometry, split policy and correction mode; bump the relevant version when old cache entries are unsafe.
7. Inference-runner defaults only when production policy changes. Keep core integration in QVQ/ZML.

Do not broaden admission beyond the ABI predicate. Do not leave a newly optimized family hidden behind the portable fallback without an explicit matrix disposition.

## Catch XLA's silent dense fallback

XLA may decline an optimized QVQ composite and compile a numerically valid dense GEMM instead. A passing output oracle, selected Zig config, native CUDA microbenchmark, or nonzero aggregate QVQ counter can therefore give false confidence about **this projection**.

Treat XLA admission as an independent capability gate, not an automatic consequence of a QVQ ABI or Zig-policy change. When a specialized shape is added, changed, or newly made the default, inspect the actual rewriter guard and the composite attributes XLA parses from the compiled HLO; a matching algorithm name elsewhere in the patch is not evidence that the guard accepts the production tuple. An unintended dense/TF32 fallback is an integration failure even when logits remain correct.

For every new or changed algorithm, shape, or BM/BN/split geometry:

1. Compare the exact admissible tuple in three places: ZML's emitted attributes, XLA's composite-rewriter predicate, and the pinned QVQ raw-ABI predicate. Include M/K/N, transition bits, grouping, split/workspace, and architecture. A difference needs an explicit `intentionally-unwired` disposition, not a presumed fast path. Inspect the predicate itself; finding an algorithm ID or BM value elsewhere in the patch is not enough.
2. For each newly admitted specialization, exercise the XLA rewriter with its **exact production attributes**. Assert that the positive tuple becomes the intended QVQ custom fusion; change one guarded field (for example BM, transition bits, K, or N) and assert the near miss is rejected or follows its documented semantic fallback. Run these tests in addition to the static contract audit.
3. On a warmed production-shape run, inspect compiled HLO/custom-call evidence **and** projection-specific kernel names or scoped dispatch counters. Verify the expected optimized calls occur and dense fallback calls for the same projection do not. A warm cache entry or Zig-selected config alone does not prove that this XLA compilation admitted the tuple; record the compiled executable/build identity and its actual launches. Other legitimate dense GEMMs in the model do not count as failures.
4. If the optimized route cannot be distinguished from fallback at runtime, add scoped telemetry or a trace before claiming an end-to-end speedup. Record expected and observed call counts, not only latency or logits. Do not infer kernel selection from throughput or numerical parity.

After changing an XLA predicate or custom-call attribute, invalidate or distinguish the previous compiled-executable/cache identity, then check a cold compile and a warmed replay of the **same** production tuple. Make the production-shape test fail if the expected native projection launches are absent or if that projection takes a dense fallback. Keep intentional fallback cases explicit and tested; do not ban unrelated dense GEMMs.

The M960 W3 gate incident is the concrete failure: QVQ accepted BM160/R10, but XLA originally rejected it and emitted `sm90_xmma_gemm_f32f32`; after matching XLA admission, the warm trace showed 15 native R10 gate calls and no dense fallback for that gate. Treat the static audit as an early warning, not as a substitute for the runtime proof above.

## Validate reachability, quality, and speed

Validate in this order:

1. Run static contract audit and repository unit tests.
2. Test QVQ directly against the existing FP32 oracle and, where useful for boundary arbitration, FP64. Preserve the repository's accepted error policy; small bounded rounding drift is not an automatic rejection, but investigate mitigations that retain both speed and accuracy.
3. Test ZML against the exact pinned QVQ revision with local overrides disabled.
4. Cover W2.5, W3, and W3.5, representative model Q/K/N shapes, B1 decode, the production batch, and full-context prefill. Cover grouped and single-projection routes where supported.
5. Capture and replay a changed-input CUDA graph through the external runtime path.
6. Prove dispatch. Use runtime counters plus Nsight Systems/Compute or SASS/kernel-name evidence. A selected config without a launch of the expected optimized symbol is not proof.
7. Benchmark only after autotuning and warmup. Compare same host/GPU/session, synchronize timing boundaries, and report useful prefill tok/s, aggregate decode tok/s, per-stream decode tok/s, launch count, and fallback count.
8. Run an end-to-end held-out quality gate when dispatch or arithmetic changes. Keep calibration/quantization data disjoint from post-evaluation data.

Fail if parity passes only because the optimized path was never entered. Add a negative test in which a near-miss shape is rejected or falls back safely.

## Coordinate paired PRs

Land ABI/kernel changes in QVQ first. Rebase ZML from current `origin/master`, pin the immutable QVQ commit, and land the consumer PR second. The QVQ PR must state its ZML disposition; the ZML PR must link the QVQ PR and attach the completed matrix and runtime proof.

Before handoff, report:

- QVQ source and pinned commits, plus whether payloads are exact or differ.
- Families added/changed and their three-state disposition.
- Tests, graph replay, runtime dispatch evidence, quality gates, and benchmarks.
- Remaining fallbacks or intentionally unwired families with owners.


## NVIDIA A100+ external ABI requirements

When a QVQ executable exposes architecture-sensitive layout or scheduling, the
external contract must carry enough information to reproduce the exact launch:

- Ampere: async-copy stage/vector/alignment assumptions and MMA/layout family.
- Hopper: TMA tensor-map/swizzle, stage count, WGMMA operand ownership,
  warpgroup/CTA geometry, dynamic SMEM and cluster requirement.
- Blackwell: exact architecture/family target, TCGen05/TMEM requirements,
  TMA/layout and block-scale metadata where applicable.

A generic algorithm name such as `hopper_direct` or `blackwell_fp4` is not
a complete cache key. Reject an external request whose compiled specialization
does not exactly honor the requested geometry/layout.
