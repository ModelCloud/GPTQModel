---
name: qvq-zml-kernel-contract
description: Synchronize QVQ hardware-optimized kernel families with ZML/StableHLO/XLA ABI loading, admission, autotuning, cache identity, and runtime dispatch. Use whenever QVQ adds or changes a CUDA kernel, ABI, shape specialization, tuning control, workspace rule, or graph behavior, and whenever ZML changes its QVQ pin, custom calls, shape policy, or inference integration.
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

## Build a bidirectional coverage matrix

For every changed or exported QVQ family, record one row using [references/coverage-matrix.md](references/coverage-matrix.md). Include:

- ABI version, symbol, algorithm ID, SM target, dtype/layout, and output accumulation type.
- Exact M/K/N and transition-bit domain; grouped versus single projection; split, BM/BN/BK, stage, warp, and workspace constraints.
- Public tuning controls and which values are aliases rather than independent axes.
- CUDA graph capture/replay behavior and cache-key fields.
- ZML loader, policy/admission function, StableHLO attributes, XLA validation, workspace calculation, and runtime dispatch proof.

Every row must end in exactly one disposition:

- `wired`: reachable through the pinned production path and proven at runtime.
- `intentionally-unwired`: has an owner, concrete reason, regression test, and tracking issue/PR.
- `not-applicable`: cannot be consumed by ZML, with a technical reason.

Missing or implicit disposition is a failure. Perform the reverse check too: every nonzero algorithm or specialized geometry emitted by ZML must exist in the pinned QVQ ABI with identical constraints.

## Propagate QVQ changes through every layer

When QVQ changes, inspect and update all affected layers in the same workstream:

1. QVQ implementation and framework-neutral ABI.
2. QVQ Bazel exports and sparse-checkout reachability.
3. ZML QVQ pin and Zig ABI struct/symbol loader.
4. ZML policy and shape admission in `zml/qvq.zig`.
5. StableHLO custom-call attributes and XLA parsing/validation/workspace logic in `third_party/xla/qvq-p32-xla-integration.patch`.
6. Compilation/autotune cache identity. Add every field that can change generated code or dispatch; bump the relevant version when old cache entries are unsafe.
7. Inference-runner defaults only when production policy changes. Keep core integration in QVQ/ZML.

Do not broaden admission beyond the ABI predicate. Do not leave a newly optimized family hidden behind the portable fallback without an explicit matrix disposition.

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
