---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-audit-and-review
description: Audit GPT-QModel quantization pipeline and math, packing, and inference kernel accuracy. File verified findings as they emerge and review fixes in the context of upstream owners and downstream consumers.
---

# GPT-QModel audits and reviews

> **Contracts first.** Read and honour [`contracts/`](../../../contracts/README.md)
> before acting. In particular, read the
> [kernel forward dtype contract](../../../contracts/kernel/forward-dtype.md):
> one activation dtype per module, and output dtype equals input dtype. A
> different dtype on a later call is a bug. Contracts outrank this skill.

## Audit the full path

Start from the public entry points. Follow values and metadata across Python
and native code, and inspect the contracts where stages hand off to each other.
Prioritize these paths in a repository-wide correctness audit:

| Area | What to check |
| --- | --- |
| Quantization pipeline and math | Trace calibration inputs, capture and replay, quantizer updates, and scales, zero points, and codes. Check agreement with the method's math and precision contract. |
| Packing and checkpoints | Trace layout, bit order, padding, dtypes, and metadata from pack and save through load and the kernel's first use. Check the round trip, including supported legacy checkpoints. |
| Inference kernel accuracy | Trace backend selection, unpacking or dequantization, accumulation, and the output epilogue. Exercise the actual native path and relevant fallbacks, shapes, dtypes, and device families. |

Use the [Torch oracle skill](../gptqmodel-torch-oracle-numerics/SKILL.md) for
independent numerical references, tolerance and tie rules, and target-device
evidence. State which paths were exercised and what remains unverified.

## File findings as they are verified

Open a new GitHub issue in `ModelCloud/GPTQModel` for each distinct defect as
soon as there is enough evidence to establish it. Check briefly for an existing
report of the same defect and link it if one exists. Continue the audit after
filing each issue. Add further reproduction or fix evidence to the issue as it
becomes available.

For a dependency-owned defect, also file or link the upstream issue and link the
fix PR when available. Use the
[upstream ownership guide](../gptqmodel-agent-workflow/SKILL.md#contribute-fixes-and-features-upstream)
to route the fix. Keep the GPT-QModel issue focused on its downstream impact.

Include the audited commit, the affected code and entry point, the trigger,
the expected and observed behavior, and the user impact. For numerical bugs,
include relevant backend, device architecture, format, dtype, and shape, plus
the measured error and reference where available. Include a minimal runnable
reproduction in a fenced code block when applicable. When the likely fix is
clear, name the owning code and include a fenced code or diff snippet where
applicable. Label proposed changes and any untested configurations. File a
verified defect promptly even if an optional snippet is still being prepared.

When reviewing a PR, also post findings relevant to the diff in the review
and link any corresponding repository issues.

## Review the root cause and its consumers

- Trace the violated invariant back to the code that owns it. Search for
  callers, sibling implementations, and shared helpers outside the diff before
  proposing a fix. A bad output can originate in an earlier stage.
- Choose the smallest coherent fix at the owning layer. Reuse an existing
  contract or implementation where it fits and avoid duplicating conversions,
  guards, or state across consumers. Check the scope of a shared fix before
  applying it to other backends.
- Identify which callers rely on current behavior and test the next consumer
  through a normal entry point. Check applicable native and fallback paths,
  existing checkpoints, model variants, and eager or compiled execution.
  Follow the [regression-test skill](../gptqmodel-regression-test-design/SKILL.md)
  for red/green evidence and a control case that preserves valid behavior.
