---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-logic-flow-diagrams
description: Require Mermaid diagrams and before/after tables for major logic-flow changes, both in the PR description and in a committed docs/ reference page. Use when a change alters control flow, resolution/selection order, dispatch, data or tensor lifecycle, or removes or replaces a decision mechanism.
---

# Logic-flow diagrams for major changes

## At a glance

A reviewer must be able to trace the old decision, its replacement and its
failure paths without reconstructing the diff. Put the new flow and old/new
comparison in the PR; keep the same diagrams in a mechanism page under
`docs/`, linked from `README.md`.

## When this skill applies

Apply before publishing changes to stages, exits or fallbacks; the source of
a decision (e.g. names vs declared tags); resolution, dispatch or selection
order; tensor materialization, caching, offload or persistence; boundaries
between processes, devices or stages; or fail-open/fail-closed behaviour.
A local fix, numerical tweak, docs-only change or dependency bump need not
include a diagram. State why in the PR. If a reviewer needs a diagram to
verify the flow, include it.

## Build the PR description from the diff

1. **Summary and problem:** give the net diff size and architectural effect.
   Show the removed mechanism in an old-flow diagram or before/after table
   with `file:line` pointers.
2. **New flow:** diagram entry point to consumers using shipped symbol names,
   not placeholders or intent. Show where each decision is made and from what
   input (e.g. a declared `:q` tag). Mark failures, ambiguity and removed
   silent fallbacks; make failure paths visually distinct from success.
3. **Details as needed:** show new abstractions/ownership and any metadata
   syntax readers must write. Use flowcharts for dispatch, sequence diagrams
   for multiple actors, state diagrams for lifecycle/retry, class/ER diagrams
   for types. Diagram non-obvious subsystem decisions; for changed failure
   semantics include the state diagram and a situation → before → after table.
   Omit diagrams that merely repeat one sentence.
4. **Evidence:** map changed areas/files to ± lines and purpose, tests to the
   guarantees they pin, and exact validation commands to measured results.
   Give units for measured costs. Source numbers from measurements or the
   diff; label estimates, never invent pass counts.
5. **Reviewer notes:** identify trade-offs, scope limits and open questions.
   Separate unimplemented follow-ups from the shipped flow.

## Mermaid authoring rules (GitHub-rendered)

- Use standalone `mermaid` fences, not raw HTML wrappers such as `<div>` or
  `<table>`: GitHub won't parse Markdown inside them.
- Keep labels short; use `<br/>`, not `<code>`. Avoid braces in labels.
  Quote labels containing `.`, parentheses, `:`, `/`, `#` or an arrow.
- Prefer `flowchart TD` or `flowchart LR` to `graph`. Use `stateDiagram-v2`
  only for a state machine; alias complex labels with `state "label" as ID`.
- Check balanced fences and a known type on each block. After opening the PR,
  re-read its body via the API and compare its Mermaid block count to the
  local count. A quick local inspection:

  ```python
  import pathlib, re
  text = pathlib.Path("PR_BODY.md").read_text()
  blocks = re.findall(r"^```mermaid\n(.*?)^```$", text, re.M | re.S)
  print(len(blocks), "blocks:", [b.strip().splitlines()[0] for b in blocks])
  ```

## Where the docs page goes

Update the existing mechanism page in the **same PR** so diagrams stay in
sync. For model trees use `docs/model-tree-resolution.md`; device placement:
`docs/multi_gpu_quantization.md`; looper/resume:
`docs/looper-checkpoint-redesign.md`. For loader, writer, packing, checkpoint,
math, kernel or format changes, use the corresponding `docs/<topic>.md`
(existing if present), not a parallel docs tree.

Title the page for the mechanism, not the PR. Open with what the flow does
and why, then diagrams, failure contracts and extension points. Link it from
the related section of `README.md` and cite the introducing PR in one line;
the page is not a changelog. Worked example: [PR #3565](https://github.com/ModelCloud/GPTQModel/pull/3565)
and [model tree resolution](../../../docs/model-tree-resolution.md).
