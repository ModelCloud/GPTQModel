---
name: gptqmodel-logic-flow-diagrams
description: Require Mermaid diagrams and before/after tables for major logic-flow changes, both in the PR description and in a committed docs/ reference page. Use when a change alters control flow, resolution/selection order, dispatch, data or tensor lifecycle, or removes or replaces a decision mechanism.
---

# Logic-flow diagrams for major changes

A change that moves *decision-making* is not fully reviewed until a reader can
see the old flow, the new flow, and the boundary between them without opening
the diff. This skill defines the minimum documentation bar for those changes and
where it lives.

## When this skill applies

Apply it when a change does any of the following:

- Alters **control flow**: new/removed stages, reordered phases, new early
  exits, new failure paths, changed fallbacks.
- Replaces a **decision mechanism**: heuristics replaced by declared metadata,
  name matching replaced by explicit tags/paths, a config key changing which
  code path runs, a plugin/registry becoming authoritative.
- Changes **resolution or selection order**: device placement, backend
  selection, kernel dispatch, layer/module resolution, checkpoint resume order.
- Changes **data or tensor lifecycle**: what is materialized, cached, freed,
  offloaded, repacked, or persisted across stages.
- Moves work **between processes, devices, or stages**: multi-GPU split,
  lazy-load/materialize boundaries, disk offload, streaming.
- Introduces a **fail-closed** or **fail-open** contract change: previously
  guessed, now raises; previously raised, now defaults.

Not required for: localized bug fixes, numerical tweaks inside one function,
doc-only changes, or dependency bumps. When unsure, ask: *"would a competent
reviewer need a diagram to be confident the flow is right?"* If yes, apply it.

## Requirements (minimum)

For every qualifying PR, produce **all three**:

1. **Flow diagram(s)** of the new logic in the PR description.
2. **Before/after comparison** — either a second diagram of the old flow or a
   table mapping each old mechanism to its replacement, with `file:line`
   pointers where the old logic lived.
3. **A committed reference page** under `docs/` that keeps the same diagrams
   for future readers (see [Where the docs page goes](#where-the-docs-page-goes)).

Diagrams are the floor, not the ceiling. Add whichever of these actually
explain the change:

| Artifact | Use for |
| --- | --- |
| Mermaid flowchart | control flow, decision trees, dispatch, fallbacks |
| Mermaid sequence diagram | multi-actor flow: caller → loader → looper → kernel |
| Mermaid state diagram | lifecycle, resume/restart, failure and retry states |
| Mermaid `classDiagram` / ER | new core types, ownership, data model |
| Before/after table | 1:1 replacement of mechanisms or semantics |
| Change inventory table | file → add/delete counts → one-line purpose |
| Measured numbers table | what the bug cost or what the fix changed, with units |
| Declaration/token syntax block | new metadata grammar readers must write |

Every diagram must carry information that prose would take a paragraph to say.
If a diagram restates one sentence, drop it.

## Accuracy rules (non-negotiable)

- Diagrams describe **code that exists in the PR**, not the intended design.
  Re-read the diff before drawing; do not draw from the commit message.
- Name real symbols: `ModulePlan.from_tree`, `_resolve_effective_module_tree`,
  `plan.expert_root()`. Placeholders like `do_stuff()` are not acceptable.
- Mark **failure and ambiguity paths** explicitly, visually distinct from the
  success path (e.g. red fill). Silent fallbacks are the usual root cause of
  the bug being fixed — show where they were removed.
- Show **where a decision is made** and what it is made from. "Resolves role"
  is too vague; "resolves role from declared `:q` tag" is the point.
- Quantitative claims need a source: measured, counted from the diff, or
  explicitly labelled as an estimate. Never invent pass counts.
- Do not include aspirational or follow-up flows as if implemented. Put those
  in a clearly separated "Not in scope / follow-up" section.

## PR description structure

Use this skeleton for qualifying PRs; drop sections that genuinely do not apply.

```markdown
## TL;DR
What changed, the net diff size, and the one-line architectural consequence.

## 1. The problem
Diagram of the OLD flow + table of the mechanisms being removed, with file:line.

## 2. The new flow
End-to-end diagram from entry point to the components that consume the result.

## 3. New core (if any)
Diagram of the new abstraction: how it is built, what it exposes.
Syntax/token table if the change introduces new metadata.

## 4. Per-subsystem detail
One small diagram per non-obvious decision; before/after where it helps.

## 5. Failure and ambiguity contract
State diagram + table: situation → before → after.

## 6. Change inventory
Table: area, file, ± lines, purpose.

## 7. Tests / guarantees
Table: test name → the invariant it pins.

## 8. Validation
Exact commands and measured results.

## 9. Reviewer notes
Trade-offs, deliberate scope limits, open questions.
```

## Mermaid authoring rules (GitHub-rendered)

These avoid the failures that actually happen on GitHub:

- **Never wrap a ```` ```mermaid ```` fence in a raw HTML block** (`<div>`,
  `<table>`). GitHub does not parse markdown inside raw HTML, so the fence
  renders as literal text. Use plain block elements or line breaks instead.
- Keep labels short and use `<br/>` for line breaks inside a label. Avoid
  `<code>` tags inside labels; plain text is safer across renderers.
- Avoid `{` and `}` in label text — Mermaid parses them as shape syntax.
  Write `specs map`, not `specs{}`.
- Prefer `flowchart TD/LR` over `graph`. Use `stateDiagram-v2` only for real
  state machines, and give states `state "label" as ID` when the label has
  spaces or punctuation.
- Quote any label containing `.`, `(`, `)`, `:`, `/`, `#`, or an arrow.
- Validate before publishing: confirm fences are balanced and every block
  starts with a known diagram type. A quick local check:
  ```bash
  python - <<'PY'
  import re, pathlib
  text = pathlib.Path("PR_BODY.md").read_text()
  blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
  print(len(blocks), "blocks", text.count("```"), "fences")
  for i, b in enumerate(blocks):
      print(i, b.strip().splitlines()[0])
  PY
  ```
- After the PR is opened, re-read the live body from the API and confirm the
  block count survived: `body.count("```mermaid")`.

## Where the docs page goes

Commit the reference page next to the subject it describes, not in a new
parallel tree:

| Change subject | Location |
| --- | --- |
| Model/definition resolution, module trees | `docs/model-tree-resolution.md` |
| Loader, writer, checkpoint, packing | `docs/<topic>.md` (existing file if present) |
| Device placement, multi-GPU | `docs/multi_gpu_quantization.md` |
| Quantization math, kernels, formats | existing `docs/` page for the method |
| Looper stages, resume | `docs/looper-checkpoint-redesign.md` |

Rules for the page:

- Title it for the mechanism, not the PR: *"Model tree resolution"*, not
  *"PR 3565"*.
- Open with **what the flow is and why it is shaped that way**, then the
  diagrams, then the contracts (fail-closed rules, extension points).
- Keep diagrams in sync with the code. If a later PR changes the flow, update
  the page in the same PR — a stale diagram is worse than none.
- Link the page from `README.md` where related features are discussed, so it is
  discoverable rather than orphaned.
- Reference the introducing PR number in one line so readers can find the
  rationale, but do not write the page as a changelog.

## Review checklist

- [ ] Change qualifies (control flow, decision mechanism, lifecycle, or
      ordering), or the PR states why diagrams are not needed.
- [ ] New-flow diagram with real symbol names, from entry point to consumers.
- [ ] Old mechanism shown or tabulated with `file:line` evidence.
- [ ] Failure/ambiguity paths visible and visually distinct.
- [ ] Numbers are measured or explicitly labelled as estimates.
- [ ] Follow-up work separated from implemented behavior.
- [ ] `docs/` reference page committed and linked from `README.md`.
- [ ] Mermaid blocks balanced, no raw-HTML wrapping, no `{}` in labels.
- [ ] Live PR body re-read after publishing to confirm rendering input.

## Reference example

PR #3565 (`fix(models): resolve module roles from explicit model trees`) is the
worked example: ten diagrams covering the removed name heuristics, the
resolve/freeze/compile path, the new `ModulePlan` queries, per-subsystem
resolution, lookup ambiguity handling, and the fail-closed contract. Its
distilled reference page is `docs/model-tree-resolution.md`.
