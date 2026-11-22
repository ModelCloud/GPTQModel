---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-agent-workflow
description: Use when starting or delegating GPT-QModel work, submitting fixes or features upstream, or following a PR through readiness, CI, and review.
---

# Agent workspaces and pull requests

> **Contracts first.** Read and honour [`contracts/`](../../../contracts/README.md)
> before acting. In particular, read the
> [kernel forward dtype contract](../../../contracts/kernel/forward-dtype.md):
> one activation dtype per module, and output dtype equals input dtype. A
> different dtype on a later call is a bug. Contracts outrank this skill.

## At a glance

Work in a fresh clone; publish as a draft and promote the same PR only after
GitHub confirms it is conflict-free. Ready for review does not authorize merge.

## Start or delegate work

Clone the remote origin URL into a directory unique to the agent and task;
branch from current `origin/main`, never commit on `main`. Give sub-agents
the task and remote URL, not a parent checkout, local branch or linked
worktree. Reuse a parent workspace only on explicit instruction, not merely
because work was delegated. For issue work, also follow the
[claim workflow](../gptqmodel-issue-claims/SKILL.md#workflow).

## Contribute fixes and features upstream

Submit every validated bug fix as a PR to the repository that owns the faulty
behavior. Send GPT-QModel fixes to `ModelCloud/GPTQModel` even when you find the
bug in a fork or downstream integration. New features are welcome as upstream
PRs too. Use these as likely owners, then trace the call and check where the
contract is implemented. A caller that misuses a dependency API owns that
integration bug.

| Behavior | Likely upstream repository |
| --- | --- |
| GPT-QModel's quantization, packing, loading, kernel dispatch, or other project behavior | [ModelCloud/GPTQModel](https://github.com/ModelCloud/GPTQModel) |
| Tokenizer loading, normalization, or template compatibility | [ModelCloud/Tokenicer](https://github.com/ModelCloud/Tokenicer). Follow the [tokenizer skill](../gptqmodel-tokenizer-normalization/SKILL.md). |
| Defusing model structures for use by the quantization pipeline | [ModelCloud/Defuser](https://github.com/ModelCloud/Defuser) |
| Logger or progress API behavior | [ModelCloud/LogBar](https://github.com/ModelCloud/LogBar) |
| Device discovery, properties, or SMI queries | [ModelCloud/Device-SMI](https://github.com/ModelCloud/Device-SMI) |

Use a task branch in the owning repository and follow its contribution rules.
Add a focused regression that reproduces the bug and validates the fix there.
Use the same ownership check for features and test the new public behavior.
If an upstream fix already exists, link it. If you are blocked from submitting
a fix, file or link the upstream issue with the reproduction and report the
blocker.

For any required GPT-QModel integration, link the owning PR, consume the
upstream release when it is available, and verify the affected GPT-QModel entry
point. Keep reusable logic in the owning library.

## Open a draft, then mark it ready

1. Push and open **as a draft**, even if validated. Target `main` unless
   explicitly directed otherwise. In the PR template, write one
   `Fixes #<issue-number>` line per issue resolved (cross-repo:
   `Fixes OWNER/REPO#<issue-number>`). Use non-closing references for related
   or partial work.
2. Review the diff, validate, fetch the current base from `origin`, then check
   GitHub: `gh pr view <pr> --json baseRefName,baseRefOid,headRefOid,mergeable`.
   Match the base name and revision to the fetched base and the head revision
   to the pushed commit. Recheck if either changes.
3. **Conflict:** stay in draft, resolve against the fetched base, rerun
   affected checks, push to the same PR and recheck. **Unknown:** wait and
   retry; if still unverifiable, leave draft and report why. **MERGEABLE:**
   once the review and validation above are complete, mark ready for review.
   For a claimed issue, first confirm the PR description includes the
   [required before/after measurements](../gptqmodel-issue-claims/SKILL.md#workflow)
   for applicable metrics.

## After marking ready

For claimed-issue PRs, follow the same PR through these checks:

1. Wait for GitHub Actions to finish the `Ruff Check` workflow's
   `ruff-check` job (the repository's code-quality gate) and any other
   configured code-quality scans for the current PR head. Confirm they passed.
   If a check fails, diagnose it, fix failures caused by the change, and wait
   for the checks on the updated head.
2. Check the PR for review summaries, PR comments, and unresolved inline
   review threads from people and automated reviewers. Address actionable
   notes, reply with the fix or rationale, and resolve threads once addressed.
   If you push a fix, repeat the CI and review checks for the new head.
3. Fetch the current base and recheck GitHub mergeability against the pushed
   head using the same revision checks as above. Resolve any new conflicts on
   the same PR and repeat the CI, review, and mergeability checks. Report any
   remaining check, review, or mergeability blocker.
