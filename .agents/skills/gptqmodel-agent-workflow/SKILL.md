---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-agent-workflow
description: Use when starting or delegating a GPT-QModel task, opening a draft PR, or following it through readiness, CI, and review.
---

# Agent workspaces and pull requests

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
