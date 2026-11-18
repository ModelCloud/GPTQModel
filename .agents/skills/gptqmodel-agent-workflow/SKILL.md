---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-agent-workflow
description: Set up a separate GPT-QModel clone for every agent or sub-agent, publish draft pull requests, and confirm mergeability before marking them ready. Use when starting or delegating work or publishing a pull request.
---

# Agent workspaces and pull requests

## At a glance

- **Start:** each agent or sub-agent clones the remote into its own directory and branches from current `origin/main`.
- **Publish:** open all new PRs as drafts; use `Fixes #<issue-number>` in the body for each issue fixed.
- **Mark ready:** review the PR and confirm GitHub reports the latest pushed head mergeable into the current remote base.

## Start or delegate work

1. Clone the remote `origin` URL into a new directory unique to the agent and
   task. Create a task branch from the new clone's `origin/main`; do not commit
   on `main`. A new branch or linked worktree in the parent's clone does not
   count as a separate clone.
2. Give each sub-agent its task and remote URL, not the parent's workspace.
   Inherit or reuse the parent's workspace only when explicitly directed to
   do so. A handoff alone is not permission to reuse a checkout.
3. For GitHub issues, also follow the
   [issue claims workflow](../gptqmodel-issue-claims/SKILL.md).

## Open a draft, then mark it ready

1. Push the task branch and open the PR **as a draft**, even when the change is
   already validated. Use `main` as the base unless explicitly directed
   otherwise. Fill in the PR template. Put one GitHub closing line,
   `Fixes #<issue-number>`, per issue the PR actually resolves. For an issue in
   another repository use `Fixes OWNER/REPO#<issue-number>`. Use a non-closing
   reference for partial or related work.
2. Before marking ready, review the diff, run the applicable validation, and
   fetch the current base from `origin` (`origin/main` by default). Confirm
   that GitHub reports the PR's current pushed head mergeable against that
   base. With GitHub CLI, inspect
   `gh pr view <pr> --json baseRefName,baseRefOid,headRefOid,mergeable`.
   Check the base and head revisions as well as the `MERGEABLE` result.
3. If GitHub reports a conflict, keep the PR in draft, resolve it against the
   fetched base, rerun affected checks, and push the fix to the same PR.
   Recheck after any head or base change. If the result is unknown, wait for
   GitHub to finish computing it and check again; if you cannot confirm
   mergeability, leave the PR in draft and report why.
4. Once reviewed, validated, and confirmed conflict-free, mark the same PR
   **Ready for review**. This does not authorize merging it.
