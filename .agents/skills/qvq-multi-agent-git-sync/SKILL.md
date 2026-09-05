---
name: qvq-multi-agent-git-sync
description: Synchronize commits safely on shared ModelCloud/QvQ branches where multiple agents work concurrently. Use in this QvQ repository before every push, after another agent updates the active branch, when pull/rebase reports conflicts, or when confirming that local and remote branch tips match.
---

# QvQ multi-agent Git synchronization

Treat every QvQ feature branch as concurrently writable. Never push a stale local tip.

## Preserve scope

- Confirm `origin` fetch and push URLs target `https://github.com/ModelCloud/QvQ.git`.
- Push only the active QvQ feature branch unless the user explicitly names another branch.
- Preserve unrelated tracked and untracked collaborator files. Stage only the intended change.
- Never resolve a conflict by blindly choosing all of `ours` or `theirs`. Read the base, local change, and incoming
  change; integrate both intents when compatible.

## Start every new branch from the live remote main tip

Multiple agents may merge pull requests while another task is running. Before creating any new feature or benchmark
branch, fetch `origin` again and branch directly from the newly resolved `origin/main`. Do not branch from a local
`main`, the current feature-branch `HEAD`, a merge commit on an old feature branch, or an `origin/main` value fetched
earlier in the session.

1. Preserve or commit the current task's intended changes and leave unrelated collaborator files untouched.
2. Confirm `origin` still points at ModelCloud/QvQ, then run `git fetch origin --prune` immediately before branching.
3. Record `git rev-parse origin/main` as the new branch's base.
4. Create the branch or worktree with `origin/main` as the explicit start point, for example:

   ```bash
   git switch -c <new-branch> --no-track origin/main
   # Or, when preserving the current worktree:
   git worktree add -b <new-branch> <new-worktree> origin/main
   ```

5. Before making changes, require both `git rev-parse HEAD` and `git merge-base HEAD origin/main` to equal the
   recorded remote-main SHA, and require `git log --oneline origin/main..HEAD` to be empty.

If selected commits from an earlier branch must carry forward, first create the clean branch from the latest
`origin/main`, then cherry-pick only those reviewed commits. Never use the earlier branch or its post-merge tip as the
new branch point. Re-fetch again before the first push because `origin/main` may have advanced during setup.

## Synchronize before every push

1. Inspect `git status --short`, the active branch, its upstream, and `git remote -v`.
2. Finish and verify the intended local work before publishing it.
3. Commit the intended files. Do not include unrelated artifacts merely to obtain a clean tree.
4. Immediately before pushing, run:

   ```bash
   git pull --rebase origin "$(git branch --show-current)"
   ```

5. If new commits arrive, inspect their diff and rerun tests proportional to the overlap and risk. A clean textual
   rebase is not proof of semantic compatibility.
6. Push the explicit branch to `origin`.
7. Compare `git rev-parse HEAD` with `git ls-remote origin refs/heads/<branch>`. Do not report success unless the
   hashes match.

If another agent pushes between steps 4 and 6 and the push is rejected, return to step 4. Never force-push a shared
branch unless the user explicitly authorizes rewriting that exact branch.

## Resolve conflicts

1. List every unmerged file with `git status --short`.
2. Inspect conflict markers and the relevant incoming commit. Preserve both agents' non-conflicting behavior, tests,
   and documentation.
3. Use `apply_patch` for the resolution, stage each resolved file, and continue with
   `GIT_EDITOR=true git rebase --continue` when retaining the existing commit message.
4. Repeat until the rebase completes.
5. Run `git diff --check`, focused tests for all overlapping areas, and lint/type checks required by the touched code.
6. Pull/rebase once more immediately before push; resolving a conflict does not waive the before-push sync rule.

Abort the rebase only when the intents cannot be reconciled safely without a user decision. Never discard
collaborator commits to make a conflict disappear.

## Report publication

Report the final branch, commit hash, remote-hash confirmation, tests rerun after the last integration, and any
preserved untracked files. Mention conflicts only when they occurred and summarize how both intents were retained.
