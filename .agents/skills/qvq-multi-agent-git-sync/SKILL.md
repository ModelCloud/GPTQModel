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
