---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-issue-claims
description: Scan open GPT-QModel GitHub issues and claim one before starting work, using an opaque per-session agent id (a hash of host and session id, never published raw) and a post-then-verify race check so that no two agents, even on the same GitHub account, work the same issue. Use whenever an agent picks up, starts, continues, hands off, or finishes work on a repository issue, or is asked to find something to work on.
---

# Claiming GPT-QModel issues

Many agents run in parallel, often on different hosts under the **same GitHub
account**. The account login alone cannot tell them apart, and two agents
fixing the same issue waste a full run of GPU and model compute. Before any
work on an issue, an agent must hold a live claim under its own session's agent id.

Use the helper; do not hand-write claim comments:

```bash
CLAIM=.agents/skills/gptqmodel-issue-claims/scripts/issue_claim.py
python3 $CLAIM whoami                 # print this agent's claim identity
python3 $CLAIM scan --available       # open issues this agent may claim
python3 $CLAIM claim 4690 --note "Fix per-linear GSQ inference-tensor crash"
python3 $CLAIM verify 4690            # exit 0 only while this agent holds it
python3 $CLAIM heartbeat 4690
python3 $CLAIM release 4690 --pr https://github.com/ModelCloud/GPTQModel/pull/NNNN
```

It needs only Python and an authenticated `gh`. `--repo OWNER/NAME` targets
another repository.

## Agent identity

A claim is owned by an opaque agent id such as `agent-e9a2bf080c86`: the first
12 hex digits of a SHA-256 over the host name and the agent session id. Agents
on different hosts, or in different sessions on one host, get different ids,
while the host name and session id themselves never appear on GitHub.
Identification is best effort; it uses whatever the harness exposes.

- **session** is the first non-empty of `AGENT_SESSION_ID`,
  `CLAUDE_CODE_SESSION_ID`, `CODEX_THREAD_ID`, `CODEX_SESSION_ID`, and
  `OMNIGENT_RUNNER_PRIMARY_SESSION_ID`. A harness that sets none of them gets
  a random id, generated once and stored in the worktree's git directory
  (`agent-claim-session-id`), so the same worktree keeps the same identity
  across shells. Any other harness can export its own id as
  `AGENT_SESSION_ID`.
- **host** is `socket.gethostname()`; `AGENT_HOST` overrides it.
- `whoami` prints the public id and, locally only, the host and session it was
  derived from. Two agents must never share a worktree without distinct
  session ids, because they would share the generated id.

Do not put host names, session ids, local paths, or other private details in
claim notes or any other public comment.

Never treat a claim as yours because it is posted by your GitHub login. Only an
exact agent-id match is yours. A GitHub assignee is not a claim either.

## Workflow

1. **Scan before choosing.** Run `scan --available` (or `scan` to see the full
   status). Skip issues that are `CLAIMED`, `ASSIGNED` to someone, or `HAS OPEN
   PR`. When the user names an issue, still run `claim` on it; never skip the
   check because the issue was given to you.
2. **Claim before work.** Run `claim <n>` before reproducing, editing, or
   running GPU jobs for the issue. A short `--note` on the intended approach
   helps humans and other agents. Act on the exit code:
   - `0` `CLAIMED` / `ALREADY HELD`: you own it; proceed.
   - `2` `NOT CLAIMED`: someone holds it, it is assigned, closed, or already has
     an open PR. Pick another issue. Use `--allow-assigned` or
     `--allow-open-pr` only when the user explicitly directs work despite it.
   - `3` `LOST RACE`: another agent claimed it at the same moment and won. Your
     claim comment is already withdrawn. Pick another issue; do not retry.
   - `1` error (network, auth): do not start work. Retry the claim.
3. **Heartbeat while working.** Run `heartbeat <n>` at least every
   `ttl_hours / 2` (2 h by default) and at milestones (repro confirmed, fix
   drafted, tests running). For long GPU runs, heartbeat before launching and
   when they finish. A claim with no heartbeat for `ttl_hours` (default 4 h, max
   24 h via `claim --ttl-hours`) is stale, and another agent may take it over.
4. **Verify before publishing.** Run `verify <n>` immediately before pushing a
   branch, opening a PR, or posting an analysis comment. If it reports `NOT
   HELD`, stop: your claim expired or lost a race. Report your partial work to
   the user instead of publishing a competing fix. Do not run `claim` again
   unless the issue is free.
5. **Release when done or abandoned.** After opening the PR, run
   `release <n> --pr <url>`. If you stop without a fix (blocked, out of scope,
   handed back to the user), run `release <n> --note "<why, what you learned>"`
   so that the next agent starts from your findings.

## How the race is resolved

GitHub comments have no compare-and-swap, so `claim` posts first and decides
afterwards:

1. Refuse if a live claim from another identity exists, or the issue is closed,
   assigned, or already has an open PR.
2. Post a claim comment. It contains a hidden marker,
   `<!-- gptqmodel-agent-claim {json} -->`, with the hashed agent id, a random
   nonce, `state`, `claimed_at`, `ttl_hours`, and `supersedes`.
3. Wait `--settle-seconds` (20 s) so that concurrent claims become visible,
   then re-read **every** comment through paginated REST.
4. Live claims are those with `state: active`, an `updated_at` within the TTL
   (GitHub's clock, not the agent's), and no later claim listing them in
   `supersedes`. The live claim with the **lowest comment id** wins. Every agent
   reads the same comments and applies the same rule, so they agree on one
   winner.
5. Losers delete their own claim comment. If deletion fails, they mark it
   `released`.

A takeover of a stale claim lists the stale comment ids in `supersedes`. Those
claims stay dead even if their original owner edits them later, so a resumed
agent cannot overrule the agent that took over. A resumed agent gets `NOT HELD`
from `heartbeat`/`verify` and must stop.

The settle window covers normal API propagation. `verify` before publishing is
the backstop if two agents both believed they won.

## Rules

- One live claim per agent session at a time unless the user directs parallel
  issues. Release one before claiming the next.
- Do not edit or delete another identity's claim comment. Stale claims are
  overridden through `supersedes`, never by modification.
- Do not post free-form "I'll take this" comments. Without the marker, other
  agents cannot see them.
- Claim only work you will start now. Do not reserve issues for later.
- Never use `--allow-assigned`/`--allow-open-pr` on your own judgment.
