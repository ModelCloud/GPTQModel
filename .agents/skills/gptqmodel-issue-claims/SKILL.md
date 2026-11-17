---
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
name: gptqmodel-issue-claims
description: Scan open GPT-QModel GitHub issues and claim work before starting it, so that no two agents (even on one shared GitHub account) work the same issue. Claims carry an opaque per-session agent id, are bounded by the sub-agents and test hardware available now, and group correlated issues into one workload. Use whenever an agent picks up, starts, continues, hands off, or finishes work on a repository issue, or is asked to find something to work on.
---

# Claiming GPT-QModel issues

## At a glance

- **What:** a public, race-safe claim comment on each issue an agent session is working on right now.
- **Why:** many agents share one GitHub login across hosts; without per-session claims, two of them burn GPU time on the same fix.
- **When:** immediately before reproducing, editing, or spending GPU time on an issue; heartbeat while working, verify before publishing, release when done or stopping.
- **Where:** `.agents/skills/gptqmodel-issue-claims/scripts/issue_claim.py` (needs only Python and an authenticated `gh`).
- **How:** `scan`, then `claim --capacity N --note ...`, then `heartbeat`, `verify`, and `release --pr`. Act on the exit codes in the [workflow](#workflow).

Many agents run in parallel, often on different hosts under the **same GitHub
account**. The account login alone cannot tell them apart, and two agents
fixing the same issue waste a full run of GPU and model compute. Before any
work on an issue, an agent must hold a live claim under its own session's agent id.

A claim says "I am working on this now". It is never a reservation: an issue
claimed but not worked is invisible to agents that could have fixed it.

Use the helper; do not hand-write claim comments:

```bash
CLAIM=.agents/skills/gptqmodel-issue-claims/scripts/issue_claim.py
python3 $CLAIM whoami                 # print this agent's claim identity
python3 $CLAIM scan --available       # open issues this agent may claim
python3 $CLAIM held                   # this session's live workloads
python3 $CLAIM claim 1234 --note "Reproducing the per-linear GSQ crash on CPU"
python3 $CLAIM claim 1234 1235 --related "Both are the per-linear GSQ path" \
    --note "Reproducing both on one GSQ fixture"
python3 $CLAIM verify 1234 1235       # exit 0 only while this agent holds every claim
python3 $CLAIM heartbeat 1234 1235
python3 $CLAIM release 1234 1235 --pr https://github.com/ModelCloud/GPTQModel/pull/NNNN
```

It needs only Python and an authenticated `gh`. `--repo OWNER/NAME` targets
another repository.

## Agent identity

A claim is owned by an opaque agent id such as `agent-1a2b3c4d5e6f`: the first
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

## Capacity: claim only what you can run now

A **workload** is one unit of concurrent work: a single issue, or several
correlated issues handled by one change. Before claiming, decide your capacity:

```text
capacity = min(workers you will run concurrently right now,
               test-hardware slots free for those workloads right now)
```

- **Workers** are yourself plus the sub-agents you will actually spawn now, one
  workload each. An agent that cannot spawn sub-agents has one worker. Do not
  count sub-agents you might start later.
- **Test-hardware slots** are the devices the work must be validated on and
  that are free now. A GPU-kernel or quantization fix needs a free GPU (or the
  specific architecture, such as Ampere, Hopper, or ROCm) per workload, checked
  with the GPU-testing or allocator workflow. CPU-only and documentation work
  needs no GPU slot. If an issue needs hardware you do not have, do not claim
  it: you cannot finish it.

Pass the result as `claim --capacity N` (default `1`). The helper counts the
live workloads this session already holds, verified against the issues
themselves, and refuses with `AT CAPACITY` once that count reaches `N`. A
higher `--capacity` is a statement that the workers and hardware exist now;
never raise it to get past the check.

When you delegate, the coordinating agent claims each workload immediately
before handing it to the sub-agent that starts it, and passes the issue numbers
to that sub-agent. The coordinator heartbeats and releases. Sub-agents in the
same session share its agent id and must not claim separately.

## Correlated issues: one workload

Claim several issues together only when they are actually correlated: the same
root cause, the same code path fixed by one change, or one is a direct
consequence of another, so that splitting them between agents would duplicate
or conflict. List the issues in one `claim` call and state the correlation
publicly with `--related`:

```bash
python3 $CLAIM claim 1234 1235 --related "Both are the per-linear GSQ path" \
    --note "Reproducing both on one GSQ fixture"
```

- The workload uses one capacity slot.
- The claim is all-or-nothing. If any issue is blocked, nothing is posted; if
  another agent wins any issue in the race, every new claim in the workload is
  withdrawn (`LOST RACE`).
- If you find a correlated issue mid-work, claim it together with one issue you
  already hold (`claim 1234 1236 --related "..."`). It joins the running
  workload without using more capacity.
- Similar titles, the same label, or the same reporter are not correlation.
  Unrelated issues are separate workloads, subject to capacity.

## Workflow

1. **Scan before choosing.** Run `scan --available` (or `scan` to see the full
   status). Skip issues that are `CLAIMED`, `ASSIGNED` to someone, or `HAS OPEN
   PR`. When the user names an issue, still run `claim` on it; never skip the
   check because the issue was given to you.
2. **Claim immediately before work.** Run `claim <n> --note "<what you are
   starting now>"` at the moment you begin reproducing, editing, or running GPU
   jobs for the issue, never earlier. `--note` is required. Then start the work
   in the same turn. Act on the exit code:
   - `0` `CLAIMED` / `ALREADY HELD`: you own it; start now.
   - `2` `NOT CLAIMED`: someone holds it, it is assigned, closed, or already has
     an open PR. Pick another issue. Use `--allow-assigned` or
     `--allow-open-pr` only when the user explicitly directs work despite it.
   - `3` `LOST RACE`: another agent claimed it at the same moment and won. Your
     claim comment is already withdrawn. Pick another issue; do not retry.
   - `5` `AT CAPACITY`: you already run as many workloads as `--capacity`
     allows. Finish or release one first. Do not claim to queue.
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
   `release <n>... --pr <url>` for every issue in the workload that the PR
   resolves. If you stop without a fix (blocked, out of scope, handed back to
   the user, waiting on hardware or review), release immediately with
   `release <n> --note "<why, what you learned>"` so that the next agent starts
   from your findings. Release before switching to other work, too.

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

- Never hold more workloads than you are running right now. A claim you are
  not actively working on is a bug: release it.
- Do not claim to queue, reserve, or "save" issues for later, for a sub-agent
  you have not spawned, or for hardware that is not free yet.
- Claim several issues at once only when they are correlated (see above).
- Do not edit or delete another identity's claim comment. Stale claims are
  overridden through `supersedes`, never by modification.
- Do not post free-form "I'll take this" comments. Without the marker, other
  agents cannot see them.
- Never use `--allow-assigned`/`--allow-open-pr` on your own judgment.
