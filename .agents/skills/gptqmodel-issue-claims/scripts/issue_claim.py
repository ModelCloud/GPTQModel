#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Claim GitHub issues for an agent session so concurrent agents never duplicate work.

Many agents share one GitHub account, so the account login cannot tell claims
apart. Every claim therefore carries an opaque agent id, ``agent-<12 hex>``: a
one-way hash of the host name and the agent session id. It distinguishes agents
without publishing either value. Identification is best effort; the helper uses
whatever session id the harness exposes. GitHub has no compare-and-swap for comments,
so claiming is post-then-verify: post a claim comment, wait for the API to
settle, re-read all comments, and keep the claim only when it is the oldest
live claim (lowest comment id). Every agent applies the same rule to the same
comments, so exactly one agent wins and the others withdraw.

A claim is a commitment to start now, not a reservation. ``claim`` therefore
requires a note on the work being started and refuses once the session holds
``--capacity`` workloads (default 1): set it to the number of workloads the
agent will run concurrently right now, bounded by both the sub-agents it will
spawn and the free GPUs/hardware it has to test on. Correlated issues that one
change resolves are claimed together, all-or-nothing, as a single workload.

Only the Python standard library and an authenticated ``gh`` CLI are required.

    issue_claim.py whoami
    issue_claim.py scan [--limit 30]
    issue_claim.py held                  # this session's live workloads
    issue_claim.py claim <issue> --note TEXT [--capacity N]
    issue_claim.py claim <issue> <issue>... --related TEXT --note TEXT
    issue_claim.py verify <issue>...     # exit 0 only while this agent holds every claim
    issue_claim.py heartbeat <issue>... [--note TEXT]
    issue_claim.py release <issue>... [--pr URL] [--note TEXT]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


DEFAULT_REPO = "ModelCloud/GPTQModel"
MARKER = "gptqmodel-agent-claim"
MARKER_RE = re.compile(r"<!--\s*" + MARKER + r"\s+(\{.*?\})\s*-->", re.DOTALL)
DEFAULT_TTL_HOURS = 4.0
MAX_TTL_HOURS = 24.0
DEFAULT_SETTLE_SECONDS = 20.0

# Session identifiers exported by agent harnesses, checked in order. AGENT_SESSION_ID is the explicit override.
SESSION_ENV_VARS = (
    "AGENT_SESSION_ID",
    "CLAUDE_CODE_SESSION_ID",
    "CODEX_THREAD_ID",
    "CODEX_SESSION_ID",
    "OMNIGENT_RUNNER_PRIMARY_SESSION_ID",
)


class ClaimError(RuntimeError):
    pass


# --------------------------------------------------------------------------- gh helpers


def gh(*args: str, input_text: str | None = None) -> str:
    proc = subprocess.run(["gh", *args], input=input_text, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ClaimError(f"gh {' '.join(args[:3])} failed: {proc.stderr.strip()}")
    return proc.stdout


def gh_json(*args: str, input_obj: dict | None = None):
    if input_obj is not None:
        out = gh(*args, "--input", "-", input_text=json.dumps(input_obj))
    else:
        out = gh(*args)
    return json.loads(out) if out.strip() else None


def graphql(query: str, **variables) -> dict:
    data = gh_json("api", "graphql", input_obj={"query": query, "variables": variables})
    if data.get("errors"):
        raise ClaimError(f"GraphQL error: {data['errors']}")
    return data["data"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fmt_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- identity


def _git_dir() -> Path | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--absolute-git-dir"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(out.stdout.strip())


def session_id() -> tuple[str, str]:
    """Return (session_id, source). Falls back to a UUID persisted per git worktree."""
    for name in SESSION_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    base = _git_dir() or Path.home() / ".cache" / "gptqmodel"
    path = base / "agent-claim-session-id"
    try:
        value = path.read_text().strip()
        if value:
            return value, f"generated ({path})"
    except OSError:
        pass
    value = f"gen-{uuid.uuid4()}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")
    except OSError:
        print(f"warning: could not persist generated session id to {path}; set AGENT_SESSION_ID", file=sys.stderr)
    return value, f"generated ({path})"


def host_name() -> str:
    return os.environ.get("AGENT_HOST", "").strip() or socket.gethostname()


def agent_id() -> str:
    """Public claim id. Hashed so that host names and session ids never appear on GitHub."""
    session, _ = session_id()
    digest = hashlib.sha256(f"gptqmodel-agent-claim\0{host_name()}\0{session}".encode()).hexdigest()
    return f"agent-{digest[:12]}"


# --------------------------------------------------------------------------- claim model


@dataclass
class Claim:
    comment_id: int
    author: str
    created_at: datetime
    updated_at: datetime
    meta: dict
    superseded: bool = False

    @property
    def agent(self) -> str:
        return str(self.meta.get("agent", ""))

    @property
    def state(self) -> str:
        return str(self.meta.get("state", ""))

    @property
    def ttl(self) -> timedelta:
        try:
            hours = float(self.meta.get("ttl_hours", DEFAULT_TTL_HOURS))
        except (TypeError, ValueError):
            hours = DEFAULT_TTL_HOURS
        return timedelta(hours=min(max(hours, 0.0), MAX_TTL_HOURS))

    @property
    def expires_at(self) -> datetime:
        # Server-side updated_at is the heartbeat; it is immune to client clock skew.
        return self.updated_at + self.ttl

    def live(self, now: datetime) -> bool:
        return self.state == "active" and not self.superseded and now < self.expires_at


@dataclass
class IssueState:
    number: int
    title: str
    state: str
    assignees: list[str]
    claims: list[Claim]
    open_prs: list[str] = field(default_factory=list)

    def live_claims(self, now: datetime) -> list[Claim]:
        return sorted((c for c in self.claims if c.live(now)), key=lambda c: c.comment_id)

    def holder(self, now: datetime) -> Claim | None:
        live = self.live_claims(now)
        return live[0] if live else None


def parse_claims(comments: list[dict]) -> list[Claim]:
    claims = []
    for c in comments:
        match = MARKER_RE.search(c.get("body") or "")
        if not match:
            continue
        try:
            meta = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        claims.append(
            Claim(
                comment_id=int(c["id"]),
                author=c.get("author") or "",
                created_at=parse_ts(c["created_at"]),
                updated_at=parse_ts(c["updated_at"]),
                meta=meta,
            )
        )
    # A takeover claim names the stale claims it replaces; they stay dead even if their owner heartbeats again.
    superseded = {int(x) for c in claims for x in c.meta.get("supersedes", []) if str(x).isdigit()}
    for c in claims:
        c.superseded = c.comment_id in superseded
    return claims


ISSUE_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      number title state
      assignees(first: 20) { nodes { login } }
      timelineItems(itemTypes: [CROSS_REFERENCED_EVENT, CONNECTED_EVENT], last: 50) {
        nodes {
          ... on CrossReferencedEvent { source { ... on PullRequest { url state } } }
          ... on ConnectedEvent { subject { ... on PullRequest { url state } } }
        }
      }
    }
  }
}
"""


def _open_prs(timeline_nodes: list[dict]) -> list[str]:
    urls = []
    for node in timeline_nodes:
        pr = node.get("source") or node.get("subject") or {}
        if pr.get("state") == "OPEN" and pr.get("url") not in urls:
            urls.append(pr["url"])
    return urls


def fetch_issue(repo: str, number: int) -> IssueState:
    owner, name = repo.split("/", 1)
    issue = graphql(ISSUE_QUERY, owner=owner, name=name, number=number)["repository"]["issue"]
    if issue is None:
        raise ClaimError(f"{repo}#{number} is not an issue")
    # REST pagination returns every comment, so no claim is missed on long threads.
    lines = gh(
        "api",
        "--paginate",
        f"repos/{repo}/issues/{number}/comments?per_page=100",
        "--jq",
        ".[] | {id, created_at, updated_at, body, author: .user.login} | tojson",
    )
    comments = [json.loads(line) for line in lines.splitlines() if line.strip()]
    return IssueState(
        number=issue["number"],
        title=issue["title"],
        state=issue["state"],
        assignees=[a["login"] for a in issue["assignees"]["nodes"]],
        claims=parse_claims(comments),
        open_prs=_open_prs(issue["timelineItems"]["nodes"]),
    )


SCAN_QUERY = """
query($owner: String!, $name: String!, $first: Int!) {
  repository(owner: $owner, name: $name) {
    issues(first: $first, states: OPEN, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number title state
        labels(first: 10) { nodes { name } }
        assignees(first: 20) { nodes { login } }
        comments(last: 100) { totalCount nodes { fullDatabaseId createdAt updatedAt body author { login } } }
        timelineItems(itemTypes: [CROSS_REFERENCED_EVENT, CONNECTED_EVENT], last: 50) {
          nodes {
            ... on CrossReferencedEvent { source { ... on PullRequest { url state } } }
            ... on ConnectedEvent { subject { ... on PullRequest { url state } } }
          }
        }
      }
    }
  }
}
"""


# --------------------------------------------------------------------------- comment bodies


def render_body(meta: dict) -> str:
    agent = meta["agent"]
    state = meta["state"]
    if state == "active":
        headline = f"**Agent claim:** `{agent}` is working on this issue now."
        detail = (
            f"Started {meta['claimed_at']}. The claim expires {meta['ttl_hours']}h after this comment was last "
            "edited (heartbeat). Other agents: do not start this issue while the claim is live."
        )
    elif state == "done":
        headline = f"**Agent claim released:** `{agent}` finished this issue."
        detail = f"Pull request: {meta['pr']}" if meta.get("pr") else "Work complete."
    else:
        headline = f"**Agent claim released:** `{agent}` is no longer working on this issue."
        detail = "The issue is free to claim."
    lines = [headline, "", detail]
    issues = meta.get("workload_issues") or []
    if len(issues) > 1:
        lines += ["", "One workload with " + ", ".join(f"#{n}" for n in issues)
                  + (f": {meta['related']}" if meta.get("related") else "")]
    if meta.get("note"):
        lines += ["", str(meta["note"])]
    lines += ["", f"<!-- {MARKER} {json.dumps(meta, sort_keys=True)} -->"]
    return "\n".join(lines)


def edit_comment(repo: str, comment_id: int, body: str) -> dict:
    return gh_json("api", "-X", "PATCH", f"repos/{repo}/issues/comments/{comment_id}", input_obj={"body": body})


def delete_comment(repo: str, comment_id: int) -> None:
    gh("api", "-X", "DELETE", f"repos/{repo}/issues/comments/{comment_id}")


# --------------------------------------------------------------------------- commands


def describe(claim: Claim, now: datetime) -> str:
    left = claim.expires_at - now
    mins = int(left.total_seconds() // 60)
    return f"{claim.agent} (comment {claim.comment_id}, expires in {mins}m)"


def cmd_whoami(args) -> int:
    session, source = session_id()
    # host and session are printed locally only; claims publish just the hashed agent id.
    print(json.dumps({"agent": agent_id(), "local_host": host_name(), "local_session": session,
                      "session_source": source}, indent=2))
    return 0


def cmd_scan(args) -> int:
    owner, name = args.repo.split("/", 1)
    me = agent_id()
    now = utcnow()
    nodes = graphql(SCAN_QUERY, owner=owner, name=name, first=args.limit)["repository"]["issues"]["nodes"]
    rows = []
    for n in nodes:
        comments = [
            {"id": c["fullDatabaseId"], "created_at": c["createdAt"], "updated_at": c["updatedAt"], "body": c["body"],
             "author": (c.get("author") or {}).get("login", "")}
            for c in n["comments"]["nodes"]
        ]
        issue = IssueState(
            number=n["number"], title=n["title"], state=n["state"],
            assignees=[a["login"] for a in n["assignees"]["nodes"]],
            claims=parse_claims(comments), open_prs=_open_prs(n["timelineItems"]["nodes"]),
        )
        holder = issue.holder(now)
        if holder and holder.agent == me:
            status = "MINE"
        elif holder:
            status = f"CLAIMED by {describe(holder, now)}"
        elif issue.open_prs:
            status = f"HAS OPEN PR {', '.join(issue.open_prs)}"
        elif issue.assignees:
            status = f"ASSIGNED to {', '.join(issue.assignees)}"
        else:
            status = "AVAILABLE"
        if n["comments"]["totalCount"] > 100 and status == "AVAILABLE":
            status += " (long thread: confirm with `claim`, which reads every comment)"
        labels = ",".join(lbl["name"] for lbl in n["labels"]["nodes"])
        rows.append({"number": issue.number, "status": status, "labels": labels, "title": issue.title})
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for r in rows:
            if args.available and not r["status"].startswith(("AVAILABLE", "MINE")):
                continue
            print(f"#{r['number']:<6} {r['status']:<40} [{r['labels']}] {r['title']}")
    return 0


def _blockers(issue: IssueState, me: str, now: datetime, args) -> list[str]:
    reasons = []
    if issue.state != "OPEN":
        reasons.append(f"issue is {issue.state}")
    holder = issue.holder(now)
    if holder and holder.agent != me:
        reasons.append(f"live claim held by {describe(holder, now)}")
    if issue.open_prs and not args.allow_open_pr:
        reasons.append(f"open pull request(s) already reference it: {', '.join(issue.open_prs)}")
    if issue.assignees and not args.allow_assigned:
        reasons.append(f"assigned to {', '.join(issue.assignees)}")
    return reasons


def workload_of(claim: Claim) -> str:
    # Claims posted before workloads existed are single-issue workloads.
    return str(claim.meta.get("workload") or f"c{claim.comment_id}")


def _ledger_path(repo: str, me: str) -> Path:
    return Path.home() / ".cache" / "gptqmodel" / "issue-claims" / f"{repo.replace('/', '__')}__{me}.json"


def _ledger_load(repo: str, me: str) -> set[int]:
    try:
        return {int(n) for n in json.loads(_ledger_path(repo, me).read_text()).get("issues", [])}
    except (OSError, ValueError, TypeError, AttributeError):
        return set()


def _ledger_save(repo: str, me: str, issues: set[int]) -> None:
    path = _ledger_path(repo, me)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"issues": sorted(issues)}) + "\n")
    except OSError as exc:
        print(f"warning: could not write claim ledger {path}: {exc}", file=sys.stderr)


def search_claimed_issues(repo: str, me: str) -> set[int]:
    """Best-effort discovery of this agent's claims made from another worktree or after a lost ledger."""
    try:
        out = gh("search", "issues", me, "--repo", repo, "--state", "open", "--match", "comments",
                 "--limit", "100", "--json", "number", "--jq", ".[].number")
    except ClaimError:
        return set()
    return {int(n) for n in out.split()}


def held_workloads(repo: str, me: str) -> dict[str, list[int]]:
    """Live workloads of this session, verified against the issues themselves."""
    candidates = _ledger_load(repo, me) | search_claimed_issues(repo, me)
    now = utcnow()
    held: dict[str, list[int]] = {}
    for number in sorted(candidates):
        holder = fetch_issue(repo, number).holder(now)
        if holder and holder.agent == me:
            held.setdefault(workload_of(holder), []).append(number)
    _ledger_save(repo, me, {n for issues in held.values() for n in issues})
    return held


def cmd_held(args) -> int:
    held = held_workloads(args.repo, agent_id())
    for workload, issues in held.items():
        print(f"{workload}: " + ", ".join(f"#{n}" for n in issues))
    print(f"{len(held)} live workload(s)")
    return 0


def cmd_claim(args) -> int:
    me = agent_id()
    numbers = sorted(set(args.issue))
    if len(numbers) > 1 and not (args.related or "").strip():
        print("NOT CLAIMED: claiming several issues requires --related naming the shared root cause or change. "
              "Claim unrelated issues as separate workloads only when capacity allows.")
        return 2
    now = utcnow()
    issues = {n: fetch_issue(args.repo, n) for n in numbers}
    holders = {n: issue.holder(now) for n, issue in issues.items()}
    mine = {n: h for n, h in holders.items() if h and h.agent == me}

    if len(mine) == len(numbers):
        for n, holder in mine.items():
            # The body must change for GitHub to bump updated_at, which is the heartbeat.
            holder.meta["heartbeat_at"] = fmt_ts(now)
            edit_comment(args.repo, holder.comment_id, render_body(holder.meta))
        print("ALREADY HELD: " + ", ".join(f"#{n}" for n in numbers) + f" by {me}; heartbeat refreshed")
        return 0

    blockers = [f"#{n}: {r}" for n in numbers if n not in mine for r in _blockers(issues[n], me, now, args)]
    if blockers:
        print("NOT CLAIMED: " + "; ".join(blockers))
        return 2

    joined = {workload_of(h) for h in mine.values()}
    if len(joined) > 1:
        print("NOT CLAIMED: the issues already belong to different workloads of this session; release one first.")
        return 2
    if joined:
        # A correlated issue found mid-work joins the running workload and uses no extra capacity.
        workload = joined.pop()
    else:
        held = held_workloads(args.repo, me)
        if len(held) >= args.capacity:
            print(f"AT CAPACITY: this session already runs {len(held)} workload(s) "
                  f"({', '.join(held)}) with --capacity {args.capacity}. Finish or release one first; "
                  "never claim issues to queue them.")
            return 5
        workload = uuid.uuid4().hex[:12]

    all_issues = sorted(set(numbers) | {n for n, h in mine.items()})
    posted: dict[int, tuple[int, dict]] = {}
    try:
        for n in numbers:
            if n in mine:
                continue
            stale = [c.comment_id for c in issues[n].claims
                     if c.state == "active" and not c.superseded and not c.live(now)]
            meta = {
                "v": 1,
                "agent": me,
                "nonce": uuid.uuid4().hex[:12],
                "state": "active",
                "claimed_at": fmt_ts(now),
                "ttl_hours": args.ttl_hours,
                "supersedes": stale,
                "workload": workload,
                "workload_issues": all_issues,
                "note": args.note,
            }
            if args.related:
                meta["related"] = args.related
            reply = gh_json("api", "-X", "POST", f"repos/{args.repo}/issues/{n}/comments",
                            input_obj={"body": render_body(meta)})
            posted[n] = (int(reply["id"]), meta)

        # Let concurrent claimers' comments become visible before deciding the winners.
        time.sleep(args.settle_seconds)
        now = utcnow()
        lost = {}
        for n, (my_id, _) in posted.items():
            holder = fetch_issue(args.repo, n).holder(now)
            if holder is None or holder.comment_id != my_id:
                lost[n] = describe(holder, now) if holder else "no live claim (own comment not visible)"
    except ClaimError:
        _withdraw(args.repo, posted)
        raise

    if lost:
        # All-or-nothing: a partial win would split one workload between agents.
        _withdraw(args.repo, posted)
        print("LOST RACE: " + "; ".join(f"#{n} won by {w}" for n, w in lost.items())
              + ". All new claims in this workload withdrawn; pick other work.")
        return 3

    for holder in mine.values():
        holder.meta["workload_issues"] = all_issues
        if args.related:
            holder.meta["related"] = args.related
        holder.meta["heartbeat_at"] = fmt_ts(now)
        edit_comment(args.repo, holder.comment_id, render_body(holder.meta))
    _ledger_save(args.repo, me, _ledger_load(args.repo, me) | set(all_issues))
    print("CLAIMED: " + ", ".join(f"#{n}" for n in all_issues) + f" as workload {workload} by {me}. "
          f"Start now; heartbeat at least every {args.ttl_hours / 2:g}h and run `verify` before pushing.")
    return 0


def _withdraw(repo: str, posted: dict[int, tuple[int, dict]]) -> None:
    for n, (comment_id, meta) in posted.items():
        try:
            delete_comment(repo, comment_id)
        except ClaimError:
            meta["state"] = "released"
            meta["note"] = "Withdrawn: another agent claimed part of this workload first."
            edit_comment(repo, comment_id, render_body(meta))


def _own_live_claim(repo: str, number: int) -> tuple[Claim | None, IssueState]:
    me = agent_id()
    issue = fetch_issue(repo, number)
    holder = issue.holder(utcnow())
    return (holder if holder and holder.agent == me else None), issue


def _not_held(n: int, issue: IssueState, hint: str) -> str:
    holder = issue.holder(utcnow())
    return f"NOT HELD: #{n}; holder is {describe(holder, utcnow()) if holder else 'nobody'}. {hint}"


def cmd_verify(args) -> int:
    rc = 0
    for n in args.issue:
        mine, issue = _own_live_claim(args.repo, n)
        if mine:
            print(f"HELD: #{n} by {mine.agent}, expires {fmt_ts(mine.expires_at)}")
        else:
            print(_not_held(n, issue, "Stop work."))
            rc = 4
    return rc


def cmd_heartbeat(args) -> int:
    rc = 0
    for n in args.issue:
        mine, issue = _own_live_claim(args.repo, n)
        if not mine:
            print(_not_held(n, issue, "Do not revive an expired claim; run `claim` again instead."))
            rc = 4
            continue
        mine.meta["heartbeat_at"] = fmt_ts(utcnow())
        if args.note:
            mine.meta["note"] = args.note
        edit_comment(args.repo, mine.comment_id, render_body(mine.meta))
        print(f"HEARTBEAT: #{n} comment {mine.comment_id}")
    return rc


def cmd_release(args) -> int:
    me = agent_id()
    released = set()
    for n in args.issue:
        issue = fetch_issue(args.repo, n)
        mine = [c for c in issue.claims if c.agent == me and c.state == "active"]
        if not mine:
            print(f"NOTHING TO RELEASE: no active claim by {me} on #{n}")
        for c in mine:
            c.meta["state"] = "done" if args.pr else "released"
            c.meta["released_at"] = fmt_ts(utcnow())
            if args.pr:
                c.meta["pr"] = args.pr
            if args.note:
                c.meta["note"] = args.note
            edit_comment(args.repo, c.comment_id, render_body(c.meta))
            print(f"RELEASED: #{n} comment {c.comment_id} ({c.meta['state']})")
        released.add(n)
    _ledger_save(args.repo, me, _ledger_load(args.repo, me) - released)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami").set_defaults(func=cmd_whoami)

    p = sub.add_parser("scan")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--available", action="store_true", help="only show issues this agent may claim")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_scan)

    sub.add_parser("held", help="list this session's live workloads").set_defaults(func=cmd_held)

    p = sub.add_parser("claim")
    p.add_argument("issue", type=int, nargs="+", help="one issue, or several correlated issues as one workload")
    p.add_argument("--note", required=True, help="the work being started now, shown in the claim comment")
    p.add_argument("--related", help="required for several issues: the shared root cause or change")
    p.add_argument("--capacity", type=int, default=1,
                   help="workloads this session can run right now: min(sub-agents it will spawn, free test hardware)")
    p.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS)
    p.add_argument("--allow-assigned", action="store_true", help="the user explicitly directed work on an assigned issue")
    p.add_argument("--allow-open-pr", action="store_true", help="the user explicitly directed work despite an open PR")
    p.set_defaults(func=cmd_claim)

    for name, func in (("verify", cmd_verify), ("heartbeat", cmd_heartbeat), ("release", cmd_release)):
        p = sub.add_parser(name)
        p.add_argument("issue", type=int, nargs="+")
        if name in ("heartbeat", "release"):
            p.add_argument("--note")
        if name == "release":
            p.add_argument("--pr", help="URL of the pull request that resolves the issue")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    if getattr(args, "capacity", 1) < 1:
        parser.error("--capacity must be at least 1")
    if getattr(args, "ttl_hours", DEFAULT_TTL_HOURS) > MAX_TTL_HOURS:
        parser.error(f"--ttl-hours may not exceed {MAX_TTL_HOURS:g}")
    try:
        return args.func(args)
    except ClaimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
