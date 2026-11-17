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

Only the Python standard library and an authenticated ``gh`` CLI are required.

    issue_claim.py whoami
    issue_claim.py scan [--limit 30]
    issue_claim.py claim <issue>
    issue_claim.py verify <issue>        # exit 0 only while this agent holds the claim
    issue_claim.py heartbeat <issue> [--note TEXT]
    issue_claim.py release <issue> [--pr URL] [--note TEXT]
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


def render_body(meta: dict, note: str | None = None) -> str:
    agent = meta["agent"]
    state = meta["state"]
    if state == "active":
        headline = f"**Agent claim:** `{agent}` is working on this issue."
        detail = (
            f"Claimed {meta['claimed_at']}. The claim expires {meta['ttl_hours']}h after this comment was last "
            "edited (heartbeat). Other agents: do not start this issue while the claim is live."
        )
    elif state == "done":
        headline = f"**Agent claim released:** `{agent}` finished this issue."
        detail = f"Pull request: {meta['pr']}" if meta.get("pr") else "Work complete."
    else:
        headline = f"**Agent claim released:** `{agent}` is no longer working on this issue."
        detail = "The issue is free to claim."
    lines = [headline, "", detail]
    if note:
        lines += ["", note]
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


def cmd_claim(args) -> int:
    me = agent_id()
    now = utcnow()
    issue = fetch_issue(args.repo, args.issue)
    holder = issue.holder(now)
    if holder and holder.agent == me:
        # The body must change for GitHub to bump updated_at, which is the heartbeat.
        holder.meta["heartbeat_at"] = fmt_ts(now)
        edit_comment(args.repo, holder.comment_id, render_body(holder.meta))
        print(f"ALREADY HELD: #{args.issue} by {me} (comment {holder.comment_id}); heartbeat refreshed")
        return 0
    blockers = _blockers(issue, me, now, args)
    if blockers:
        print(f"NOT CLAIMED: #{args.issue}: " + "; ".join(blockers))
        return 2

    stale = [c.comment_id for c in issue.claims if c.state == "active" and not c.superseded and not c.live(now)]
    meta = {
        "v": 1,
        "agent": me,
        "nonce": uuid.uuid4().hex[:12],
        "state": "active",
        "claimed_at": fmt_ts(now),
        "ttl_hours": args.ttl_hours,
        "supersedes": stale,
    }
    posted = gh_json("api", "-X", "POST", f"repos/{args.repo}/issues/{args.issue}/comments",
                     input_obj={"body": render_body(meta, args.note)})
    my_id = int(posted["id"])

    # Let concurrent claimers' comments become visible before deciding the winner.
    time.sleep(args.settle_seconds)
    issue = fetch_issue(args.repo, args.issue)
    now = utcnow()
    holder = issue.holder(now)
    if holder is not None and holder.comment_id == my_id:
        print(f"CLAIMED: #{args.issue} by {me} (comment {my_id}). Heartbeat at least every "
              f"{args.ttl_hours / 2:g}h and run `verify` before pushing or opening a PR.")
        return 0

    winner = describe(holder, now) if holder else "no live claim (own comment not visible)"
    try:
        delete_comment(args.repo, my_id)
    except ClaimError:
        meta["state"] = "released"
        edit_comment(args.repo, my_id, render_body(meta, "Withdrawn: another agent claimed this issue first."))
    print(f"LOST RACE: #{args.issue}: winner is {winner}. Own claim withdrawn; pick another issue.")
    return 3


def _own_live_claim(repo: str, number: int) -> tuple[Claim | None, IssueState]:
    me = agent_id()
    issue = fetch_issue(repo, number)
    holder = issue.holder(utcnow())
    return (holder if holder and holder.agent == me else None), issue


def cmd_verify(args) -> int:
    mine, issue = _own_live_claim(args.repo, args.issue)
    if mine:
        print(f"HELD: #{args.issue} by {mine.agent}, expires {fmt_ts(mine.expires_at)}")
        return 0
    holder = issue.holder(utcnow())
    print(f"NOT HELD: #{args.issue}; holder is {describe(holder, utcnow()) if holder else 'nobody'}. Stop work.")
    return 4


def cmd_heartbeat(args) -> int:
    mine, issue = _own_live_claim(args.repo, args.issue)
    if not mine:
        holder = issue.holder(utcnow())
        print(f"NOT HELD: #{args.issue}; holder is {describe(holder, utcnow()) if holder else 'nobody'}. "
              "Do not revive an expired claim; run `claim` again instead.")
        return 4
    mine.meta["heartbeat_at"] = fmt_ts(utcnow())
    edit_comment(args.repo, mine.comment_id, render_body(mine.meta, args.note))
    print(f"HEARTBEAT: #{args.issue} comment {mine.comment_id}")
    return 0


def cmd_release(args) -> int:
    me = agent_id()
    issue = fetch_issue(args.repo, args.issue)
    mine = [c for c in issue.claims if c.agent == me and c.state == "active"]
    if not mine:
        print(f"NOTHING TO RELEASE: no active claim by {me} on #{args.issue}")
        return 0
    for c in mine:
        c.meta["state"] = "done" if args.pr else "released"
        c.meta["released_at"] = fmt_ts(utcnow())
        if args.pr:
            c.meta["pr"] = args.pr
        edit_comment(args.repo, c.comment_id, render_body(c.meta, args.note))
        print(f"RELEASED: #{args.issue} comment {c.comment_id} ({c.meta['state']})")
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

    p = sub.add_parser("claim")
    p.add_argument("issue", type=int)
    p.add_argument("--note", help="one line on the planned approach, shown in the claim comment")
    p.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_HOURS)
    p.add_argument("--settle-seconds", type=float, default=DEFAULT_SETTLE_SECONDS)
    p.add_argument("--allow-assigned", action="store_true", help="the user explicitly directed work on an assigned issue")
    p.add_argument("--allow-open-pr", action="store_true", help="the user explicitly directed work despite an open PR")
    p.set_defaults(func=cmd_claim)

    for name, func in (("verify", cmd_verify), ("heartbeat", cmd_heartbeat), ("release", cmd_release)):
        p = sub.add_parser(name)
        p.add_argument("issue", type=int)
        if name in ("heartbeat", "release"):
            p.add_argument("--note")
        if name == "release":
            p.add_argument("--pr", help="URL of the pull request that resolves the issue")
        p.set_defaults(func=func)

    args = parser.parse_args(argv)
    if getattr(args, "ttl_hours", DEFAULT_TTL_HOURS) > MAX_TTL_HOURS:
        parser.error(f"--ttl-hours may not exceed {MAX_TTL_HOURS:g}")
    try:
        return args.func(args)
    except ClaimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
