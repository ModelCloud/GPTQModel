# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Agent issue claims: exactly one session wins, even when every agent shares one GitHub login."""

import importlib.util
import sys
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / ".agents" / "skills" / "gptqmodel-issue-claims" / "scripts" / "issue_claim.py"
)
HOSTS_SESSIONS = [(f"host{h}", f"sess{s}") for h in (1, 2) for s in ("a", "b")]


@pytest.fixture
def ic(monkeypatch):
    spec = importlib.util.spec_from_file_location("gptqmodel_issue_claim", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)

    class FakeIssue:
        """In-memory GitHub issue thread; comment ids increase like GitHub's."""

        def __init__(self):
            self.comments = {}
            self.next_id = 1000
            self.clock = mod.utcnow()
            self.lock = threading.Lock()
            self.post_barrier = None
            self.settle_barrier = None
            self.local = threading.local()

        def post(self, *args, input_obj=None):
            if self.post_barrier is not None:
                self.post_barrier.wait(timeout=10)
            with self.lock:
                cid, self.next_id = self.next_id, self.next_id + 1
                ts = mod.fmt_ts(self.clock)
                self.comments[cid] = {"id": cid, "created_at": ts, "updated_at": ts, "body": input_obj["body"]}
            return {"id": cid}

        def edit(self, repo, cid, body):
            with self.lock:
                self.comments[cid].update(body=body, updated_at=mod.fmt_ts(self.clock))

        def fetch(self, repo, number):
            with self.lock:
                comments = [dict(c) for _, c in sorted(self.comments.items())]
            return mod.IssueState(number, "t", "OPEN", [], mod.parse_claims(comments), [])

        def settle(self, seconds):
            if self.settle_barrier is not None:
                self.settle_barrier.wait(timeout=10)

    fake = FakeIssue()
    # The real agent_id() runs, so the tests also cover hashing host + session into the public id.
    monkeypatch.setattr(mod, "host_name", lambda: fake.local.host_session[0])
    monkeypatch.setattr(mod, "session_id", lambda: (fake.local.host_session[1], "test"))
    monkeypatch.setattr(mod, "utcnow", lambda: fake.clock)
    monkeypatch.setattr(mod, "fetch_issue", fake.fetch)
    monkeypatch.setattr(mod, "gh_json", fake.post)
    monkeypatch.setattr(mod, "edit_comment", fake.edit)
    monkeypatch.setattr(mod, "delete_comment", lambda repo, cid: fake.comments.pop(cid))
    monkeypatch.setattr(mod, "time", SimpleNamespace(sleep=fake.settle))
    mod.fake = fake
    return mod


def _args(**overrides):
    values = {"repo": "R", "issue": 1, "note": None, "ttl_hours": 4.0, "settle_seconds": 0.0,
              "allow_assigned": False, "allow_open_pr": False, "pr": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _as(ic, agent, cmd, **overrides):
    ic.fake.local.host_session = agent
    return cmd(_args(**overrides))


def _public_id(ic, agent):
    ic.fake.local.host_session = agent
    return ic.agent_id()


def test_simultaneous_claims_under_one_login_have_one_winner(ic):
    fake = ic.fake
    # Every agent passes the pre-check before anyone posts, then all re-read together.
    agents = HOSTS_SESSIONS
    fake.post_barrier = threading.Barrier(len(agents))
    fake.settle_barrier = threading.Barrier(len(agents))
    codes = {}

    def run(agent):
        codes[agent] = _as(ic, agent, ic.cmd_claim)

    threads = [threading.Thread(target=run, args=(a,)) for a in agents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    winners = [a for a, rc in codes.items() if rc == 0]
    assert len(winners) == 1
    assert sorted(rc for rc in codes.values()) == [0, 3, 3, 3]
    # Losers withdraw their comment; the surviving claim names the winning session by its hashed id only.
    assert len(fake.comments) == 1
    holder = fake.fetch("R", 1).holder(fake.clock)
    assert holder.agent == _public_id(ic, winners[0])
    assert len({_public_id(ic, a) for a in agents}) == len(agents)
    body = next(iter(fake.comments.values()))["body"]
    for host, session in agents:
        assert host not in body and session not in body

    fake.post_barrier = fake.settle_barrier = None
    for other in agents:
        expected = 0 if other == winners[0] else 2
        assert _as(ic, other, ic.cmd_claim) == expected
        assert _as(ic, other, ic.cmd_verify) == (0 if expected == 0 else 4)
    assert len(fake.comments) == 1


def test_stale_claim_takeover_cannot_be_revived(ic):
    fake = ic.fake
    first, second, third = HOSTS_SESSIONS[:3]
    assert _as(ic, first, ic.cmd_claim) == 0
    first_id = min(fake.comments)

    fake.clock += timedelta(hours=4, minutes=1)
    assert _as(ic, second, ic.cmd_claim) == 0
    assert _as(ic, first, ic.cmd_heartbeat) == 4

    # Even if the stale comment is edited later, the takeover's `supersedes` keeps it dead.
    fake.edit("R", first_id, fake.comments[first_id]["body"])
    assert _as(ic, first, ic.cmd_verify) == 4
    assert _as(ic, second, ic.cmd_verify) == 0

    assert _as(ic, second, ic.cmd_release, pr="https://github.com/ModelCloud/GPTQModel/pull/1") == 0
    released = fake.fetch("R", 1).claims[-1]
    assert released.state == "done" and released.meta["pr"].endswith("/pull/1")
    assert _as(ic, third, ic.cmd_claim) == 0


def test_heartbeat_keeps_claim_alive(ic):
    fake = ic.fake
    owner, rival = HOSTS_SESSIONS[:2]
    assert _as(ic, owner, ic.cmd_claim) == 0
    for _ in range(3):
        fake.clock += timedelta(hours=3)
        assert _as(ic, owner, ic.cmd_heartbeat) == 0
    assert _as(ic, rival, ic.cmd_claim) == 2
