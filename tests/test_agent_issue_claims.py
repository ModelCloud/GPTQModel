# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Agent issue claims: one session per issue, bounded by capacity, correlated issues as one workload."""

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
def ic(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("gptqmodel_issue_claim", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)

    class FakeRepo:
        """In-memory GitHub issue threads; comment ids increase across the repository like GitHub's."""

        def __init__(self):
            self.comments = {}
            self.next_id = 1000
            self.clock = mod.utcnow()
            self.lock = threading.Lock()
            self.post_barrier = None
            self.settle_barrier = None
            self.before_post = None
            self.local = threading.local()

        def post(self, *args, input_obj=None):
            number = int(args[-1].split("/")[-2])
            if self.before_post is not None:
                self.before_post(number)
            if self.post_barrier is not None:
                self.post_barrier.wait(timeout=10)
            with self.lock:
                cid, self.next_id = self.next_id, self.next_id + 1
                ts = mod.fmt_ts(self.clock)
                self.comments[cid] = {"id": cid, "issue": number, "created_at": ts, "updated_at": ts,
                                      "body": input_obj["body"]}
            return {"id": cid}

        def edit(self, repo, cid, body):
            with self.lock:
                self.comments[cid].update(body=body, updated_at=mod.fmt_ts(self.clock))

        def thread(self, number):
            with self.lock:
                return [dict(c) for _, c in sorted(self.comments.items()) if c["issue"] == number]

        def fetch(self, repo, number):
            return mod.IssueState(number, "t", "OPEN", [], mod.parse_claims(self.thread(number)), [])

        def search(self, repo, me):
            with self.lock:
                return {c["issue"] for c in self.comments.values() if me in c["body"]}

        def settle(self, seconds):
            if self.settle_barrier is not None:
                self.settle_barrier.wait(timeout=10)

    fake = FakeRepo()
    # The real agent_id() runs, so the tests also cover hashing host + session into the public id.
    monkeypatch.setattr(mod, "host_name", lambda: fake.local.host_session[0])
    monkeypatch.setattr(mod, "session_id", lambda: (fake.local.host_session[1], "test"))
    monkeypatch.setattr(mod, "utcnow", lambda: fake.clock)
    monkeypatch.setattr(mod, "fetch_issue", fake.fetch)
    monkeypatch.setattr(mod, "search_claimed_issues", fake.search)
    monkeypatch.setattr(mod, "gh_json", fake.post)
    monkeypatch.setattr(mod, "edit_comment", fake.edit)
    monkeypatch.setattr(mod, "delete_comment", lambda repo, cid: fake.comments.pop(cid))
    monkeypatch.setattr(mod, "time", SimpleNamespace(sleep=fake.settle))
    monkeypatch.setattr(mod, "_ledger_path", lambda repo, me: tmp_path / f"{me}.json")
    mod.fake = fake
    return mod


def _args(**overrides):
    values = {"repo": "R", "issue": [1], "note": "Reproducing now", "related": None, "capacity": 1,
              "ttl_hours": 4.0, "settle_seconds": 0.0, "allow_assigned": False, "allow_open_pr": False, "pr": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def _as(ic, agent, cmd, **overrides):
    ic.fake.local.host_session = agent
    return cmd(_args(**overrides))


def _public_id(ic, agent):
    ic.fake.local.host_session = agent
    return ic.agent_id()


def _held(ic, agent):
    return ic.held_workloads("R", _public_id(ic, agent))


def test_simultaneous_claims_under_one_login_have_one_winner(ic):
    fake = ic.fake
    agents = HOSTS_SESSIONS
    # Every agent passes the pre-check before anyone posts, then all re-read together.
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
    assert sorted(codes.values()) == [0, 3, 3, 3]
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


def test_capacity_blocks_queueing_extra_issues(ic):
    fake = ic.fake
    agent = HOSTS_SESSIONS[0]
    assert _as(ic, agent, ic.cmd_claim, issue=[1]) == 0
    # Default capacity is one workload: a second unrelated issue would only be queued.
    assert _as(ic, agent, ic.cmd_claim, issue=[2]) == 5
    assert fake.thread(2) == []

    # Capacity backed by more sub-agents/hardware admits exactly that many workloads.
    assert _as(ic, agent, ic.cmd_claim, issue=[2], capacity=2) == 0
    assert _as(ic, agent, ic.cmd_claim, issue=[3], capacity=2) == 5
    assert len(_held(ic, agent)) == 2

    # Releasing a workload frees its slot. Capacity is per session, never per shared login.
    assert _as(ic, agent, ic.cmd_release, issue=[1], pr="https://github.com/ModelCloud/GPTQModel/pull/1") == 0
    assert _as(ic, agent, ic.cmd_claim, issue=[3], capacity=2) == 0
    assert _as(ic, HOSTS_SESSIONS[1], ic.cmd_claim, issue=[4]) == 0


def test_capacity_counts_claims_missing_from_the_local_ledger(ic, tmp_path):
    agent = HOSTS_SESSIONS[0]
    assert _as(ic, agent, ic.cmd_claim, issue=[1]) == 0
    for ledger in tmp_path.glob("*.json"):
        ledger.unlink()
    # Search still finds the live claim (for example one made from another worktree).
    assert _as(ic, agent, ic.cmd_claim, issue=[2]) == 5


def test_correlated_issues_claim_as_one_workload(ic):
    fake = ic.fake
    agent = HOSTS_SESSIONS[0]
    assert _as(ic, agent, ic.cmd_claim, issue=[1, 2]) == 2, "several issues need a stated correlation"
    assert fake.comments == {}

    assert _as(ic, agent, ic.cmd_claim, issue=[2, 1], related="same per-linear GSQ root cause") == 0
    held = _held(ic, agent)
    assert list(held.values()) == [[1, 2]], "correlated issues use one capacity slot"
    claim = fake.fetch("R", 1).holder(fake.clock)
    assert claim.meta["workload_issues"] == [1, 2] and "#1, #2" in fake.thread(1)[0]["body"]

    # A correlated issue discovered mid-work joins the running workload without extra capacity.
    assert _as(ic, agent, ic.cmd_claim, issue=[1, 3], related="same root cause, found while fixing #1") == 0
    assert list(_held(ic, agent).values()) == [[1, 2, 3]]
    assert fake.fetch("R", 1).holder(fake.clock).meta["workload_issues"] == [1, 3]
    assert _as(ic, agent, ic.cmd_claim, issue=[4]) == 5


def test_correlated_workload_is_all_or_nothing(ic):
    fake = ic.fake
    agent, rival = HOSTS_SESSIONS[:2]

    # A rival already holds one of the issues: nothing is posted.
    assert _as(ic, rival, ic.cmd_claim, issue=[2]) == 0
    assert _as(ic, agent, ic.cmd_claim, issue=[1, 2], related="shared fix") == 2
    assert fake.thread(1) == []

    # A rival wins one issue during the race: every new claim in the workload is withdrawn.
    def rival_slips_in(number):
        if number == 4:
            fake.before_post = None
            fake.local.host_session, mine = rival, fake.local.host_session
            assert ic.cmd_claim(_args(issue=[4], capacity=2)) == 0
            fake.local.host_session = mine

    fake.before_post = rival_slips_in
    assert _as(ic, agent, ic.cmd_claim, issue=[3, 4], related="shared fix") == 3
    me = _public_id(ic, agent)
    assert not any(me in c["body"] for c in fake.comments.values())
    assert _held(ic, agent) == {}


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


def test_heartbeat_keeps_claim_and_note_alive(ic):
    fake = ic.fake
    owner, rival = HOSTS_SESSIONS[:2]
    assert _as(ic, owner, ic.cmd_claim, note="Bisecting the GSQ crash") == 0
    for _ in range(3):
        fake.clock += timedelta(hours=3)
        assert _as(ic, owner, ic.cmd_heartbeat, note=None) == 0
    assert "Bisecting the GSQ crash" in fake.thread(1)[0]["body"]
    assert _as(ic, rival, ic.cmd_claim) == 2
