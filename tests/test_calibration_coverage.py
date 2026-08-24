from __future__ import annotations

import optimize.calibration_coverage as coverage


class _Profile:
    def __init__(self, names=(), total_tokens=0):
        self.names = frozenset(names)
        self.total_tokens = total_tokens

    def merge(self, other):
        return _Profile(self.names | other.names, self.total_tokens + other.total_tokens)


def test_minimum_tokens_admits_only_least_redundant_candidate(monkeypatch):
    profiles = {
        "positive": _Profile({"positive"}, 100),
        "least_redundant": _Profile({"least_redundant"}, 100),
        "more_redundant": _Profile({"more_redundant"}, 100),
    }
    scores = {
        frozenset(): 100.0,
        frozenset({"positive"}): 10.0,
        frozenset({"least_redundant"}): 110.0,
        frozenset({"more_redundant"}): 120.0,
        frozenset({"positive", "least_redundant"}): 11.0,
        frozenset({"positive", "more_redundant"}): 15.0,
        frozenset({"positive", "least_redundant", "more_redundant"}): 20.0,
    }
    monkeypatch.setattr(
        coverage.DatasetProfile,
        "empty_like",
        staticmethod(lambda name, ref, max_samples: _Profile()),
    )
    monkeypatch.setattr(coverage, "score", lambda profile, ref: scores[profile.names])

    selected, order, _, _, warnings = coverage.greedy_select(
        profiles,
        _Profile(),
        min_gain=0.0,
        max_samples=1,
        greedy_threads=1,
        target_tokens=500,
        min_target_tokens=200,
        target_tokens_mode="gain",
    )

    assert [row["name"] for row in order] == ["positive", "least_redundant"]
    assert selected.total_tokens == 200
    assert warnings == [
        "Desired token target 500 not reached (reached 200); positive conditional gain is exhausted."
    ]
