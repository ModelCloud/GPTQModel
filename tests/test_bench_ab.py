# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The A/B harness has to be trustworthy before its numbers are.

These run on CPU and use a scripted sampler, so they assert the protocol rather
than the machine: which arm runs first, how the noise floor is derived, and that
an identical-work comparison is never reported as an effect.
"""

import pytest

from scripts import bench_ab


@pytest.fixture()
def scripted(monkeypatch):
    """Replace the sampler with a scripted sequence per round."""
    calls: list[str] = []
    plan: dict[str, list[float]] = {"baseline": [], "candidate": []}
    current = {"arm": None}

    def fake_sample_ms(fn, *, iters=0, warmup=0, sync=None):
        del iters, warmup, sync
        arm = fn()
        calls.append(arm)
        current["arm"] = arm
        return plan[arm].pop(0)

    monkeypatch.setattr(bench_ab, "sample_ms", fake_sample_ms)
    monkeypatch.setattr(bench_ab, "_warm_gpu", lambda *a, **k: None)

    def arm(name):
        def run():
            return name
        return run

    def run(rounds, base, cand):
        plan["baseline"] = list(base)
        plan["candidate"] = list(cand)
        return bench_ab.ab(arm("baseline"), arm("candidate"), rounds=rounds, warm_gpu_seconds=0)

    return calls, run


def test_arms_alternate_order(scripted):
    calls, run = scripted
    run(4, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    assert calls == [
        "baseline", "candidate",
        "candidate", "baseline",
        "baseline", "candidate",
        "candidate", "baseline",
    ]


def test_identical_arms_are_never_an_effect(scripted):
    _, run = scripted
    result = run(6, [2.0, 2.1, 1.9, 2.0, 2.05, 1.95], [2.0, 2.1, 1.9, 2.0, 2.05, 1.95])
    assert result.speedup == pytest.approx(1.0)
    assert result.verdict() == "within noise"


def test_noise_floor_follows_the_spread(scripted):
    _, run = scripted
    tight = run(6, [1.0, 1.01, 0.99, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert tight.noise_floor < 0.05

    _, run = scripted
    loose = run(6, [1.0, 2.0, 0.5, 1.5, 0.8, 1.2], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert loose.noise_floor > tight.noise_floor


def test_a_real_change_is_reported_as_an_effect(scripted):
    _, run = scripted
    result = run(6, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert result.speedup == pytest.approx(2.0)
    assert result.verdict() == "effect"


def test_change_inside_the_noise_is_not_an_effect(scripted):
    _, run = scripted
    # 10% faster, but this machine moves 40% between rounds.
    base = [1.0, 1.4, 0.8, 1.2, 1.0, 1.0]      # median 1.0
    cand = [0.9, 1.26, 0.72, 1.08, 0.9, 0.9]   # median 0.9
    result = run(6, base, cand)
    assert result.speedup == pytest.approx(1.0 / 0.9, rel=1e-6)
    assert result.noise_floor > 0.11
    assert result.verdict() == "within noise"


def test_equal_speedups_and_slowdowns_use_the_same_threshold(scripted):
    _, run = scripted
    baseline = [0.9, 0.95, 0.95, 1.0, 1.0, 1.045, 1.045, 1.1]
    faster = run(8, baseline, [1 / 1.1] * 8)
    slower = run(8, baseline, [1.1] * 8)
    assert faster.noise_floor == pytest.approx(0.095)
    assert slower.noise_floor == pytest.approx(0.095)
    assert faster.verdict() == slower.verdict() == "effect"


def test_custom_synchronizer_reaches_each_sample(monkeypatch):
    seen = []
    def sync():
        pass
    monkeypatch.setattr(bench_ab, "_warm_gpu", lambda *_: None)
    monkeypatch.setattr(bench_ab, "sample_ms", lambda fn, **kwargs: (seen.append(kwargs["sync"]), 1.0)[1])
    bench_ab.ab(lambda: None, lambda: None, rounds=4, sync=sync)
    assert seen == [sync] * 8


def test_warmup_uses_selected_device(monkeypatch):
    seen = []
    device = bench_ab.torch.device("cuda:1")
    monkeypatch.setattr(bench_ab, "_warm_gpu", lambda seconds, gpu: seen.append((seconds, gpu)))
    monkeypatch.setattr(bench_ab, "sample_ms", lambda fn, **kwargs: 1.0)
    bench_ab.self_check(lambda: None, rounds=4, warm_gpu_seconds=0.5,
                        warm_gpu_device=device, sync=lambda: None)
    assert seen == [(0.5, device)]


def test_sample_uses_custom_synchronizer():
    calls = []
    bench_ab.sample_ms(lambda: calls.append("work"), iters=2, warmup=1,
                       sync=lambda: calls.append("sync"))
    assert calls == ["work", "sync", "work", "sync", "work", "sync"]


@pytest.mark.parametrize("kwargs", ({"rounds": 3}, {"iters": 0}, {"warmup": -1}))
def test_ab_rejects_invalid_sample_counts(kwargs):
    with pytest.raises(ValueError):
        bench_ab.ab(lambda: None, lambda: None, warm_gpu_seconds=0, **kwargs)


def test_self_check_uses_identical_arms(scripted):
    calls, run = scripted
    calls.clear()
    plan_calls = []

    def work():
        return "baseline"

    original = bench_ab.sample_ms
    bench_ab.sample_ms = lambda fn, **kw: (plan_calls.append(fn()), 1.0)[1]
    try:
        result = bench_ab.self_check(work, rounds=4)
    finally:
        bench_ab.sample_ms = original
    assert plan_calls == ["baseline"] * 8
    assert result.verdict() == "within noise"


def test_stats_percentiles():
    stats = bench_ab.ArmStats((1.0, 2.0, 3.0, 4.0))
    assert stats.minimum == 1.0
    assert stats.median == pytest.approx(2.5)
    assert stats.p25 == pytest.approx(1.75)
    assert stats.p75 == pytest.approx(3.25)


@pytest.mark.parametrize("samples", ((), (1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0)))
def test_stats_reject_insufficient_or_invalid_samples(samples):
    with pytest.raises(ValueError):
        bench_ab.ArmStats(samples)
