# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from tests.models.w4a_producer_reconstruct import relative_token_error, search_scale


class Producer:
    def __init__(self):
        self.global_scale = torch.tensor(1.)

    def set_scale(self, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid scale")
        self.global_scale.fill_(value)


def test_equal_token_objective_matches_scalar_oracle():
    target = torch.tensor([[100., 0.], [1., 2.]], dtype=torch.float64)
    actual = torch.tensor([[99., 1.], [0., 3.]], dtype=torch.float64)
    error, count = relative_token_error(actual, target)
    assert count == 2
    assert error == pytest.approx(2 / 10000 + 2 / 5, abs=1e-12)


def test_objective_does_not_hide_small_tokens():
    error, count = relative_token_error(torch.tensor([[100., 0.], [0., 0.]]),
                                        torch.tensor([[100., 0.], [1., 0.]]))
    assert error / count == .5


@pytest.mark.parametrize("actual,target", [
    (torch.zeros(2, 3), torch.zeros(2, 3)),
    (torch.ones(2, 3), torch.ones(3, 3)),
    (torch.ones(3), torch.ones(3)),
    (torch.empty(0, 3), torch.empty(0, 3)),
    (torch.full((2, 3), float("nan")), torch.ones(2, 3)),
])
def test_objective_rejects_invalid_inputs(actual, target):
    with pytest.raises(ValueError):
        relative_token_error(actual, target)


def test_search_selects_strict_minimum_and_keeps_scale_on_ties():
    producer = Producer()
    report = search_scale(producer, lambda: (float(producer.global_scale) - 1.25) ** 2,
                          (.75, 1., 1.25, 1.5))
    assert producer.global_scale.item() == 1.25
    assert report["selected_loss"] == 0
    assert report["initial_loss"] == .25 ** 2
    assert len(report["trials"]) == 4
    report = search_scale(producer, lambda: 1., (.75, 1., 1.5))
    assert producer.global_scale.item() == 1.25
    assert report["selected_scale"] == report["initial_scale"]


@pytest.mark.parametrize("failure", ["exception", "nan", "negative"])
def test_search_restores_scale_after_candidate_failure(failure):
    producer = Producer()
    def objective():
        if producer.global_scale.item() != 1.:
            if failure == "exception":
                raise RuntimeError("candidate failure")
            return float("nan") if failure == "nan" else -1.
        return 2.
    with pytest.raises((ValueError, RuntimeError)):
        search_scale(producer, objective, (.75, 1.25))
    assert producer.global_scale.item() == 1.


@pytest.mark.parametrize("ratios", [(), (0.,), (-1.,), (float("nan"),), (True,)])
def test_search_rejects_invalid_ratios(ratios):
    with pytest.raises(ValueError):
        search_scale(Producer(), lambda: 0., ratios)


def test_checkpoint_shell_does_not_inherit_stale_producer_reports(tmp_path):
    from tests.models.w4a_nvfp4_norm_qat import _copy_checkpoint_shell

    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    names = ("w4a_producer_calibration_report.json", "w4a_producer_reconstruction_report.json")
    for name in names:
        (source / name).write_text("source report")
    (source / "config.json").write_text("{}")
    _copy_checkpoint_shell(source, output)
    assert (output / "config.json").is_symlink()
    for name in names:
        assert not (output / name).exists()
        (output / name).write_text("new report")
        assert (source / name).read_text() == "source report"
