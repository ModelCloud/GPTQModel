# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""model.measure() on a real (tiny) model: identity is exact, drift is visible.

The metric maths and grading are pinned in test_measure_metrics.py; this file
checks the wiring - token rows in, reference resolved, one fused forward per
batch, report out - and the property that makes the numbers trustworthy:
comparing a model with itself must produce exact zeros, and a perturbed copy must
move them.
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.measure import (
    GradedThresholdConfig,
    KLDConfig,
    MeasureConfig,
    MeasurementError,
    Metric,
    MetricGrades,
    PPLConfig,
    RMSEConfig,
    TOP_NConfig,
)
from gptqmodel.measure.runner import measure_models


VOCAB = 128
ROWS = [
    [5, 9, 12, 33, 40, 7, 19, 22, 88, 61],
    [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    source = tmp_path_factory.mktemp("measure") / "native"
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer = Tokenizer(WordLevel(vocab={"<pad>": 0, "<s>": 1, "eos": 2}, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<s>", eos_token="eos",
    ).save_pretrained(source)
    return GPTQModel.load(str(source), device="cpu"), source


def _config(reference, **overrides):
    base = {
        "metrics": (PPLConfig(window=8, stride=4), KLDConfig(positions=4),
                    TOP_NConfig(top_k=(1, 5)), RMSEConfig()),
        "dataset": ROWS,
        "reference": reference,
        "max_batches": 2,
    }
    base.update(overrides)
    return MeasureConfig(**base)


def _perturbed(reference, scale=0.05):
    copy = type(reference)(reference.config)
    copy.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for parameter in copy.parameters():
            parameter.add_(torch.randn_like(parameter) * scale)
    return copy.eval()


class _Counting(torch.nn.Module):
    """Delegates to a model and counts forwards, to prove the metrics fuse."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.inner(*args, **kwargs)


def test_identity_is_exact_through_the_model_api(tiny_model):
    model, _source = tiny_model

    report = model.measure(_config(reference=model.model))

    assert report.values["KLD"] == pytest.approx(0.0, abs=1e-6)
    assert report.values["RMSE"] == pytest.approx(0.0, abs=1e-6)
    for width, value in report.values["TOP_N"].items():
        assert value == 1.0, width
    assert report.values["PPL"] == pytest.approx(report.reference["PPL"], rel=1e-6)
    assert report.overall is None  # no grades configured
    assert report.passed is None


def test_all_four_metrics_share_one_forward_per_batch(tiny_model):
    model, _source = tiny_model
    reference = _Counting(model.model)
    candidate = _Counting(model.model)

    report = measure_models(candidate, reference, [torch.tensor(row).unsqueeze(0) for row in ROWS],
                            _config(reference=reference))

    assert reference.calls == 2      # one per batch, not one per metric
    assert candidate.calls == 2
    assert set(report.values) == {"PPL", "KLD", "TOP_N", "RMSE"}


def test_perturbed_weights_move_the_metrics(tiny_model):
    model, _source = tiny_model

    report = model.measure(_config(reference=_perturbed(model.model)))

    assert report.values["KLD"] > 0
    assert report.values["RMSE"] > 0
    assert report.values["TOP_N"][1] < 1.0


def test_grades_come_from_the_metric_configs(tiny_model):
    model, _source = tiny_model

    graded_pass = model.measure(
        MeasureConfig(metrics=(KLDConfig(positions=4, threshold=GradedThresholdConfig(
            grades={MetricGrades.PASS: "<=1.0", MetricGrades.FAIL: "fallthrough"})),),
            dataset=ROWS, reference=model.model, max_batches=2)
    )
    assert graded_pass.overall is MetricGrades.PASS

    graded_fail = model.measure(
        MeasureConfig(metrics=(KLDConfig(positions=4, threshold=GradedThresholdConfig(
            grades={MetricGrades.PASS: "<=0.0", MetricGrades.FAIL: "fallthrough"})),),
            dataset=ROWS, reference=_perturbed(model.model), max_batches=2)
    )
    assert graded_fail.overall is MetricGrades.FAIL
    assert "KLD" in graded_fail.violations


def test_strict_raises_only_on_a_failing_grade(tiny_model):
    model, _source = tiny_model
    failing = GradedThresholdConfig(
        grades={MetricGrades.PASS: "<=0.0", MetricGrades.FAIL: "fallthrough"}
    )

    with pytest.raises(MeasurementError):
        model.measure(MeasureConfig(
            metrics=(KLDConfig(positions=4, threshold=failing),),
            dataset=ROWS, reference=_perturbed(model.model), max_batches=2, strict=True,
        ))

    warning = GradedThresholdConfig(
        grades={MetricGrades.WARN: "<=1.0", MetricGrades.FAIL: "fallthrough"}
    )
    report = model.measure(MeasureConfig(
        metrics=(KLDConfig(positions=4, threshold=warning),),
        dataset=ROWS, reference=_perturbed(model.model), max_batches=2, strict=True,
    ))
    assert report.overall is MetricGrades.WARN  # a warning never raises


def test_missing_reference_and_dataset_are_actionable(tiny_model):
    model, _source = tiny_model

    with pytest.raises(ValueError, match="needs `reference`"):
        model.measure(MeasureConfig(dataset=ROWS))
    with pytest.raises(ValueError, match="needs `dataset`"):
        model.measure(MeasureConfig(reference=model.model))


def test_only_requested_metrics_appear(tiny_model):
    model, _source = tiny_model

    report = model.measure(
        MeasureConfig(metrics=Metric.KLD, dataset=ROWS, reference=model.model, max_batches=2)
    )

    assert set(report.values) == {"KLD"}
    assert report.positions > 0
