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

import math

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
    TOPKConfig,
)
from gptqmodel.measure.runner import capture_reference, measure_against, measure_models
from gptqmodel.models.base import BaseQModel


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
                    TOPKConfig(k=(1, 5)), RMSEConfig()),
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
    for width, value in report.values["TOP_K"].items():
        assert value == 1.0, width
    assert report.values["PPL"] == pytest.approx(report.reference["PPL"], rel=1e-6)
    assert report.overall is None  # no grades configured
    assert report.passed is None


class _ConstantLogits(torch.nn.Module):
    """Constant logits, independent of the tokens: a hand-written oracle."""

    def __init__(self, logits, dtype=torch.float32):
        super().__init__()
        values = logits if torch.is_tensor(logits) else torch.tensor(logits)
        self.register_buffer("values", values.to(dtype))

    def forward(self, input_ids, attention_mask):
        return self.values.expand(*input_ids.shape, self.values.numel())


class _MeasureHost:
    """Borrow BaseQModel.measure() without loading a checkpoint."""

    measure = BaseQModel.measure
    _measurement_reference = BaseQModel._measurement_reference


def _measure_fixed_logits(candidate, reference):
    host = _MeasureHost()
    host.model = candidate
    return host.measure(MeasureConfig(
        metrics=(KLDConfig(), RMSEConfig(), TOPKConfig(k=(1,))),
        dataset=[[0, 0]],
        reference=reference,
    )).values


def _fp64_kl(reference_logits, candidate_logits):
    """Independent KL oracle: FP64 log-softmax of the supplied logits."""

    reference = torch.log_softmax(torch.tensor(reference_logits, dtype=torch.float64), dim=-1)
    candidate = torch.log_softmax(torch.tensor(candidate_logits, dtype=torch.float64), dim=-1)
    return float((reference.exp() * (reference - candidate)).sum())


def test_fp32_top_token_flip_is_visible_through_the_model_api():
    reference_logits = [16.0, 16.0 + 1 / 256]
    candidate_logits = [16.0 + 1 / 256, 16.0]

    values = _measure_fixed_logits(
        _ConstantLogits(candidate_logits), _ConstantLogits(reference_logits)
    )

    # Both logits narrow to 16 in FP16, which hid the flip entirely.
    assert values["TOP_K"][1] == 0.0
    assert values["RMSE"] == pytest.approx(1 / 256, rel=1e-6)
    assert values["KLD"] == pytest.approx(
        _fp64_kl(reference_logits, candidate_logits), rel=1e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_finite_identity_above_the_fp16_range_is_exact_through_the_model_api(dtype):
    logits = torch.tensor([65536.0, 0.0], dtype=dtype)
    assert torch.isfinite(logits).all()  # finite input, not an overflow fixture

    values = _measure_fixed_logits(
        _ConstantLogits(logits, dtype=dtype), _ConstantLogits(logits, dtype=dtype)
    )

    # FP16 capture made 65536 infinite, so identity reported NaN.
    assert values["KLD"] == 0.0
    assert values["RMSE"] == 0.0
    assert values["TOP_K"][1] == 1.0


def test_all_four_metrics_share_one_forward_per_batch(tiny_model):
    model, _source = tiny_model
    reference = _Counting(model.model)
    candidate = _Counting(model.model)

    report = measure_models(candidate, reference, [torch.tensor(row).unsqueeze(0) for row in ROWS],
                            _config(reference=reference))

    assert reference.calls == 2      # one per batch, not one per metric
    assert candidate.calls == 2
    assert set(report.values) == {"PPL", "KLD", "TOP_K", "RMSE"}


def test_perturbed_weights_move_the_metrics(tiny_model):
    model, _source = tiny_model

    report = model.measure(_config(reference=_perturbed(model.model)))

    assert report.values["KLD"] > 0
    assert report.values["RMSE"] > 0
    assert report.values["TOP_K"][1] < 1.0


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

def test_batch_groups_equal_length_rows(tiny_model):
    model, _source = tiny_model
    reference = _Counting(model.model)
    candidate = _Counting(model.model)
    rows = [torch.tensor(row).unsqueeze(0) for row in ROWS]  # all length 10

    report = measure_models(candidate, reference, rows, MeasureConfig(
        metrics=Metric.PPL, dataset=ROWS, reference=reference, batch=2, max_batches=4,
    ))

    assert reference.calls == 1      # 2 rows of equal length -> 1 batch
    assert candidate.calls == 1
    assert report.batches == 1


def test_unequal_rows_are_padded_safely_into_one_batch(tiny_model):
    model, _source = tiny_model
    reference = _perturbed(model.model)
    rows = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6, 7]])]
    metrics = (PPLConfig(window=8, stride=4), KLDConfig(positions=2))
    config = {"metrics": metrics, "dataset": ROWS, "reference": reference}

    batched = measure_models(model.model, reference, rows,
                             MeasureConfig(batch=2, **config))
    separate = measure_models(model.model, reference, rows,
                              MeasureConfig(batch=1, **config))

    # Unequal rows now batch. Padding is inert where it must be - the padded
    # targets are masked out of the perplexity sum, which is a pure average -
    # and the divergence survives only as float reduction-order noise.
    assert batched.batches == 1
    assert separate.batches == 2
    assert batched.positions == separate.positions
    assert batched.values["PPL"] == pytest.approx(separate.values["PPL"], rel=1e-4)
    assert batched.values["KLD"] == pytest.approx(separate.values["KLD"], rel=1e-2)


def test_positions_reports_what_was_actually_sampled(tiny_model):
    model, _source = tiny_model
    rows = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6, 7]])]

    report = measure_models(model.model, model.model, rows, MeasureConfig(
        metrics=KLDConfig(positions=64), dataset=ROWS, reference=model.model,
    ))

    # 2 usable positions in the first row and 3 in the second, not the 64 asked
    # for: the report must not claim a sample it did not read.
    assert report.positions == 5


class _FixedLogits(torch.nn.Module):
    """Token-independent logits; the candidate flips one known position.

    Every row sees the same two-vocabulary distribution, so the expected
    divergence is arithmetic rather than model-dependent. Exactly one of the
    nine sampled next-token positions (the first token of the short row) is
    flipped, which makes the expected KLD, RMSE and top-1 exact.
    """

    def __init__(self, candidate=False):
        super().__init__()
        self.candidate = candidate

    def forward(self, input_ids, attention_mask):
        logits = torch.tensor([2.0, 0.0]).expand(*input_ids.shape, 2)
        if not self.candidate:
            return logits
        return torch.where((input_ids[:, :1] == 1).unsqueeze(-1), logits.flip(-1), logits)


def test_batch_size_does_not_change_sampled_positions_or_metrics():
    # Ragged rows: lengths 2, 5 and 5. Batch 1/2/3 all read the same nine
    # transitions; batch=2 and batch=3 exercise a short final batch.
    rows = [
        torch.tensor([[1, 0]]),
        torch.tensor([[0, 0, 0, 0, 0]]),
        torch.tensor([[0, 0, 0, 0, 0]]),
    ]
    config = MeasureConfig(
        metrics=(KLDConfig(positions=64), RMSEConfig(), TOPKConfig(k=(1,))),
        batch=1,
        max_batches=3,
    )

    reports = {}
    for batch in (1, 2, 3):
        reports[batch] = measure_models(
            _FixedLogits(candidate=True),
            _FixedLogits(),
            rows,
            MeasureConfig(
                metrics=config.metrics,
                batch=batch,
                max_batches=config.max_batches,
            ),
        )

    # 1 usable position in the length-2 row plus 4 in each length-5 row.
    expected_kld = 2.0 * math.tanh(1.0) / 9.0
    expected_rmse = math.sqrt(8.0 / 18.0)
    expected_top1 = 8.0 / 9.0
    for batch, report in reports.items():
        assert report.positions == 9, batch
        assert report.values["KLD"] == pytest.approx(expected_kld, rel=1e-5), batch
        assert report.values["RMSE"] == pytest.approx(expected_rmse, rel=1e-5), batch
        assert report.values["TOP_K"][1] == pytest.approx(expected_top1, rel=1e-5), batch

    assert reports[1].values == reports[2].values == reports[3].values


def test_batch_pools_positions_instead_of_averaging_batch_means():
    # Four rows: a short row that batches with a long one must not have its
    # positions dropped, and the short final batch must not get an equal vote
    # with a full batch. Expected values come from the same exact arithmetic as
    # above, now over 1 + 4 + 4 + 4 = 13 positions with one flipped.
    rows = [
        torch.tensor([[1, 0]]),
        torch.tensor([[0, 0, 0, 0, 0]]),
        torch.tensor([[0, 0, 0, 0, 0]]),
        torch.tensor([[0, 0, 0, 0, 0]]),
    ]

    report = measure_models(
        _FixedLogits(candidate=True),
        _FixedLogits(),
        rows,
        MeasureConfig(
            metrics=(KLDConfig(positions=64), RMSEConfig(), TOPKConfig(k=(1,))),
            batch=3,
            max_batches=4,
        ),
    )

    assert report.batches == 2  # 3 rows, then a one-row final batch
    assert report.positions == 13
    assert report.values["KLD"] == pytest.approx(2.0 * math.tanh(1.0) / 13.0, rel=1e-5)
    assert report.values["RMSE"] == pytest.approx(math.sqrt(8.0 / 26.0), rel=1e-5)
    assert report.values["TOP_K"][1] == pytest.approx(12.0 / 13.0, rel=1e-5)


def test_capture_then_measure_matches_the_one_shot_api(tiny_model):
    model, _source = tiny_model
    reference = _perturbed(model.model)
    rows = [torch.tensor(row).unsqueeze(0) for row in ROWS]
    config = MeasureConfig(
        metrics=(PPLConfig(window=8, stride=4), KLDConfig(positions=2)),
        dataset=ROWS, reference=reference, batch=2, max_batches=2,
    )

    capture = capture_reference(reference, rows, config)
    split = measure_against(model.model, capture, config)
    one_shot = measure_models(model.model, reference, rows, config)

    assert split.values == one_shot.values
    assert split.reference == one_shot.reference
    assert split.batches == one_shot.batches

def test_measure_against_rejects_a_capture_missing_the_requested_metric(tiny_model):
    model, _source = tiny_model
    reference = _perturbed(model.model)
    rows = [torch.tensor(row).unsqueeze(0) for row in ROWS]

    ppl_only = MeasureConfig(
        metrics=(PPLConfig(window=8, stride=4),),
        dataset=ROWS, reference=reference, batch=2, max_batches=2,
    )
    ppl_capture = capture_reference(reference, rows, ppl_only)
    with pytest.raises(ValueError, match="KLD"):
        measure_against(model.model, ppl_capture, _config(reference=reference))

    kld_only = MeasureConfig(
        metrics=(KLDConfig(positions=2),),
        dataset=ROWS, reference=reference, batch=2, max_batches=2,
    )
    kld_capture = capture_reference(reference, rows, kld_only)
    with pytest.raises(ValueError, match="PPL"):
        measure_against(
            model.model,
            kld_capture,
            MeasureConfig(
                metrics=(PPLConfig(window=8, stride=4),),
                dataset=ROWS, reference=reference, batch=2, max_batches=2,
            ),
        )

