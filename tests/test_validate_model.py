# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""model.validate() on a real (tiny) model: identity is exact, drift is visible.

The metric maths are pinned in test_validate_metrics.py; this file checks the
wiring - token rows in, reference model resolved, report out - and the one
property that matters for trust: comparing a model with itself must produce
zeros, and a perturbed copy must move the numbers.
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.validate import (
    KLDConfig,
    Metric,
    PPLConfig,
    RMSEConfig,
    TOP_NConfig,
    ValidateConfig,
)
from gptqmodel.validate.runner import ValidationError


VOCAB = 128
ROWS = [
    [5, 9, 12, 33, 40, 7, 19, 22, 88, 61],
    [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    source = tmp_path_factory.mktemp("validate") / "native"
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    model.save_pretrained(source)
    tokenizer = Tokenizer(WordLevel(vocab={"<pad>": 0, "<s>": 1, "eos": 2}, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<s>", eos_token="eos",
    ).save_pretrained(source)

    loaded = GPTQModel.load(str(source), device="cpu")
    return loaded, source


def _config(reference, **overrides):
    base = {
        "metrics": (PPLConfig(window=8, stride=4), KLDConfig(positions=4),
                    TOP_NConfig(top_k=(1, 5)), RMSEConfig()),
        "dataset": ROWS,
        "reference": reference,
        "max_batches": 2,
    }
    base.update(overrides)
    return ValidateConfig(**base)


def test_identity_is_exact_through_the_model_api(tiny_model):
    model, _source = tiny_model

    report = model.validate(_config(reference=model.model))

    assert report.values["KLD"] == pytest.approx(0.0, abs=1e-6)
    assert report.values["RMSE"] == pytest.approx(0.0, abs=1e-6)
    for width, value in report.values["TOP_N"].items():
        assert value == 1.0, width
    assert report.values["PPL"] == pytest.approx(report.reference["PPL"], rel=1e-6)
    assert report.passed is None  # nothing was checked, so nothing passed
    assert report.batches == 2


def test_perturbed_weights_move_the_metrics(tiny_model):
    model, _source = tiny_model
    reference = model.model

    perturbed = type(reference)(reference.config)
    perturbed.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for parameter in perturbed.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    perturbed.eval()

    report = model.validate(_config(reference=perturbed))

    assert report.values["KLD"] > 0
    assert report.values["RMSE"] > 0
    assert report.values["TOP_N"][1] < 1.0


def test_thresholds_trip_and_strict_raises(tiny_model):
    model, _source = tiny_model
    reference = model.model

    loose = model.validate(_config(reference=reference, thresholds={"KLD": 1.0}))
    assert loose.passed is True

    # Identity satisfies a TOP_N floor of 1.0, so break it explicitly instead.
    broken_report = model.validate(
        _config(reference=_perturbed(model.model), thresholds={"TOP_N": 1.0})
    )
    assert broken_report.passed is False
    assert "TOP_N" in broken_report.violations

    with pytest.raises(ValidationError):
        model.validate(
            _config(reference=_perturbed(model.model), thresholds={"TOP_N": 1.0}, strict=True)
        )


def test_missing_reference_and_dataset_are_actionable(tiny_model):
    model, _source = tiny_model

    with pytest.raises(ValueError, match="needs `reference`"):
        model.validate(ValidateConfig(dataset=ROWS))
    with pytest.raises(ValueError, match="needs `dataset`"):
        model.validate(ValidateConfig(reference=model.model))


def test_only_requested_metrics_appear(tiny_model):
    model, _source = tiny_model

    report = model.validate(
        ValidateConfig(metrics=(Metric.KLD,), dataset=ROWS, reference=model.model, max_batches=2)
    )

    assert set(report.values) == {"KLD"}
    assert report.positions > 0


def _perturbed(reference):
    copy = type(reference)(reference.config)
    copy.load_state_dict(reference.state_dict())
    with torch.no_grad():
        for parameter in copy.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    copy.eval()
    return copy
