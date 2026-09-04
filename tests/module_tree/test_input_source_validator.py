# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from gptqmodel.looper.input_source_validator import (
    InputSourceCapture,
    InputSourceValidationError,
    validate_input_sources,
)
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.input_source import InputSourceId, NamedInput, UniqueInput
from gptqmodel.quantization.config import QuantizeConfig


def _named(name, module=None):
    return NamedModule(
        module or nn.Linear(4, 4, bias=False),
        name=name,
        full_name=name,
        layer_index=0,
        tree_scope_id="scope",
        subset_id=0,
    )


def _source(*modules):
    return {
        InputSourceId(scope="scope", subset_id=0): list(modules),
    }


def _report(first, second, first_values, second_values, *, atol=0.0, rtol=0.0):
    return validate_input_sources(
        _source(first, second),
        {
            first.full_name: first_values,
            second.full_name: second_values,
        },
        atol=atol,
        rtol=rtol,
    )


def test_validator_accepts_identity_and_equal_values():
    first = _named("first")
    second = _named("second")
    shared = torch.randn(2, 4)
    first_values = [shared]
    second_values = [shared]
    assert _report(first, second, first_values, second_values).ok

    first_values = [torch.ones(2, 4)]
    second_values = [torch.ones(2, 4)]
    assert _report(first, second, first_values, second_values).ok


def test_validator_reports_shape_dtype_value_and_call_count():
    first = _named("first")
    second = _named("second")

    first_values = [torch.ones(2, 4)]
    second_values = [torch.ones(3, 4)]
    report = _report(first, second, first_values, second_values)
    assert not report.ok
    assert report.mismatches[0].reason == "shape"

    first_values = [torch.ones(2, 4)]
    second_values = [torch.ones(2, 4, dtype=torch.float64)]
    report = _report(first, second, first_values, second_values)
    assert report.mismatches[0].reason == "dtype"

    first_values = [torch.ones(2, 4)]
    second_values = [torch.zeros(2, 4)]
    report = _report(first, second, first_values, second_values)
    assert report.mismatches[0].reason == "value"

    first_values = [torch.ones(2, 4), torch.ones(2, 4)]
    second_values = [torch.ones(2, 4)]
    report = _report(first, second, first_values, second_values)
    assert report.mismatches[0].reason == "call_count"

    first_values = [torch.ones(2, 4)]
    second_values = [torch.ones(2, 4) + 1e-5]
    assert _report(
        first,
        second,
        first_values,
        second_values,
        atol=1e-4,
        rtol=1e-4,
    ).ok


def test_validator_ignores_uninvoked_modules_and_reports_diagnostics():
    first = _named("first")
    second = _named("second")
    first_values = [torch.ones(2, 4)]
    second_values = []
    assert _report(first, second, first_values, second_values).ok

    second_values = [torch.zeros(2, 4)]
    report = _report(first, second, first_values, second_values)
    with torch.no_grad():
        try:
            report.raise_if_failed()
        except InputSourceValidationError as error:
            message = str(error)
        else:
            raise AssertionError("expected validation failure")
    assert "first" in message
    assert "second" in message
    assert "(2, 4)" in message


class _KwargModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states=None, **kwargs):
        return hidden_states if hidden_states is not None else kwargs["hidden_states"]


def test_capture_records_first_tensor_input_and_limits_calls():
    named = _named("module", _KwargModule())
    value = torch.randn(2, 4)
    with InputSourceCapture([named], max_calls=4) as capture:
        for _ in range(6):
            named.module(hidden_states=value)
    assert capture.captured[named.full_name] == [value] * 4


def test_capture_prefers_first_positional_tensor_and_removes_hooks():
    named = _named("module")
    positional = torch.randn(2, 4)
    with InputSourceCapture([named]) as capture:
        named.module(positional)
    assert capture.captured[named.full_name] == [positional]
    assert not named.module._forward_pre_hooks

    named.module(positional)
    assert capture.captured[named.full_name] == [positional]


def test_named_and_unique_source_specs_are_hashable():
    assert InputSourceId(scope="scope", name="latent").kind == "named"
    assert InputSourceId(scope="scope", module="module").kind == "unique"
    assert NamedInput("latent") != UniqueInput()


def test_quantize_config_round_trips_input_source_validation():
    config = QuantizeConfig(validate_input_sources=True)
    payload = config.to_dict()
    assert payload["meta"]["validate_input_sources"] is True
    restored = QuantizeConfig.from_quant_config(payload)
    assert restored.validate_input_sources is True
