# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Contiguity invariant checks for model parameters and buffers."""

import pytest
import torch

from gptqmodel.utils.tensor_layout import (
    assert_model_tensors_contiguous,
    find_non_contiguous_tensors,
    iter_model_tensors,
)


def test_contiguous_model_reports_nothing():
    module = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))

    assert find_non_contiguous_tensors(module) == []
    assert_model_tensors_contiguous(module, label="post-load")


def test_transposed_parameter_is_reported_with_its_layout():
    module = torch.nn.Linear(8, 4)
    module.weight = torch.nn.Parameter(module.weight.data.t())

    offenders = find_non_contiguous_tensors(module)

    assert len(offenders) == 1
    offender = offenders[0]
    assert offender.name == "weight"
    assert offender.kind == "parameter"
    assert offender.shape == (8, 4)
    # A transposed view keeps the original storage, so the stride is swapped.
    assert offender.stride == (1, 8)


def test_non_contiguous_buffer_is_reported():
    module = torch.nn.Module()
    module.register_buffer("mask", torch.zeros(4, 4).t())

    offenders = find_non_contiguous_tensors(module)

    assert [item.name for item in offenders] == ["mask"]
    assert offenders[0].kind == "buffer"


def test_assert_raises_with_actionable_message():
    module = torch.nn.Linear(8, 4)
    module.weight = torch.nn.Parameter(module.weight.data.t())

    with pytest.raises(ValueError) as excinfo:
        assert_model_tensors_contiguous(module, label="post-quantize")

    message = str(excinfo.value)
    assert "post-quantize" in message
    assert "non-contiguous" in message
    assert "weight" in message


def test_assert_lists_only_the_requested_number_of_offenders():
    module = torch.nn.Sequential(*[torch.nn.Linear(4, 4) for _ in range(5)])
    for submodule in module:
        submodule.weight = torch.nn.Parameter(submodule.weight.data.t())

    with pytest.raises(ValueError) as excinfo:
        assert_model_tensors_contiguous(module, label="post-quantize", max_report=2)

    message = str(excinfo.value)
    assert "5 non-contiguous" in message
    assert "and 3 more" in message


def test_tied_parameter_is_yielded_once():
    first = torch.nn.Linear(4, 4)
    second = torch.nn.Linear(4, 4)
    second.weight = first.weight
    holder = torch.nn.Sequential(first, second)

    weight_names = [
        name for name, kind, _ in iter_model_tensors(holder) if kind == "parameter" and name.endswith("weight")
    ]

    assert weight_names == ["0.weight"]


def test_meta_tensors_are_skipped():
    module = torch.nn.Module()
    module.register_buffer("offloaded", torch.empty(4, 4, device="meta"))
    module.register_buffer("real", torch.zeros(4, 4))

    names = [name for name, _, _ in iter_model_tensors(module)]

    assert names == ["real"]
    assert_model_tensors_contiguous(module, label="post-load")


def test_none_entries_are_skipped():
    module = torch.nn.Linear(4, 2, bias=False)
    module.register_buffer("empty", None)

    names = [name for name, _, _ in iter_model_tensors(module)]

    assert names == ["weight"]
    assert_model_tensors_contiguous(module, label="post-load")
