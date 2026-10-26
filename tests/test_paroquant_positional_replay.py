# SPDX-License-Identifier: Apache-2.0

import functools
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from gptqmodel.looper.paroquant_processor import (
    ParoQuantProcessor,
    _ParoQuantReplayBatch,
)
from gptqmodel.models.definitions.gemma4 import (
    _GEMMA4_ALL_PER_LAYER_INPUTS,
    _prepare_gemma4_replay_kwargs,
)


class _Rotary(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        layer_type: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones_like(ids), torch.zeros_like(ids)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(layer_idx=0, layer_type="full_attention")

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert _GEMMA4_ALL_PER_LAYER_INPUTS not in kwargs
        return hidden_states + per_layer_input


@pytest.fixture
def processor() -> ParoQuantProcessor:
    model = torch.nn.Module()
    model.rotary = _Rotary()
    definition = SimpleNamespace(model=model, rotary_embedding="rotary")
    definition.prepare_layer_replay_kwargs = functools.partial(
        _prepare_gemma4_replay_kwargs,
        definition,
    )
    processor = object.__new__(ParoQuantProcessor)
    processor.gptq_model = definition
    processor._get_root_rotary = lambda: None
    return processor


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("positional", [False, True])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_gemma_replay_preserves_positional_inputs(
    processor: ParoQuantProcessor,
    streamed: bool,
    positional: bool,
    requires_grad: bool,
) -> None:
    hidden = torch.ones(1, 2, 4, requires_grad=requires_grad)
    per_layer = torch.full_like(hidden, 2.0, requires_grad=requires_grad)
    inputs = [hidden, per_layer] if positional else [hidden]
    kwargs = {_GEMMA4_ALL_PER_LAYER_INPUTS: per_layer.unsqueeze(2)}
    for _ in range(2):
        if streamed:
            result = processor._forward_replay_batch(
                _Layer(),
                replay_batch=_ParoQuantReplayBatch(
                    inputs=inputs,
                    input_kwargs=kwargs,
                    target=hidden,
                    position_ids=None,
                    attention_mask=None,
                    row_count=2,
                ),
                cache_kwargs=True,
            )
        else:
            result = processor._forward_group_batch(
                _Layer(),
                batch_index=0,
                input_batch=inputs,
                input_kwargs=kwargs,
                attention_mask=None,
                position_ids=None,
            )
        torch.testing.assert_close(result, torch.full_like(hidden, 3.0))
        assert set(kwargs) == {_GEMMA4_ALL_PER_LAYER_INPUTS}
        if requires_grad:
            grad_hidden, grad_per_layer = torch.autograd.grad(result.sum(), (hidden, per_layer))
            torch.testing.assert_close(grad_hidden, torch.ones_like(hidden))
            torch.testing.assert_close(grad_per_layer, torch.ones_like(per_layer))


def test_prepared_cache_distinguishes_positional_layout(
    processor: ParoQuantProcessor,
) -> None:
    layer = _Layer()
    hidden = torch.ones(1, 2, 4)
    per_layer = torch.full_like(hidden, 2.0)
    kwargs = {_GEMMA4_ALL_PER_LAYER_INPUTS: per_layer.unsqueeze(2)}
    for inputs in ([hidden], [hidden, per_layer], [hidden]):
        result = processor._forward_group_batch(
            layer,
            batch_index=0,
            input_batch=inputs,
            input_kwargs=kwargs,
            attention_mask=None,
            position_ids=None,
        )
        torch.testing.assert_close(result, torch.full_like(hidden, 3.0))


def test_legacy_kwargs_helper_defaults_to_hidden_states(
    processor: ParoQuantProcessor,
) -> None:
    kwargs = processor._prepare_group_forward_kwargs(
        _Layer(),
        x=torch.ones(1, 2, 4),
        input_kwargs={_GEMMA4_ALL_PER_LAYER_INPUTS: torch.ones(1, 2, 1, 4)},
        attention_mask=None,
        position_ids=None,
    )
    assert kwargs["per_layer_input"].shape == (1, 2, 4)


@pytest.mark.parametrize("cache", [False, True])
def test_secondary_input_gradients_are_not_cached(
    processor: ParoQuantProcessor,
    cache: bool,
) -> None:
    def prepare_kwargs(
        layer: torch.nn.Module,
        layer_input: list[torch.Tensor],
        additional_inputs: dict[str, Any],
        target_device: torch.device,
    ) -> dict[str, Any]:
        return {**additional_inputs, "derived": layer_input[1].square()}

    processor.gptq_model.prepare_layer_replay_kwargs = prepare_kwargs
    layer = torch.nn.Identity()
    hidden = torch.ones(1, 2, 4)
    secondary = torch.full_like(hidden, 2.0, requires_grad=True)
    inputs = [hidden, secondary]
    kwargs = {}
    for _ in range(2):
        prepared = processor._prepare_group_forward_kwargs(
            layer,
            x=hidden,
            layer_inputs=inputs,
            input_kwargs=kwargs,
            attention_mask=None,
            position_ids=None,
            cache=cache,
        )
        (gradient,) = torch.autograd.grad(prepared["derived"].sum(), (secondary,))
        torch.testing.assert_close(gradient, torch.full_like(secondary, 4.0))
