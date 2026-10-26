# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Cached MiMo layer replay must match the native hybrid-attention forward."""

from importlib import import_module

import pytest
import torch
from test_mimo_gptq_roundtrip import make_source
from torch import nn

from gptqmodel.models.definitions.mimo_v2 import MimoV2QModel


@pytest.mark.parametrize("mask_kind", ["none", "padding", "mapping"])
@pytest.mark.parametrize("first_sliding", [False, True])
def test_mimo_layer_replay_matches_native_forward(tmp_path, first_sliding, mask_kind):
    model, _ = make_source(tmp_path / "source", tp=4)
    if first_sliding:
        model.model.layers = nn.ModuleList(reversed(model.model.layers))
    instance = object.__new__(MimoV2QModel)
    nn.Module.__init__(instance)
    instance.model = model
    layers = model.model.layers
    ids = torch.arange(4, 20).repeat(2, 1)
    positions = torch.arange(17, 33).repeat(2, 1)
    mask = torch.ones_like(ids)
    mask[0, :2] = 0
    mask[1, -3:] = 0
    if mask_kind == "none":
        mask = None
    elif mask_kind == "mapping":
        modeling = import_module(type(model).__module__)
        mask_kwargs = {"config": model.config, "input_embeds": model.model.embed_tokens(ids), "attention_mask": mask,
                       "cache_position": torch.arange(16), "past_key_values": None, "position_ids": positions}
        mask = {"full_attention": modeling.create_causal_mask(**mask_kwargs),
                "sliding_window_attention": modeling.create_sliding_window_causal_mask(**mask_kwargs)}
    captured = {}
    expected_outputs = []

    def capture_first(module, args, kwargs):
        captured["hidden"] = args[0]
        captured["mask"] = kwargs["attention_mask"]
        captured["kwargs"] = instance.capture_first_layer_input_kwargs(
            args, kwargs, torch.device("cpu"),
            {key: value for key, value in kwargs.items() if key not in ("attention_mask", "position_ids")},
        )

    def capture_output(module, args, output):
        expected_outputs.append(output)

    handles = [layers[0].register_forward_pre_hook(capture_first, with_kwargs=True)]
    handles.extend(layer.register_forward_hook(capture_output) for layer in layers)
    with torch.inference_mode():
        instance.run_input_capture(
            {"input_ids": ids, "attention_mask": mask, "position_ids": positions},
            use_cache=False, data_device=torch.device("cpu"),
        )
    for handle in handles:
        handle.remove()

    # Replay runs after the outer rotary modules have been offloaded.
    model.model.rotary_emb.to("meta")
    model.model.swa_rotary_emb.to("meta")
    hidden = captured["hidden"]
    for layer, expected in zip(layers, expected_outputs):
        # The shared executors replay the first layer's mask and positions for every layer.
        replay_kwargs = {**captured["kwargs"], "attention_mask": captured["mask"], "position_ids": positions,
                         "use_cache": False}
        actual_kwargs = instance.prepare_layer_replay_kwargs(layer, [hidden], replay_kwargs, torch.device("cpu"))
        assert "_mimo_v2_replay_inputs" not in actual_kwargs
        with torch.inference_mode():
            hidden = layer(hidden, **actual_kwargs)
        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
