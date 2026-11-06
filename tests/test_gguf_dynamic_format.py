# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A per-module GGUF `format` in `dynamic` must select the ggml type.

A width alone is ambiguous: width 4 is `Q4_0` by default, width 5 is `Q5_K_M`
and width 2 is `Q2_0`. `GGUFConfig` accepts a separate `format` so a plan can
pick a different type at the same width, but `create_quant_module` used to read
only the digits and fall back to the default type for that width. A mixed-type
checkpoint therefore reloaded every module at the *default* type for its width,
silently discarding the requested one.
"""

import json

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import (
    _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH,
    GGUFConfig,
    QuantizeConfig,
    clone_weight_only_config_for_module,
)
from gptqmodel.utils.model import create_quant_module


NAME = "model.layers.0.mlp.down_proj"

# (requested alias, width) pairs where the requested type is NOT the default
# type for that width -- exactly the cases the bug destroyed.
NON_DEFAULT = [
    ("q4_k", 4),  # default for 4 is q4_0
    ("q4_1", 4),  # default for 4 is q4_0
    ("q2_k", 2),  # default for 2 is q2_0
    ("q5_0", 5),  # default for 5 is q5_k_m
]


def _gguf_module(config, *, dynamic=None):
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.model.layers[0].mlp = torch.nn.Module()
    root.model.layers[0].mlp.down_proj = torch.nn.Linear(256, 256, bias=False)

    create_quant_module(
        name=NAME,
        linear_cls=GGUFTorchLinear,
        bits=config.runtime_bits,
        desc_act=False,
        dynamic=config.dynamic if dynamic is None else dynamic,
        group_size=-1,
        module=root,
        submodule=root.get_submodule(NAME),
        sym=True,
        device="cpu",
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=FORMAT.GGUF,
        register_buffers=True,
    )
    return root.get_submodule(NAME)


def test_the_cases_really_are_non_default():
    """Guard the table above: a default-type case would not exercise the bug."""
    for alias, width in NON_DEFAULT:
        assert _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH[width] != alias


@pytest.mark.parametrize("alias,width", NON_DEFAULT)
def test_per_module_format_beats_the_default_type_for_that_width(alias, width):
    config = GGUFConfig(bits="q4_k", dynamic={NAME: {"bits": width, "format": alias}})
    assert _gguf_module(config).gguf_tensor_qtype == alias.upper()


@pytest.mark.parametrize(
    "base,overrides,expected",
    [
        ("q4_k", {"bits": 4}, "Q4_K"),
        ("q3_k", {"bits": 4}, "Q4_K"),
        ("q4_1", {"bits": 5}, "Q5_1"),
        ("q4_0", {"format": "q_k"}, "Q4_K"),
        ("q4_0", {"bits": 4, "format": "q_k"}, "Q4_K"),
        ("q4_k", {"bits": "q2_k"}, "Q2_K"),
    ],
)
def test_reload_uses_the_same_type_and_packed_layout_as_quantization(base, overrides, expected):
    config = GGUFConfig(bits=base, dynamic={NAME: overrides})
    per_module_config = clone_weight_only_config_for_module(config, NAME)
    quantized = _gguf_module(per_module_config, dynamic={})
    assert quantized.gguf_tensor_qtype == expected
    quantized.qweight.copy_(torch.arange(quantized.qweight.numel()).reshape_as(quantized.qweight).to(torch.uint8))

    reloaded_config = QuantizeConfig.from_quant_config(json.loads(json.dumps(config.to_dict())))
    reloaded = _gguf_module(reloaded_config)
    assert reloaded.gguf_tensor_qtype == expected
    reloaded.load_state_dict(quantized.state_dict(), strict=True)
    assert torch.equal(reloaded.qweight, quantized.qweight)


def test_raw_format_only_override_replaces_the_inherited_type():
    config = GGUFConfig(bits="q4_0")
    assert _gguf_module(config, dynamic={NAME: {"format": "q_k"}}).gguf_tensor_qtype == "Q4_K"


@pytest.mark.parametrize("dynamic", [None, {}, {"other_module": {"bits": 2}}, {NAME: {"sym": True}}])
def test_no_type_override_preserves_the_base_type(dynamic):
    config = GGUFConfig(bits="q4_k", dynamic=dynamic)
    assert _gguf_module(config).gguf_tensor_qtype == "Q4_K"


def test_negative_dynamic_match_keeps_the_original_module():
    config = GGUFConfig(bits="q4_k", dynamic={f"-:{NAME}": {}})
    assert isinstance(_gguf_module(config), torch.nn.Linear)
