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

import pytest
import torch

from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH, GGUFConfig


NAME = "model.layers.0.mlp.down_proj"

# (requested alias, width) pairs where the requested type is NOT the default
# type for that width -- exactly the cases the bug destroyed.
NON_DEFAULT = [
    ("q4_k", 4),  # default for 4 is q4_0
    ("q4_1", 4),  # default for 4 is q4_0
    ("q2_k", 2),  # default for 2 is q2_0
    ("q5_0", 5),  # default for 5 is q5_k_m
]


def _gguf_module(alias, width):
    try:
        from gptqmodel.nn_modules.qlinear.gguf_cpp import GGUFCppKernel
    except Exception:  # pragma: no cover - depends on the ggml runtime
        pytest.skip("ggml runtime is unavailable")
    from gptqmodel.utils.model import create_quant_module

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.model.layers[0].mlp = torch.nn.Module()
    root.model.layers[0].mlp.down_proj = torch.nn.Linear(256, 256, bias=False)

    config = GGUFConfig(bits="q4_k", dynamic={NAME: {"bits": width, "format": alias}})
    create_quant_module(
        name=NAME,
        linear_cls=GGUFCppKernel,
        bits=config.runtime_bits,
        desc_act=False,
        dynamic=config.dynamic,
        group_size=-1,
        module=root,
        submodule=root.get_submodule(NAME),
        sym=True,
        device="cpu",
        lm_head_name="lm_head",
        pack_dtype=torch.int32,
        format=FORMAT.GGUF,
        register_buffers=False,
    )
    return root.get_submodule(NAME).gguf_tensor_qtype


def test_the_cases_really_are_non_default():
    """Guard the table above: a default-type case would not exercise the bug."""
    for alias, width in NON_DEFAULT:
        assert _GGUF_DEFAULT_BITS_ALIAS_BY_WIDTH[width] != alias


@pytest.mark.parametrize("alias,width", NON_DEFAULT)
def test_per_module_format_beats_the_default_type_for_that_width(alias, width):
    assert _gguf_module(alias, width) == alias.upper()
