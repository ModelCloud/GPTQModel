# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.qwen3_5_text import Qwen3_5TextQModel
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq_axis_policy import (
    apply_qvq_transform_axis_overrides,
    qvq_shared_input_seed,
    qvq_transform_axis_overrides_from_config,
    resolve_qvq_transform_axes,
    set_qvq_transform_axis_metadata,
)


@pytest.mark.parametrize("definition", (Qwen3_5QModel, Qwen3_5TextQModel))
def test_qwen38_declares_all_shared_p32_groups_and_folded_mlp_axes(definition):
    assert definition.qvq_grouped_p32_candidates == {
        "qkv": (
            ("q_proj", "k_proj", "v_proj"),
            ("in_proj_qkv", "in_proj_z"),
        ),
        "gate_up": (("gate_proj", "up_proj"),),
    }
    assert definition.qvq_transform_axis_overrides == {
        "self_attn.v_proj": (True, False),
        "mlp.gate_proj": (True, False),
        "mlp.up_proj": (True, False),
        "mlp.down_proj": (False, True),
    }


def test_qwen38_shared_input_seeds_are_layer_local_and_role_agnostic():
    groups = Qwen3_5TextQModel.qvq_grouped_p32_candidates
    prefix = "model.layers.7"
    qkv = [
        qvq_shared_input_seed(f"{prefix}.self_attn.{name}", groups)
        for name in ("q_proj", "k_proj", "v_proj")
    ]
    linear = [
        qvq_shared_input_seed(f"{prefix}.linear_attn.{name}", groups)
        for name in ("in_proj_qkv", "in_proj_z")
    ]
    mlp = [
        qvq_shared_input_seed(f"{prefix}.mlp.{name}", groups)
        for name in ("gate_proj", "up_proj")
    ]
    assert len(set(qkv)) == len(set(linear)) == len(set(mlp)) == 1
    assert len({qkv[0], linear[0], mlp[0]}) == 3
    assert qvq_shared_input_seed(f"{prefix}.mlp.down_proj", groups) is None
    assert qkv[0] != qvq_shared_input_seed(
        "model.layers.8.self_attn.q_proj", groups
    )


def test_qvq_folded_axis_metadata_roundtrip_and_loaded_module_application():
    overrides = Qwen3_5TextQModel.qvq_transform_axis_overrides
    config = SimpleNamespace(meta={})
    payload = set_qvq_transform_axis_metadata(config, overrides)
    assert payload["schema"] == "qvq.transform-axes.v1"
    assert qvq_transform_axis_overrides_from_config(config) == overrides

    model = torch.nn.Module()
    model.mlp = torch.nn.Module()
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(
            model.mlp,
            name,
            QVQLinear(bits=2, in_features=16, out_features=16),
        )
    assert apply_qvq_transform_axis_overrides(model, overrides) == 3
    assert resolve_qvq_transform_axes("model.layers.0.mlp.gate_proj", overrides) == (
        True,
        False,
    )
    assert model.mlp.gate_proj.output_hadamard is False
    assert model.mlp.up_proj.output_hadamard is False
    assert model.mlp.down_proj.input_hadamard is False
    assert model.mlp.down_proj.output_hadamard is True


def test_qvq_axis_metadata_rejects_conflicting_checkpoint_contract():
    config = SimpleNamespace(meta={})
    set_qvq_transform_axis_metadata(config, {"mlp.gate_proj": (True, False)})
    with pytest.raises(ValueError, match="conflicting"):
        set_qvq_transform_axis_metadata(
            config, {"mlp.gate_proj": (False, False)}
        )
