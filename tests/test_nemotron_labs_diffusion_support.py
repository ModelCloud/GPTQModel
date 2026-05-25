# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from transformers import AutoModel

from gptqmodel.models import auto
from gptqmodel.models.definitions.nemotron_labs_diffusion import NemotronLabsDiffusionQModel


def test_nemotron_labs_diffusion_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="nemotron_labs_diffusion")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/nemotron_labs_diffusion") is NemotronLabsDiffusionQModel


def test_nemotron_labs_diffusion_definition_matches_remote_code_layout():
    layer_modules = NemotronLabsDiffusionQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert NemotronLabsDiffusionQModel.require_trust_remote_code is True
    assert NemotronLabsDiffusionQModel.loader is AutoModel
    assert NemotronLabsDiffusionQModel.lm_head == "diffusion_head"
    assert NemotronLabsDiffusionQModel.pre_lm_head_norm_module == "encoder.norm"
    assert NemotronLabsDiffusionQModel.awq_scale_optimize_shape_dependent_modules == ["self_attn.o_proj"]
    assert NemotronLabsDiffusionQModel.extract_layers_node() == ["encoder.layers"]
    assert flat_modules == {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
