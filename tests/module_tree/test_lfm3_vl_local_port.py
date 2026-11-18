# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent live-owner inventory for native LFM3-VL variants."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from test_lfm3_vl_local_port import FIXTURES, media_inputs, native_model, tiny_processor
from transformers import AutoModelForImageTextToText, Lfm3VlConfig

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.lfm3_vl import LFM3VLQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.utils.model import find_modules


@pytest.mark.parametrize(
    "kv_heads,bias,gate,sandwich",
    [(1, False, True, True), (2, True, True, False), (4, False, False, True), (2, True, False, False)],
)
def test_every_native_checkpoint_owner_has_an_explicit_plan(monkeypatch, kv_heads, bias, gate, sandwich):
    model = native_model(kv_heads=kv_heads, bias=bias, gate=gate, sandwich=sandwich)
    _convert_model_with_defuser(LFM3VLQModel, model, cleanup_original=False)
    with monkeypatch.context() as context:
        context.setattr(LFM3VLQModel, "require_load_processor", False)
        wrapper = LFM3VLQModel(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    assert MODEL_MAP["lfm3_vl"] is LFM3VLQModel
    assert LFM3VLQModel.extract_layers_node() == ["model.language_model.layers"]
    for index, layer in enumerate(model.model.language_model.layers):
        expected = {f"feed_forward.w{i}" for i in (1, 2, 3)}
        expected.update({"conv.in_proj", "conv.out_proj"} if index == 0 else
                        {f"self_attn.{p}_proj" for p in ("q", "k", "v", "o")})
        for name in layer.state_dict():
            owner = name.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, owner
            assert spec.quantize == (owner in expected), owner
            if not spec.quantize:
                assert "!" in spec.flags, owner
        actual = {name for name in find_modules(layer) if plan.find(name).quantize}
        assert actual == expected
        for owner in expected:
            spec = plan.find(owner)
            assert spec is not None and spec.quantize
        assert plan.find("self_attn.g_proj").quantize is False
        assert plan.find("conv.conv").quantize is False
        assert plan.find("self_attn.gate_proj") is None
        assert plan.find("self_attn.g_proj.undeclared") is None
        assert plan.find("conv.conv.undeclared") is None
    blocks = LFM3VLQModel.simple_layer_modules(model.config, SimpleNamespace(dynamic=None))
    assert sum(len(block) for block in blocks) == 9
    assert all(not block.startswith(("vision_tower", "multi_modal_projector"))
               for group in blocks for block in group)
    inputs = media_inputs(tiny_processor())
    with torch.inference_mode():
        reference = model(**inputs, use_cache=False).logits[:, -1]
        cache = model(**inputs, use_cache=True).past_key_values
        token = torch.tensor([[5]])
        ids = torch.cat((inputs.input_ids, token), dim=1)
        mask = torch.ones_like(ids)
        reference = model(**{**inputs, "input_ids": ids, "attention_mask": mask}, use_cache=False).logits[:, -1]
        continued = model(input_ids=token, attention_mask=mask,
                          past_key_values=cache, use_cache=True).logits[:, -1]
    assert torch.isfinite(reference).all()
    torch.testing.assert_close(continued, reference, rtol=3e-5, atol=3e-6)


# sha256 of json.dumps(sorted((name, shape))) over the 424 tensors of
# LiquidAI/LFM3-VL-2B@23387beda43c60310348e83e25f78f33c3140d3a safetensors
# headers, after native checkpoint-conversion renames (lm_head is tied).
PINNED_INVENTORY_SHA256 = "bbafb498fa2221d32817f7ace4dd030d736467d1c84c8ac07aa221f98a45fbab"


def test_pinned_checkpoint_names_and_shapes_match_native_meta_shell():
    config = Lfm3VlConfig.from_dict(json.loads((FIXTURES / "checkpoint_config.json").read_text()))
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    shapes = {name: list(value.shape) for name, value in model.state_dict().items() if name != "lm_head.weight"}
    assert len(shapes) == 424
    # Representative released names that native conversion renames.
    assert shapes["model.language_model.layers.2.self_attn.g_proj.weight"] == [16, 2048]  # ckpt: gate_proj
    assert shapes["model.vision_tower.patch_embed.weight"] == [1024, 1536]  # ckpt: patchifier
    assert shapes["model.vision_tower.layers.2.self_attn.q_norm.weight"] == [64]  # ckpt: q_layernorm
    digest = hashlib.sha256(json.dumps(sorted(shapes.items())).encode()).hexdigest()
    assert digest == PINNED_INVENTORY_SHA256
