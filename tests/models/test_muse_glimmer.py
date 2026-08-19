# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path
from types import SimpleNamespace

import pytest
import torch
from model_test import ModelTest
from PIL import Image
from torch import nn

from gptqmodel import BACKEND
from gptqmodel.models import auto
from gptqmodel.models.definitions import muse_glimmer
from gptqmodel.models.definitions.muse_glimmer import MuseGlimmerQModel
from gptqmodel.utils.model import MODALITY


def test_muse_glimmer_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="muse_glimmer")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda path: None)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/muse-glimmer") is MuseGlimmerQModel


def test_muse_glimmer_module_tree_matches_text_decoder_order():
    layer_modules = MuseGlimmerQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    assert MuseGlimmerQModel.extract_layers_node() == ["model.language_model.layers"]
    assert MuseGlimmerQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert MuseGlimmerQModel.rotary_embedding == "model.language_model.rotary_emb"
    assert MuseGlimmerQModel.require_load_processor is True
    assert MuseGlimmerQModel.support_batch_quantize is False
    assert MuseGlimmerQModel.modality == [MODALITY.TEXT, MODALITY.IMAGE_TO_TEXT]


@pytest.mark.parametrize(
    ("layer_index", "is_local_attention", "expected_mask_factory", "expect_position_embeddings"),
    [
        (0, True, "sliding", True),
        (3, False, "causal", False),
    ],
)
def test_muse_glimmer_replay_rebuilds_layer_specific_mask_and_rope(
    monkeypatch,
    layer_index,
    is_local_attention,
    expected_mask_factory,
    expect_position_embeddings,
):
    calls = []

    def fake_mask_factory(name):
        def create_mask(**kwargs):
            calls.append((name, kwargs))
            return f"{name}-mask"

        return create_mask

    monkeypatch.setattr(muse_glimmer, "create_causal_mask", fake_mask_factory("causal"))
    monkeypatch.setattr(muse_glimmer, "create_sliding_window_causal_mask", fake_mask_factory("sliding"))

    config = SimpleNamespace(layer_rope_theta=[500000.0, 500000.0, 500000.0, 0])
    self_attention = SimpleNamespace(
        config=config,
        is_local_attention=is_local_attention,
        layer_idx=layer_index,
    )
    layer = SimpleNamespace(config=config, self_attn=self_attention)
    hidden_states = torch.zeros(1, 4, 8)
    position_embeddings = (torch.ones(1, 4, 8), torch.zeros(1, 4, 8))
    additional_inputs = {
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_embeddings": position_embeddings,
    }

    qmodel = object.__new__(MuseGlimmerQModel)
    nn.Module.__init__(qmodel)
    result = qmodel.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs=additional_inputs,
        target_device=torch.device("cpu"),
    )

    assert result["attention_mask"] == f"{expected_mask_factory}-mask"
    assert result["position_ids"].tolist() == [[0, 1, 2, 3]]
    if expect_position_embeddings:
        assert result["position_embeddings"] is position_embeddings
    else:
        assert result["position_embeddings"] is None

    assert len(calls) == 1
    mask_name, mask_kwargs = calls[0]
    assert mask_name == expected_mask_factory
    assert mask_kwargs["attention_mask"].dtype == torch.bool
    assert mask_kwargs["layer_idx"] == layer_index


def test_muse_glimmer_replay_matches_dense_text_forward():
    configuration = pytest.importorskip("transformers.models.muse_glimmer.configuration_muse_glimmer")
    modeling = pytest.importorskip("transformers.models.muse_glimmer.modeling_muse_glimmer")

    torch.manual_seed(0)
    config = configuration.MuseGlimmerTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
        sliding_window=2,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = modeling.MuseGlimmerTextModel(config).eval()

    input_ids = torch.tensor([[1, 3, 4, 5, 6]])
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    dense_output = model(input_ids=input_ids, use_cache=False).last_hidden_state

    hidden_states = model.embed_tokens(input_ids)
    position_embeddings = model.rotary_emb(hidden_states, position_ids)
    qmodel = object.__new__(MuseGlimmerQModel)
    nn.Module.__init__(qmodel)

    for layer in model.layers:
        replay_kwargs = qmodel.prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=[hidden_states],
            additional_inputs={
                "attention_mask": None,
                "position_ids": position_ids,
                "position_embeddings": position_embeddings,
            },
            target_device=torch.device("cpu"),
        )
        hidden_states = layer(hidden_states, **replay_kwargs)

    replay_output = model.norm(hidden_states)
    torch.testing.assert_close(replay_output, dense_output, rtol=0, atol=0)


class TestMuseGlimmer(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Muse-Glimmer-30B"  # meta-models/Muse-Glimmer-30B
    USE_FLASH_ATTN = False
    TRUST_REMOTE_CODE = False
    DELETE_QUANTIZED_MODEL = False
    EVAL_BATCH_SIZE = 1
    LOAD_BACKEND = BACKEND.AUTO
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"

    def test_muse_glimmer(self):
        with self.model_compat_test_context():
            model, _tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
                need_eval=False,
                call_perform_post_quant_validation=False,
            )

        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.open(image_path)},
                    {"type": "text", "text": "What is in this image?"},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        output = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"Output:\n{output}")

        self.assertIn("snow", output.lower())
