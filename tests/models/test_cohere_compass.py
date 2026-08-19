# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os.path
from pathlib import Path
from types import SimpleNamespace

import torch
from model_test import ModelTest
from PIL import Image
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.cohere_compass import CohereCompassQModel
from gptqmodel.quantization.awq.quantize.scale import apply_scale


MODEL_ID = "/monster/data/model/North-Micro-Vision-Instruct"


def test_cohere_compass_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="cohere_compass")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/north-micro-vision") is CohereCompassQModel


def test_cohere_compass_module_tree_matches_parallel_residual_decoder():
    layer_modules = CohereCompassQModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    assert layer_modules == [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],
    ]
    assert CohereCompassQModel.require_load_processor is True
    assert CohereCompassQModel.require_trust_remote_code is False
    assert CohereCompassQModel.require_pkgs == ["transformers>=5.16.0.dev0"]
    assert CohereCompassQModel.pre_lm_head_norm_module == "model.language_model.norm"
    assert CohereCompassQModel.rotary_embedding == "model.language_model.rotary_emb"
    assert CohereCompassQModel.awq_preserve_explicit_position_embeddings is True
    assert CohereCompassQModel.extract_layers_node() == ["model.language_model.layers"]


def test_cohere_compass_base_modules_include_visual_and_language_roots():
    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 4)
            self.layers = nn.ModuleList([nn.Identity()])
            self.norm = nn.LayerNorm(4)
            self.rotary_emb = nn.Identity()

    class _CoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Identity()
            self.language_model = _LanguageModel()

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _CoreModel()

    base_modules = set(CohereCompassQModel.get_base_modules(_Wrapper()))

    assert base_modules == {
        "model.visual",
        "model.language_model.embed_tokens",
        "model.language_model.norm",
        "model.language_model.rotary_emb",
    }


def test_cohere_compass_pre_quantize_hook_materializes_multimodal_roots():
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.embed_tokens = nn.Embedding(4, 4)
    model.model.language_model.norm = nn.LayerNorm(4)
    model.model.language_model.rotary_emb = nn.Identity()
    model.model.visual = nn.Linear(4, 4)

    qmodel = object.__new__(CohereCompassQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    materialized = []

    def shell_module_materialize(module, device):
        materialized.append((module, device))
        return module

    qmodel.shell_module_materialize = shell_module_materialize
    qmodel.pre_quantize_generate_hook_start()

    assert materialized == [
        (model.model.language_model.embed_tokens, torch.device("cpu")),
        (model.model.language_model.norm, torch.device("cpu")),
        (model.model.language_model.rotary_emb, torch.device("cpu")),
        (model.model.visual, torch.device("cpu")),
    ]
    qmodel._stop_outer_input_capture()


def test_cohere_compass_replay_kwargs_follow_layer_attention_type(monkeypatch):
    model = nn.Module()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            rope_parameters={
                "sliding_attention": {"rope_type": "default"},
                "full_attention": None,
            },
        )
    )
    qmodel = object.__new__(CohereCompassQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model

    sliding_layer = SimpleNamespace(self_attn=SimpleNamespace(layer_idx=0))
    full_layer = SimpleNamespace(self_attn=SimpleNamespace(layer_idx=1))
    captured_position_embeddings = (torch.randn(1, 4, 2), torch.randn(1, 4, 2))

    sliding_kwargs = qmodel.prepare_layer_replay_kwargs(
        layer=sliding_layer,
        layer_input=[torch.randn(1, 4, 8)],
        additional_inputs={"position_embeddings": captured_position_embeddings},
        target_device=torch.device("cpu"),
    )
    full_kwargs = qmodel.prepare_layer_replay_kwargs(
        layer=full_layer,
        layer_input=[torch.randn(1, 4, 8)],
        additional_inputs={"position_embeddings": captured_position_embeddings},
        target_device=torch.device("cpu"),
    )

    assert sliding_kwargs["position_embeddings"] is captured_position_embeddings
    assert full_kwargs["position_embeddings"] is None

    monkeypatch.setattr(
        BaseQModel,
        "awq_get_modules_for_scaling",
        lambda self, module, input_feat, module_kwargs: module_kwargs,
    )
    awq_kwargs = qmodel.awq_get_modules_for_scaling(
        full_layer,
        input_feat={"self_attn.q_proj": torch.randn(1, 4, 8)},
        module_kwargs={
            "position_embeddings": captured_position_embeddings,
            "_awq_feature_kwargs": {
                "self_attn.q_proj": {"position_embeddings": captured_position_embeddings},
                "mlp.gate_proj": {},
            },
        },
    )

    assert awq_kwargs["position_embeddings"] is None
    assert awq_kwargs["_awq_feature_kwargs"]["self_attn.q_proj"]["position_embeddings"] is None
    assert awq_kwargs["_awq_feature_kwargs"]["mlp.gate_proj"]["position_embeddings"] is None


def test_cohere_compass_isolated_replay_matches_outer_text_loop():
    from transformers.models.cohere_compass.configuration_cohere_compass import CohereCompassTextConfig
    from transformers.models.cohere_compass.modeling_cohere_compass import CohereCompassTextModel

    config = CohereCompassTextConfig(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        sliding_window=2,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": None,
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 50_000,
                "mrope_interleaved": True,
                "mrope_section": [1, 1, 1],
            },
        },
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    language_model = CohereCompassTextModel(config).eval()

    wrapper = nn.Module()
    wrapper.config = SimpleNamespace(text_config=config)
    wrapper.model = nn.Module()
    wrapper.model.language_model = language_model
    qmodel = object.__new__(CohereCompassQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = wrapper

    torch.manual_seed(0)
    inputs_embeds = torch.randn(1, 5, config.hidden_size)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    position_ids = torch.arange(5, dtype=torch.long).view(1, 1, 5).expand(4, 1, -1)
    visual_pos_masks = torch.tensor([[False, True, False, True, False]])
    deepstack_visual_embeds = [torch.randn(2, config.hidden_size) for _ in range(3)]

    native_layer_inputs = []
    native_layer_kwargs = []
    handles = []
    for layer in language_model.layers:
        def capture_layer_inputs(module, args, kwargs, *, _layer=layer):
            del module, _layer
            native_layer_inputs.append(args[0].detach().clone())
            native_layer_kwargs.append(dict(kwargs))

        handles.append(layer.register_forward_pre_hook(capture_layer_inputs, with_kwargs=True))

    qmodel._start_outer_input_capture()
    try:
        with torch.no_grad():
            native_output = language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            ).last_hidden_state
        captured_outer_kwargs = qmodel.capture_first_layer_input_kwargs(
            args=(),
            kwargs={},
            batch_device=torch.device("cpu"),
            layer_input_kwargs={},
        )
    finally:
        qmodel._stop_outer_input_capture()
        for handle in handles:
            handle.remove()

    replay_cache = dict(captured_outer_kwargs)
    replay_cache["attention_mask"] = native_layer_kwargs[0]["attention_mask"]
    replay_cache["position_embeddings"] = native_layer_kwargs[0]["position_embeddings"]
    replay_cache["use_cache"] = False

    hidden_states = native_layer_inputs[0].clone()
    with torch.no_grad():
        for layer_index, layer in enumerate(language_model.layers):
            replay_kwargs = qmodel.prepare_layer_replay_kwargs(
                layer=layer,
                layer_input=[hidden_states],
                additional_inputs=dict(replay_cache),
                target_device=torch.device("cpu"),
            )

            native_position_embeddings = native_layer_kwargs[layer_index]["position_embeddings"]
            if native_position_embeddings is None:
                assert replay_kwargs["position_embeddings"] is None
            else:
                torch.testing.assert_close(
                    replay_kwargs["position_embeddings"][0],
                    native_position_embeddings[0],
                )
                torch.testing.assert_close(
                    replay_kwargs["position_embeddings"][1],
                    native_position_embeddings[1],
                )
            torch.testing.assert_close(
                replay_kwargs["attention_mask"],
                native_layer_kwargs[layer_index]["attention_mask"],
            )

            hidden_states = layer(hidden_states, **replay_kwargs)
            qmodel.update_layer_replay_kwargs_from_output(
                layer=layer,
                layer_output=hidden_states,
                layer_input_kwargs=replay_cache,
                target_device=torch.device("cpu"),
            )
            if layer_index + 1 < len(language_model.layers):
                torch.testing.assert_close(hidden_states, native_layer_inputs[layer_index + 1])

        torch.testing.assert_close(language_model.norm(hidden_states), native_output)


def test_cohere_compass_layer_norm_supports_awq_scale_equivalence():
    from transformers.models.cohere_compass.modeling_cohere_compass import CohereCompassLayerNorm

    module = nn.Module()
    module.input_layernorm = CohereCompassLayerNorm(4)
    module.self_attn = nn.Module()
    module.self_attn.q_proj = nn.Linear(4, 4, bias=False)
    hidden_states = torch.randn(2, 3, 4)
    scales = torch.tensor([0.5, 0.75, 1.25, 2.0])

    reference = module.self_attn.q_proj(module.input_layernorm(hidden_states))
    apply_scale(
        module,
        [("input_layernorm", ["self_attn.q_proj"], scales, 0.0)],
    )
    scaled = module.self_attn.q_proj(module.input_layernorm(hidden_states))

    torch.testing.assert_close(scaled, reference, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(module.input_layernorm.weight).all()
    assert torch.isfinite(module.self_attn.q_proj.weight).all()


def test_cohere_compass_native_shell_and_processor_match_definition():
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
    with init_empty_weights(include_buffers=True):
        shell = AutoModelForImageTextToText.from_config(config, trust_remote_code=False)

    layer = shell.model.language_model.layers[0]

    assert config.model_type == "cohere_compass"
    assert auto.check_and_get_model_definition(MODEL_ID) is CohereCompassQModel
    assert hasattr(shell.model, "visual")
    assert hasattr(shell.model, "language_model")
    assert hasattr(layer, "input_layernorm")
    assert hasattr(layer.self_attn, "q_proj")
    assert hasattr(layer.self_attn, "k_proj")
    assert hasattr(layer.self_attn, "v_proj")
    assert hasattr(layer.self_attn, "o_proj")
    assert hasattr(layer.mlp, "gate_proj")
    assert hasattr(layer.mlp, "up_proj")
    assert hasattr(layer.mlp, "down_proj")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=False)
    image = Image.open(Path(__file__).parent / "ovis" / "10016.jpg").convert("RGB")
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What do you see?"},
            ],
        }
    ]
    inputs = CohereCompassQModel.prepare_inputs_for_conversations(processor, conversations)

    assert set(inputs) == {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
    }
    assert inputs.input_ids.shape[0] == 1
    assert inputs.pixel_values.ndim == 2
    assert inputs.image_grid_thw.shape == (1, 3)


class TestCohereCompass(ModelTest):
    NATIVE_MODEL_ID = MODEL_ID
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    OFFLOAD_TO_DISK = False
    EVAL_BATCH_SIZE = 8
    MODEL_COMPAT_FAST_LAYER_COUNT = 4
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    EVAL_TASKS_SLOW = {
        "gsm8k_platinum_cot": {
            "chat_template": True,
            "evalution_use_model_path": True,
            "evalution_batch_size": "auto",
            "evalution_model_args": {
                "dtype": "bfloat16",
                "attn_implementation": "paged|flash_attention_2",
                "device": "cuda:0",
            },
            "evalution_suite_kwargs": {
                "batch_size": 32,
                "max_new_tokens": 256,
                "stream": True,
            },
            "acc,num": {
                "value": 0.47229114971050457,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
        "arc_challenge": {
            "chat_template": True,
            "acc": {
                "value": 0.3242320819112628,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
            "acc_norm": {
                "value": 0.3515358361774744,
                "floor_pct": 0.04,
                "ceil_pct": 1.0,
            },
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)

    def test_cohere_compass(self):
        with self.model_compat_test_context():
            model, _tokenizer, processor = self.quantModel(
                self.NATIVE_MODEL_ID,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                dtype=self.TORCH_DTYPE,
                batch_size=1,
            )

        try:
            image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovis/10016.jpg")
            image = Image.open(image_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "What do you see?"},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            self.assertTrue(response.strip())
            self.check_kernel(model, self.KERNEL_INFERENCE)
            task_results = self._post_quant_eval_records.get(self._current_load_backend())
            self.assertIsNotNone(task_results)
            expected_metrics = {
                "gsm8k_platinum_cot": ("acc,num",),
                "arc_challenge": ("acc", "acc_norm"),
            }
            for task_name, metric_names in expected_metrics.items():
                self.assertIn(task_name, task_results)
                for metric_name in metric_names:
                    self.assertIsNotNone(
                        self._resolve_metric_key(metric_name, task_results[task_name]),
                        f"Metric `{metric_name}` missing from task `{task_name}`",
                    )
            self.check_results(task_results)
        finally:
            self._cleanup_quantized_model(model, enabled=self.DELETE_QUANTIZED_MODEL)
