# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from model_test import ModelTest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from transformers import PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig, WeightOnlyConfig
from gptqmodel.models import auto
from gptqmodel.models.definitions.solar_open import SolarOpenQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig, VramStrategy
from gptqmodel.utils.hf import build_shell_model
from gptqmodel.utils.structure import LazyTurtle


MODEL_PATH = Path("/monster/data/model/Solar-Open-100B")
CALIBRATION_TEXTS = [
    "solar open tiny calibration sample with enough tokens to exercise every routed expert",
    "another synthetic calibration prompt for the shared expert and attention projections",
]


def _tiny_model():
    transformers = pytest.importorskip("transformers")
    SolarOpenConfig = getattr(transformers, "SolarOpenConfig", None)
    SolarOpenForCausalLM = getattr(transformers, "SolarOpenForCausalLM", None)
    if SolarOpenConfig is None or SolarOpenForCausalLM is None:
        pytest.skip("Installed Transformers does not provide Solar Open")

    config = SolarOpenConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        moe_intermediate_size=8,
        n_routed_experts=3,
        n_shared_experts=1,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    config._experts_implementation = "eager"
    return SolarOpenForCausalLM(config).eval()


def _save_tiny_tokenizer(model_dir: Path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        CALIBRATION_TEXTS,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)
    return fast_tokenizer


def test_solar_open_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="solar_open")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/solar-open") is SolarOpenQModel


@pytest.mark.skipif(not MODEL_PATH.is_dir(), reason="Solar Open checkpoint is unavailable")
def test_solar_open_local_checkpoint_selects_definition():
    assert auto.check_and_get_model_definition(str(MODEL_PATH), trust_remote_code=False) is SolarOpenQModel


def test_solar_open_module_tree_covers_gqa_and_moe_paths():
    config = SimpleNamespace(n_routed_experts=3)
    quantize_config = SimpleNamespace(dynamic=None)
    layer_modules = SolarOpenQModel.simple_layer_modules(config, quantize_config)
    flat_modules = {name for block in layer_modules for name in block}
    capture_modules = {
        name
        for block in SolarOpenQModel.full_layer_modules(
            config,
            include_capture_only=True,
        )
        for name in block
    }

    assert SolarOpenQModel.dynamic_expert_index == "n_routed_experts"
    assert SolarOpenQModel.extract_layers_node() == ["model.layers"]
    assert "self_attn.q_proj" in flat_modules
    assert "self_attn.k_proj" in flat_modules
    assert "self_attn.v_proj" in flat_modules
    assert "self_attn.o_proj" in flat_modules
    assert "mlp.experts.0.gate_proj" in flat_modules
    assert "mlp.experts.1.up_proj" in flat_modules
    assert "mlp.experts.2.down_proj" in flat_modules
    assert "mlp.shared_experts.gate_proj" in flat_modules
    assert "mlp.shared_experts.up_proj" in flat_modules
    assert "mlp.shared_experts.down_proj" in flat_modules
    assert "mlp.gate" not in flat_modules
    assert "input_layernorm:!" in capture_modules
    assert "post_attention_layernorm:!" in capture_modules
    assert "mlp.gate:!" in capture_modules


def test_solar_open_defuser_expands_routed_experts_without_changing_forward():
    from defuser.model_registry import MODEL_CONFIG

    assert "solar_open" in MODEL_CONFIG

    torch.manual_seed(0)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 7, 8, 2]])
    packed_experts = model.model.layers[0].mlp.experts

    assert hasattr(packed_experts, "gate_up_proj")
    with torch.inference_mode():
        expected = model(input_ids=input_ids, use_cache=False).logits

    assert _convert_model_with_defuser(SolarOpenQModel, model, cleanup_original=False) is True

    experts = model.model.layers[0].mlp.experts
    assert not hasattr(experts, "gate_up_proj")
    assert isinstance(experts[0].gate_proj, nn.Linear)
    assert isinstance(experts[0].up_proj, nn.Linear)
    assert isinstance(experts[0].down_proj, nn.Linear)
    with torch.inference_mode():
        actual = model(input_ids=input_ids, use_cache=False).logits

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_solar_open_lazy_turtle_materializes_defused_expert_weights(tmp_path):
    source = _tiny_model()
    assert _convert_model_with_defuser(SolarOpenQModel, source, cleanup_original=False) is True
    source_expert = source.model.layers[0].mlp.experts[0]
    expected_gate = source_expert.gate_proj.weight.detach().clone()
    expected_up = source_expert.up_proj.weight.detach().clone()
    expected_down = source_expert.down_proj.weight.detach().clone()
    source.save_pretrained(tmp_path, safe_serialization=True)

    shell = build_shell_model(
        SolarOpenQModel.loader,
        config=copy.deepcopy(source.config),
        trust_remote_code=False,
        device_map={"": "cpu"},
        _fast_init=True,
    )
    assert _convert_model_with_defuser(SolarOpenQModel, shell, cleanup_original=False) is True
    turtle = LazyTurtle(
        model_local_path=str(tmp_path),
        config=shell.config,
        model_init_kwargs={},
        module_tree=SolarOpenQModel.module_tree,
        hf_conversion_map_reversed=SolarOpenQModel.resolve_hf_conversion_map_reversed(shell),
        target_model=shell,
    )

    expert = shell.model.layers[0].mlp.experts[0]
    turtle.materialize_submodule(
        target_model=shell,
        target_submodule=expert,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(expert.gate_proj.weight, expected_gate)
    torch.testing.assert_close(expert.up_proj.weight, expected_up)
    torch.testing.assert_close(expert.down_proj.weight, expected_down)


@pytest.mark.cpu
@pytest.mark.slow
def test_solar_open_tiny_quantization_smoke(tmp_path):
    native_dir = tmp_path / "native"
    quantized_dir = tmp_path / "quantized"
    source = _tiny_model()
    source.save_pretrained(native_dir, safe_serialization=True)
    tokenizer = _save_tiny_tokenizer(native_dir)
    calibration = []
    for text in CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        calibration.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )

    model = GPTQModel.load(
        str(native_dir),
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=16,
            desc_act=False,
            device="cpu",
            moe=MoEConfig(routing=ExpertsRoutingOverride()),
        ),
        backend=BACKEND.TORCH,
        attn_implementation="eager",
    )
    model.quantize(
        calibration,
        batch_size=1,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )
    model.save(quantized_dir)

    quantized_model = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.TORCH,
        device="cpu",
    )
    modules = dict(quantized_model.named_modules())
    for layer_index in range(source.config.num_hidden_layers):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module_name = f"model.model.layers.{layer_index}.self_attn.{projection}"
            assert isinstance(modules[module_name], TorchLinear), module_name

        for expert_index in range(source.config.n_routed_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                module_name = f"model.model.layers.{layer_index}.mlp.experts.{expert_index}.{projection}"
                assert isinstance(modules[module_name], TorchLinear), module_name

        for projection in ("gate_proj", "up_proj", "down_proj"):
            module_name = f"model.model.layers.{layer_index}.mlp.shared_experts.{projection}"
            assert isinstance(modules[module_name], TorchLinear), module_name


class TestSolarOpen(ModelTest):
    NATIVE_MODEL_ID = str(MODEL_PATH)
    TRUST_REMOTE_CODE = False
    USE_FLASH_ATTN = False
    LOAD_BACKEND = BACKEND.AUTO
    EVAL_BATCH_SIZE = 16
    EVAL_SINGLE_GPU = False
    MOE_VRAM_STRATEGY = VramStrategy.BALANCED
    MODEL_COMPAT_FAST_LAYER_POSITION = "first"
    # Calibrate every routed expert instead of depending on native top-8 traffic.
    MOE_CONFIG = MoEConfig(routing=ExpertsRoutingOverride(num_experts_per_tok="all"))
    EVAL_TASKS_SLOW = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3618, "floor_pct": 0.04},
            "acc_norm": {"value": 00.3882, "floor_pct": 0.04},
        },
    }
    EVAL_TASKS_FAST = ModelTest.derive_fast_eval_tasks(EVAL_TASKS_SLOW)
    SAVE_PATH = "./temp/solar_open1-test"
    WEIGHT_ONLY = WeightOnlyConfig()

    def test_solar_open(self):
        self.quantize_and_evaluate()


__all__ = ["TestSolarOpen"]
