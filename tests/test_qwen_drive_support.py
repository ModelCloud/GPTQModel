# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import builtins
import json

import pytest
import torch
from eager_ssm_kernels import eager_torch_ssm_kernels

from gptqmodel import GPTQModel
from gptqmodel.models import auto
from gptqmodel.models.definitions.qwen_drive import QwenDriveQModel
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.ports.qwen_drive import QwenDriveConfig, QwenDriveForConditionalGeneration
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils import hf
from tests.test_qwen_drive_local_port import media_inputs, tiny_config, write_checkpoint


@pytest.fixture(autouse=True)
def _pin_eager_torch_ssm_kernels():
    """Run this module's CPU forwards on the Torch SSM/DeltaNet reference.

    ``transformers`` binds ``causal_conv1d``/``mamba_ssm``/``fla`` at import time
    when the distribution is installed, and those implementations are CUDA-only.
    The invariants here feed CPU tensors, so pin the reference kept on
    ``__wrapped__`` (upstream precedent #5537) instead of failing on the device.
    """

    with eager_torch_ssm_kernels():
        yield


@pytest.mark.parametrize("subfolder", [None, "selected"])
@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("bypass", [None, "0", "1"])
def test_qwen_drive_public_selection_precedes_optional_registration(tmp_path, monkeypatch, subfolder, trust, bypass):
    source = tmp_path / "checkpoint"
    selected = source / subfolder if subfolder else source
    expected, marker = write_checkpoint(selected)
    if subfolder:
        # A root-level decoy must not override explicitly selected metadata.
        (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    if bypass is None:
        monkeypatch.delenv("GPTQ_BYPASS_SAFETY_CHECKS", raising=False)
    else:
        monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", bypass)

    def deny_registration(*args, **kwargs):
        raise AssertionError("owned Qwen-Drive must precede optional registration")

    original_import = builtins.__import__

    def deny_official(name, *args, **kwargs):
        if name == "qwen_drive" or name.startswith("qwen_drive."):
            raise AssertionError("owned Qwen-Drive must not import the official package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(hf, "ensure_qwen_drive_registered", deny_registration)
    monkeypatch.setattr(builtins, "__import__", deny_official)
    selection = {"subfolder": subfolder} if subfolder else {}
    assert type(hf.load_model_config(source, trust_remote_code=trust, **selection)) is QwenDriveConfig
    assert auto.check_and_get_model_definition(source, trust_remote_code=trust, **selection) is QwenDriveQModel
    loaded = GPTQModel.load(str(source), trust_remote_code=trust, device="cpu", dtype=torch.float32, **selection)
    assert type(loaded.model) is QwenDriveForConditionalGeneration
    assert loaded.trust_remote_code is False
    assert not hasattr(loaded.model, "planning_expert")
    assert not hasattr(loaded.model, "perception_model")
    assert all(key.startswith("vlm.") for key in loaded.model.state_dict())
    inputs = media_inputs()
    with torch.inference_mode():
        torch.testing.assert_close(loaded(**inputs, use_cache=False).logits,
                                   expected.vlm(**inputs, use_cache=False).logits, rtol=0, atol=0)
        assert loaded.generate(**inputs, max_new_tokens=1, do_sample=False).shape == (1, 6)
    assert not marker.exists()


def test_qwen_drive_definition_contract_and_module_tree():
    assert isinstance(QwenDriveQModel.loader, LocalModelLoader)
    assert QwenDriveQModel.require_trust_remote_code is False
    assert QwenDriveQModel.require_load_processor is False
    assert QwenDriveQModel.lm_head == "vlm.lm_head"
    assert QwenDriveQModel.pre_lm_head_norm_module == "vlm.model.language_model.norm"
    assert QwenDriveQModel.rotary_embedding == "vlm.model.language_model.rotary_emb"
    assert QwenDriveQModel.extract_layers_node() == ["vlm.model.language_model.layers"]

    blocks = QwenDriveQModel.simple_layer_modules(model_config=tiny_config(), quantize_config=QuantizeConfig())
    flat = {name for block in blocks for name in block}
    assert {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    } <= flat
    assert {
        "self_attn.q_norm", "self_attn.k_norm", "linear_attn", "linear_attn.norm", "linear_attn.conv1d",
        "linear_attn.in_proj_a", "linear_attn.in_proj_b",
    }.isdisjoint(flat)
