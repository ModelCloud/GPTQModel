# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hymba must quantize in half precision (#4888).

Two independent generic defects:

* A lazy offload shell materializes the base *modules* returned by
  ``get_base_modules()`` plus any definition-declared direct-meta modules. A
  bare ``nn.Parameter`` on a decoder-root prefix node (Hymba's
  ``model.memory_tokens``) was in neither set, so it stayed on ``meta`` and the
  capture forward raised ``Tensor on device cpu is not on the expected device
  meta!``.
* ``assert_loaded_model_state`` rejected a valid bf16 load because Hymba's
  remote constructor allocates ``mamba.A_log`` in fp32 from an explicit fp32
  ``A``. transformers loads the bf16 checkpoint value into that fp32 parameter
  without changing its dtype; the values are exact.

The tests use small synthetic fixtures: the real Hymba needs remote code with
``mamba_ssm``/``causal_conv1d``/``flash_attn``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig, PreTrainedModel

from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.models.definitions.hymba import HymbaQModel
from gptqmodel.utils.hf import assert_loaded_model_state


class _RootBareParameterModel(nn.Module):
    """Decoder root owning a bare parameter, like ``HymbaModel.memory_tokens``."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.model.final_layernorm = nn.LayerNorm(4)
        self.model.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.model.memory_tokens = nn.Parameter(torch.empty(4, device="meta"))


def test_root_level_bare_parameter_is_detected_as_direct_meta_module():
    model = _RootBareParameterModel()

    assert HymbaQModel.get_modules_with_direct_meta_tensors(model) == ["model"]


def test_stage_inputs_capture_materializes_the_root_bare_parameter():
    model = _RootBareParameterModel()
    calls = []

    class _FakeGPTQModel:
        def __init__(self, wrapped):
            self.model = wrapped

        def get_modules_with_direct_meta_tensors(self, wrapped):
            return HymbaQModel.get_modules_with_direct_meta_tensors(wrapped)

        def shell_direct_meta_materialize(self, *, target_submodule, device=None):
            calls.append((target_submodule, device))
            target_submodule.register_parameter(
                "memory_tokens",
                nn.Parameter(torch.zeros(4, device=device or "cpu")),
            )

    stage = StageInputsCapture(SimpleNamespace(gptq_model=_FakeGPTQModel(model)))

    stage._materialize_modules_with_direct_meta_tensors(torch.device("cpu"))

    assert not model.model.memory_tokens.is_meta
    assert model.model.memory_tokens.device.type == "cpu"
    assert calls == [(model.model, torch.device("cpu"))]


class _GuardConfig(LlamaConfig):
    model_type = "guard_hymba_4888"


class _GuardModel(PreTrainedModel):
    """One parameter constructed bf16 and one the constructor allocates in fp32."""

    config_class = _GuardConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder_weight = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        self.construction_fp32 = nn.Parameter(torch.randn(16, dtype=torch.float32))
        self.post_init()


def test_dtype_guard_accepts_fp32_widened_from_bf16():
    assert_loaded_model_state(
        _GuardModel(_GuardConfig()),
        api_name="test",
        expected_dtype=torch.bfloat16,
        require_uniform_dtype=True,
    )


def test_dtype_guard_still_rejects_corrupted_fp32():
    model = _GuardModel(_GuardConfig())
    # The constructor made this bf16; a broken load/cast left it fp32.
    model.decoder_weight.data = model.decoder_weight.data.float()

    with pytest.raises(RuntimeError, match="floating parameters are not all torch.bfloat16"):
        assert_loaded_model_state(
            model,
            api_name="test",
            expected_dtype=torch.bfloat16,
            require_uniform_dtype=True,
        )


class _AllFp32ConstructedModel(PreTrainedModel):
    """Constructed outside transformers' dtype context: the record is uninformative."""

    config_class = _GuardConfig

    def __init__(self, config):
        super().__init__(config)
        self.weight = nn.Parameter(torch.ones(8, dtype=torch.float32))
        self.post_init()


def test_dtype_guard_ignores_an_all_fp32_construction_record():
    """A record taken without a dtype context must not exempt a wrong dtype."""

    model = _AllFp32ConstructedModel(_GuardConfig())

    assert getattr(model, "_gptqmodel_construction_parameter_dtypes", None) is None
    with pytest.raises(RuntimeError, match="floating parameters are not all torch.bfloat16"):
        assert_loaded_model_state(
            model,
            api_name="test",
            expected_dtype=torch.bfloat16,
            require_uniform_dtype=True,
        )


class _TinyConfig(LlamaConfig):
    model_type = "tiny_hymba_4888"


class _TinyHymbaLikeModel(PreTrainedModel):
    """A real ``from_pretrained`` load with one construction-time fp32 param."""

    config_class = _TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Mirrors ``HymbaMambaMixer.A_log``: built from an explicit fp32 tensor.
        self.A_log = nn.Parameter(torch.randn(4, dtype=torch.float32))
        self.post_init()


def test_real_bf16_load_keeps_the_construction_fp32_parameter(tmp_path):
    torch.manual_seed(0)
    config = _TinyConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
    )
    _TinyHymbaLikeModel(config).to(torch.bfloat16).save_pretrained(tmp_path)

    loaded = _TinyHymbaLikeModel.from_pretrained(tmp_path, dtype=torch.bfloat16)

    assert loaded.A_log.dtype is torch.float32
    assert loaded.linear.weight.dtype is torch.bfloat16
    assert_loaded_model_state(
        loaded,
        api_name="_TinyHymbaLikeModel.from_pretrained",
        expected_dtype=torch.bfloat16,
        require_uniform_dtype=True,
    )
