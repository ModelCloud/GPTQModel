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
  constructor allocates ``mamba.A_log`` in fp32 from an explicit fp32 ``A``.
  The package-owned port declares ``A_log`` in
  ``_keep_in_fp32_modules_strict``, so transformers keeps it fp32 and the guard
  exempts it through that declaration, without GPT-QModel hooking
  ``PreTrainedModel.post_init``.

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

        def shell_direct_meta_materialize(
            self, *, target_submodule, device=None, module_path=None
        ):
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
    _keep_in_fp32_modules_strict = ["construction_fp32"]

    def __init__(self, config):
        super().__init__(config)
        self.decoder_weight = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        self.construction_fp32 = nn.Parameter(torch.randn(16, dtype=torch.float32))
        self.post_init()


def _construct(model_cls):
    return model_cls(_GuardConfig())


def test_hymba_checkpoint_flex_hint_loads_with_supported_attention(tmp_path):
    """#5090: the original checkpoint serializes a Transformers-4 `flex` hint.

    Transformers 5 validates `_attn_implementation` before the remote
    `HymbaModel.__init__` copied `attn_implementation_new` over it, so the
    checkpoint failed to load. The package-owned `HymbaConfig` maps the hint to
    its portable SDPA implementation instead of relying on a config shim.
    """

    from gptqmodel.models.ports.hymba.configuration_hymba import HymbaConfig
    from tests.test_hymba_local_port import tiny_config

    config = tiny_config(attn_implementation_new="flex")
    config.save_pretrained(tmp_path)
    reloaded = HymbaConfig.from_pretrained(tmp_path)
    assert reloaded._attn_implementation == "sdpa"


def test_dtype_guard_accepts_fp32_widened_from_bf16():
    assert_loaded_model_state(
        _construct(_GuardModel),
        api_name="test",
        expected_dtype=torch.bfloat16,
        require_uniform_dtype=True,
    )


def test_dtype_guard_still_rejects_corrupted_fp32():
    model = _construct(_GuardModel)
    # The constructor made this bf16; a broken load/cast left it fp32.
    model.decoder_weight.data = model.decoder_weight.data.float()

    with pytest.raises(RuntimeError, match="floating parameters are not all torch.bfloat16"):
        assert_loaded_model_state(
            model,
            api_name="test",
            expected_dtype=torch.bfloat16,
            require_uniform_dtype=True,
        )


class _UndeclaredFp32Model(PreTrainedModel):
    """A constructor-made fp32 parameter that the model never declared."""

    config_class = _GuardConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder_weight = nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
        self.weight = nn.Parameter(torch.ones(8, dtype=torch.float32))
        self.post_init()


def test_dtype_guard_rejects_an_undeclared_fp32_parameter():
    """Only a keep-in-fp32 declaration exempts an fp32 parameter."""

    model = _construct(_UndeclaredFp32Model)

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
    _keep_in_fp32_modules_strict = ["A_log"]

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


def test_hymba_port_pins_a_log_to_fp32():
    from gptqmodel.models.ports.hymba.modeling_hymba import HymbaPreTrainedModel

    assert "A_log" in HymbaPreTrainedModel._keep_in_fp32_modules_strict


def test_load_never_wraps_pretrained_model_post_init():
    original = PreTrainedModel.post_init
    _construct(_GuardModel)
    assert PreTrainedModel.post_init is original
    assert not hasattr(_construct(_GuardModel), "_gptqmodel_construction_parameter_dtypes")
