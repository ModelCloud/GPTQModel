# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""RTN must treat a parameterless placeholder as a coverage miss.

Nemotron-H declares ``mixer.fc1_latent_proj``/``fc2_latent_proj`` for the
latent-MoE layout. With ``moe_latent_size`` unset transformers builds an
``nn.Identity`` there. ``find_modules`` correctly skips it, but the weight-only
prefix fallback wrapped it in a ``NamedModule`` and aborted every RTN run on
``next(module.parameters())`` (#5270). The calibrated looper already records
such a name as a coverage miss instead of resolving it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.utils.model import find_modules


def _looper(strict: bool = False) -> WeightOnlyLooper:
    """A looper holding only the state ``_resolve_named_module`` reads."""
    looper = object.__new__(WeightOnlyLooper)
    looper.gptq_model = SimpleNamespace(layer_modules_strict=strict)
    return looper


class _LatentMoE(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1_latent_proj = nn.Identity()
        self.experts = nn.Linear(hidden, hidden, bias=False)


def _resolve(looper: WeightOnlyLooper, layer: nn.Module, module_name: str):
    return looper._resolve_named_module(
        layer_module=layer,
        full=find_modules(layer, name=""),
        layer_index=0,
        layer_path="layers.0.mixer",
        module_name=module_name,
        is_lm_head_module=False,
    )


def test_declared_identity_resolves_to_a_coverage_miss():
    looper = _looper()
    layer = _LatentMoE()

    assert _resolve(looper, layer, "fc1_latent_proj") is None

    snapshot = looper.layer_module_coverage.snapshot()
    assert snapshot["fc1_latent_proj"]["hit_layers"] == 0
    assert snapshot["fc1_latent_proj"]["miss_layers"] == 1


def test_strict_mode_still_reports_the_miss():
    """A placeholder is a miss, not a module, so strict mode keeps failing."""
    looper = _looper(strict=True)

    with pytest.raises(ValueError, match="fc1_latent_proj"):
        _resolve(looper, _LatentMoE(), "fc1_latent_proj")


def test_supported_module_still_resolves_through_the_full_map():
    looper = _looper()
    layer = _LatentMoE()

    named = _resolve(looper, layer, "experts")

    assert isinstance(named, NamedModule)
    assert named.full_name == "layers.0.mixer.experts"
    assert named.module is layer.experts
    assert looper.layer_module_coverage.snapshot()["experts"]["hit_layers"] == 1


def test_named_module_names_a_parameterless_module():
    with pytest.raises(ValueError, match="no parameters to quantize"):
        NamedModule(nn.Identity(), name="fc1", full_name="layers.0.mixer.fc1", layer_index=0)


def test_tiny_nemotron_h_rtn_quantize_completes(tmp_path):
    """The issue's regression: RTN on a non-latent Nemotron-H MoE layer."""
    nemotron = pytest.importorskip("transformers.models.nemotron_h")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    config = nemotron.NemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        hybrid_override_pattern="ME",
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        ssm_state_size=16,
        mamba_num_heads=2,
        mamba_head_dim=16,
        n_groups=2,
        conv_kernel=4,
        expand=2,
        tie_word_embeddings=False,
    )
    model = nemotron.NemotronHForCausalLM(config).eval()
    # The non-latent layout is the one that builds `nn.Identity` placeholders.
    moe = model.model.layers[1].mixer
    assert isinstance(moe.fc1_latent_proj, nn.Identity)
    model.save_pretrained(tmp_path)

    vocabulary = {"[UNK]": 0, "hello": 1, "world": 2}
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab=vocabulary, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]", eos_token="[UNK]"
    ).save_pretrained(tmp_path)

    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization.config import RTNConfig

    quantized = GPTQModel.load(
        str(tmp_path), RTNConfig(bits=4, group_size=16, device="cpu"), dtype=torch.float32
    )
    quantized.quantize()

    linear_count = sum(isinstance(m, TorchLinear) for m in quantized.model.modules())
    assert linear_count > 0
    # The placeholder stays untouched instead of aborting the run.
    assert isinstance(quantized.model.model.layers[1].mixer.fc1_latent_proj, nn.Identity)
