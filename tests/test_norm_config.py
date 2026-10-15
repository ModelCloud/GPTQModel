# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for the generalized ``NormConfig``: type selector, handlers and LayerNorm."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Gemma2Config, Gemma2ForCausalLM, GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM

from gptqmodel import NormConfig, NormType
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.definitions.gemma2 import Gemma2QModel
from gptqmodel.models.definitions.gpt2 import GPT2QModel
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.quantization.norm_tuning import norm_type_of, resolve_norm_params, tune_norm_params


def _bare(cls, hf_model=None):
    model = BaseQModel.__new__(cls)
    nn.Module.__init__(model)
    model.quantized = True
    model.load_quantized_model = False
    if hf_model is not None:
        model.model = hf_model
    return model


def _tiny_qwen3(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
        tie_word_embeddings=False,
    )).eval()


def _tiny_gpt2(seed: int = 0) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=2, n_inner=64,
    )).eval()


def _tiny_gemma2(seed: int = 0) -> Gemma2ForCausalLM:
    torch.manual_seed(seed)
    return Gemma2ForCausalLM(Gemma2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
        tie_word_embeddings=False,
    )).eval()


def _kl(student, teacher, input_ids) -> float:
    with torch.no_grad():
        t = F.log_softmax(teacher(input_ids=input_ids).logits.float(), -1)
        s = F.log_softmax(student(input_ids=input_ids).logits.float(), -1)
    return float((t.exp() * (t - s)).sum(-1).mean())


class _MixedQModel(BaseQModel):
    """Definition with one RMSNorm and one LayerNorm per layer, plus a LayerNorm head norm."""

    pre_lm_head_norm_module = "model.final_norm"

    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:norm",),
            "ln": ("ln:!:norm",),
            "proj": ("proj:!",),
        },
    ]


class _LegacyFlagQModel(BaseQModel):
    module_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!:rmsnorm",),
        },
    ]


def _mixed_model(blocks: int = 2, dim: int = 8) -> nn.Module:
    torch.manual_seed(0)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(dim)
            self.ln = nn.LayerNorm(dim)
            self.proj = nn.Linear(dim, dim, bias=False)

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(Block() for _ in range(blocks))
            self.model.final_norm = nn.LayerNorm(dim)

    return Wrapper()


def test_norm_type_classification():
    assert norm_type_of(nn.RMSNorm(8)) is NormType.RMSNORM
    assert norm_type_of(nn.LayerNorm(8)) is NormType.LAYERNORM
    assert norm_type_of(type("LlamaRMSNorm", (nn.Module,), {})()) is NormType.RMSNORM
    assert norm_type_of(type("BloomLayerNorm", (nn.Module,), {})()) is NormType.LAYERNORM
    # Gated and fused norms are not plain gains and stay unclassified.
    assert norm_type_of(type("Qwen3NextRMSNormGated", (nn.Module,), {})()) is None
    assert norm_type_of(nn.Linear(4, 4)) is None


def test_norm_module_names_filter_by_type():
    model = _bare(_MixedQModel, _mixed_model())

    assert model.norm_module_names() == ["model.layers.0.input_layernorm", "model.layers.1.input_layernorm"]
    assert model.norm_module_names((NormType.LAYERNORM,)) == [
        "model.layers.0.ln",
        "model.layers.1.ln",
        "model.final_norm",
    ]
    assert model.norm_module_names(("rmsnorm", "layernorm")) == [
        "model.layers.0.input_layernorm",
        "model.layers.0.ln",
        "model.layers.1.input_layernorm",
        "model.layers.1.ln",
        "model.final_norm",
    ]


def test_norm_module_names_reject_untagged_types():
    model = _bare(_MixedQModel, _mixed_model())
    assert model.norm_module_names((NormType.LAYERNORM,)) != model.norm_module_names((NormType.RMSNORM,))

    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            block = nn.Module()
            block.proj = nn.Linear(4, 4, bias=False)
            self.model.layers = nn.ModuleList([block])
            self.model.final_norm = nn.LayerNorm(4)

    class BrokenQModel(BaseQModel):
        module_tree = ["model", "layers", "#", {"proj": ("proj:!:norm",)}]

    with pytest.raises(ValueError, match="no normalization handler supports"):
        _bare(BrokenQModel, Broken()).norm_module_names()


def test_legacy_rmsnorm_flag_is_still_collected():
    assert _LegacyFlagQModel.get_norm_modules() == ["input_layernorm"]
    assert _LegacyFlagQModel.has_norm_flag("input_layernorm:!:rmsnorm")
    assert _LegacyFlagQModel.has_norm_flag("input_layernorm:!:norm")
    assert not _LegacyFlagQModel.has_norm_flag("input_layernorm:!")


def test_gain_probe_resolves_parameterization():
    standard = nn.Sequential(nn.RMSNorm(8))
    standard[0].weight.data.mul_(1.4)
    assert resolve_norm_params(standard, ["0"], (), (NormType.RMSNORM,))[0].offset == 0.0

    gemma = _tiny_gemma2()
    name = "model.layers.0.input_layernorm"
    assert type(gemma.get_submodule(name)).__name__.endswith("RMSNorm")
    entry = resolve_norm_params(gemma, [name], (), (NormType.RMSNORM,))[0]
    assert entry.offset == 1.0
    assert entry.scale == pytest.approx(1.0, abs=1e-6)


def test_gain_probe_rejects_unknown_parameterization():
    class ScaledRMSNorm(nn.Module):
        """Effective gain is `0.25 * weight`, matching neither supported hypothesis."""

        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, hidden_states):
            normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
            return 0.25 * self.weight * normed

    model = nn.Sequential(ScaledRMSNorm(8))
    with pytest.raises(ValueError, match="unrecognized gain parameterization"):
        resolve_norm_params(model, ["0"], (), (NormType.RMSNORM,))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gain_probe_resolves_offsets_in_serving_dtypes(dtype):
    """Serving checkpoints load in fp16/bf16; the probe must still resolve the gain."""

    # Fresh classes per run: the probe cache is keyed by class, so reusing a
    # module-level class here would return the cached offset without probing.
    class PlainRMSNorm(nn.RMSNorm):
        pass

    class OffsetRMSNorm(nn.Module):
        """Gemma-style `1 + weight` gain."""

        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(dim, dtype=dtype))

        def forward(self, hidden_states):
            normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
            return (1.0 + self.weight) * normed

    plain = nn.Sequential(PlainRMSNorm(16).to(dtype))
    offset = nn.Sequential(OffsetRMSNorm(16))

    assert resolve_norm_params(plain, ["0"], (), (NormType.RMSNORM,))[0].offset == 0.0
    assert resolve_norm_params(offset, ["0"], (), (NormType.RMSNORM,))[0].offset == 1.0


def test_regularization_scale_matches_independent_oracle():
    model = nn.Module()
    model.standard = nn.RMSNorm(16)
    model.offset = _tiny_gemma2().model.layers[0].input_layernorm
    model.layer = nn.LayerNorm(16)
    with torch.no_grad():
        model.standard.weight.copy_(torch.linspace(0.2, 1.6, 16))
        model.offset.weight.copy_(torch.linspace(-0.3, 0.4, model.offset.weight.numel()))
        model.layer.weight.copy_(torch.linspace(0.5, 1.5, 16))
        model.layer.bias.zero_()

    entries = {entry.label: entry for entry in resolve_norm_params(
        model,
        ["standard", "offset", "layer"],
        (),
        (NormType.RMSNORM, NormType.LAYERNORM),
    )}

    for label, entry in entries.items():
        # Independent float64 oracle for `mean(|stored + offset|)`.
        if entry.kind == "bias":
            gain = entries[f"{entry.module}.weight"]
            oracle = float((gain.param.detach().double() + gain.offset).abs().mean().clamp_min(1e-6))
        else:
            oracle = float((entry.param.detach().double() + entry.offset).abs().mean().clamp_min(1e-6))
        assert entry.scale == pytest.approx(oracle, rel=1e-6, abs=1e-9), label

    # RMSNorm parity with the first release: the scale is the legacy float32 formula.
    legacy = float(model.standard.weight.detach().float().abs().mean().clamp_min(1e-6))
    assert entries["standard.weight"].scale == legacy
    # A zero-initialized bias shares the gain scale instead of its own degenerate one.
    assert entries["layer.bias"].scale == entries["layer.weight"].scale
    assert entries["offset.weight"].offset == 1.0


def test_layer_norm_tuning_trains_weight_and_bias():
    teacher = _tiny_gpt2()
    student = copy.deepcopy(teacher)
    names = _bare(GPT2QModel, student).norm_module_names((NormType.LAYERNORM,))
    assert len(names) == 2 * 2 + 1
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name in names:
            module = student.get_submodule(name)
            module.weight.mul_(1 + 0.3 * torch.randn(module.weight.shape, generator=generator))
            module.bias.add_(0.2 * torch.randn(module.bias.shape, generator=generator))
    before = {name: tensor.detach().clone() for name, tensor in student.state_dict().items()}
    grad_flags = {name: param.requires_grad for name, param in student.named_parameters()}
    input_ids = torch.randint(0, 64, (8, 16), generator=generator)
    batches = [{"input_ids": input_ids[i:i + 2]} for i in range(0, 8, 2)]
    kl_before = _kl(student, teacher, input_ids)

    stats = tune_norm_params(
        student,
        teacher,
        batches,
        names,
        NormConfig(teacher=teacher, types=(NormType.LAYERNORM,), steps=60, lr=1e-2, reg=0.0),
    )

    assert stats["types"] == ["layernorm"]
    assert stats["modules"] == names
    assert _kl(student, teacher, input_ids) < 0.5 * kl_before
    changed = {name for name, tensor in student.state_dict().items() if not torch.equal(tensor, before[name])}
    assert changed == {f"{name}.{kind}" for name in names for kind in ("weight", "bias")}
    assert {name: param.requires_grad for name, param in student.named_parameters()} == grad_flags


def test_offset_gains_train_despite_regularization():
    """Gemma-style `1 + weight` gains must not be pinned by a near-zero stored scale."""
    teacher = _tiny_gemma2()
    student = copy.deepcopy(teacher)
    names = _bare(Gemma2QModel, student).norm_module_names()
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for name in names:
            weight = student.get_submodule(name).weight
            weight.add_(0.1 * torch.randn(weight.shape, generator=generator))
    before = {name: tensor.detach().clone() for name, tensor in student.state_dict().items()}
    input_ids = torch.randint(0, 64, (8, 16), generator=generator)
    batches = [{"input_ids": input_ids[i:i + 2]} for i in range(0, 8, 2)]
    kl_before = _kl(student, teacher, input_ids)

    stats = tune_norm_params(
        student,
        teacher,
        batches,
        names,
        NormConfig(teacher=teacher, steps=60, lr=1e-2, reg=1e-3),
    )

    assert stats["modules"] == names
    assert _kl(student, teacher, input_ids) < 0.5 * kl_before
    changed = {name for name, tensor in student.state_dict().items() if not torch.equal(tensor, before[name])}
    assert changed == {f"{name}.weight" for name in names}
    # The tuned gains move; they are not frozen at the checkpoint value by the penalty.
    moved = max(
        float((student.get_submodule(name).weight.detach() - before[f"{name}.weight"]).abs().max()) for name in names
    )
    assert moved > 1e-3


def test_qwen3_rmsnorm_tuning_still_filters_layer_norm_types():
    model = _bare(Qwen3QModel, _tiny_qwen3())
    assert model.norm_module_names((NormType.RMSNORM,)) == model.norm_module_names()
    assert model.norm_module_names((NormType.LAYERNORM,)) == []
