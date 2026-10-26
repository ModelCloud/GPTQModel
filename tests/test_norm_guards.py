# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Guard and failure-path coverage for the generalized norm tuner.

``tests/test_norm_config.py`` covers the happy paths: type selection, gain
probing and LayerNorm training. This module pins the contract of every
rejection and skip path, so a bad ``:norm`` tag or an unsupported
parameterization fails loudly instead of training the wrong tensor.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel import NormConfig, NormType
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.norm_tuning import _probe_gain_offset, resolve_norm_params, tune_norm_params


def _bare(cls, hf_model=None):
    model = BaseQModel.__new__(cls)
    nn.Module.__init__(model)
    model.quantized = True
    model.load_quantized_model = False
    if hf_model is not None:
        model.model = hf_model
    return model


class _NormLM(nn.Module):
    """Minimal causal LM whose only normalization module is `norm`."""

    def __init__(self, norm: nn.Module, dim: int = 8, vocab: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.norm = norm
        self.proj = nn.Linear(dim, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.proj(self.norm(self.embed(input_ids))))


def test_norm_config_validates_lr_reg_and_defaults_types():
    with pytest.raises(ValueError):
        NormConfig(teacher="t", lr=0)
    with pytest.raises(ValueError):
        NormConfig(teacher="t", reg=-1)
    # `types=None` is the documented "unspecified" spelling of the default.
    assert NormConfig(teacher="t", types=None).types == (NormType.RMSNORM,)


def test_resolve_norm_params_rejects_missing_and_unsupported_modules():
    with pytest.raises(ValueError, match="is not a module of this model"):
        resolve_norm_params(nn.Sequential(nn.RMSNorm(8)), ["9"], (), (NormType.RMSNORM,))
    with pytest.raises(ValueError, match="is not a supported normalization module"):
        resolve_norm_params(nn.Sequential(nn.Linear(8, 8)), ["0"], (), (NormType.RMSNORM,))


def test_resolve_norm_params_rejects_non_float_parameters():
    class IntGainRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.int64), requires_grad=False)

        def forward(self, hidden_states):
            normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
            return self.weight * normed

    class IntBiasRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.int64), requires_grad=False)

        def forward(self, hidden_states):
            normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
            return self.weight * normed

    with pytest.raises(ValueError, match="is not a floating point trainable gain"):
        resolve_norm_params(nn.Sequential(IntGainRMSNorm(8)), ["0"], (), (NormType.RMSNORM,))
    with pytest.raises(ValueError, match="is not a floating point trainable parameter"):
        resolve_norm_params(nn.Sequential(IntBiasRMSNorm(8)), ["0"], (), (NormType.RMSNORM,))


def test_resolve_norm_params_rejects_bias_shape_mismatch():
    class MismatchedBiasRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim + 1))

        def forward(self, hidden_states):
            normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
            return self.weight * normed

    with pytest.raises(ValueError, match="does not match"):
        resolve_norm_params(nn.Sequential(MismatchedBiasRMSNorm(8)), ["0"], (), (NormType.RMSNORM,))


def test_resolve_norm_params_rejects_empty_selection():
    model = nn.Sequential(nn.RMSNorm(8))
    with pytest.raises(ValueError, match="no normalization modules selected"):
        resolve_norm_params(model, [], (), (NormType.RMSNORM,))
    with pytest.raises(ValueError, match="no normalization modules selected"):
        resolve_norm_params(model, ["0"], ("0",), (NormType.RMSNORM,))
    with pytest.raises(ValueError, match="no normalization modules selected"):
        resolve_norm_params(model, ["0"], (), (NormType.LAYERNORM,))


def test_resolve_norm_params_deduplicates_repeated_module_names():
    """A repeated name must train its parameter once, not double its penalty."""
    model = nn.Sequential(nn.RMSNorm(8))
    entries = resolve_norm_params(model, ["0", "0", "0"], (), (NormType.RMSNORM,))
    assert [entry.label for entry in entries] == ["0.weight"]


def test_probe_still_rejects_unknown_parameterization_at_large_gains():
    """Large checkpoint gains must not hide an unsupported parameterization."""

    class ScaledLargeRMSNorm(nn.Module):
        """Effective gain is `0.25 * weight`, matching neither supported hypothesis."""

        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.full((dim,), 16.0))

        def forward(self, hidden_states):
            normed = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + 1e-6)
            return 0.25 * self.weight * normed

    with pytest.raises(ValueError, match="unrecognized gain parameterization"):
        resolve_norm_params(nn.Sequential(ScaledLargeRMSNorm(8)), ["0"], (), (NormType.RMSNORM,))


def test_probe_gain_offset_requires_a_float_weight():
    class NoWeightRMSNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    with pytest.raises(ValueError, match="has no floating point"):
        _probe_gain_offset(NoWeightRMSNorm(), "no_weight")


def test_probe_gain_offset_rejects_broken_modules():
    class ExplodingRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, hidden_states):
            raise RuntimeError("boom")

    class NonTensorRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, hidden_states):
            return "not a tensor"

    class BadShapeRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, hidden_states):
            return hidden_states[:1]

    with pytest.raises(ValueError, match="cannot probe"):
        resolve_norm_params(nn.Sequential(ExplodingRMSNorm(8)), ["0"], (), (NormType.RMSNORM,))
    with pytest.raises(ValueError, match="did not return a tensor"):
        resolve_norm_params(nn.Sequential(NonTensorRMSNorm(8)), ["0"], (), (NormType.RMSNORM,))
    with pytest.raises(ValueError, match="returned shape"):
        resolve_norm_params(nn.Sequential(BadShapeRMSNorm(8)), ["0"], (), (NormType.RMSNORM,))


def test_probe_gain_offset_checks_each_instance():
    class CountingRMSNorm(nn.RMSNorm):
        calls = 0

        def forward(self, hidden_states):
            type(self).calls += 1
            return super().forward(hidden_states)

    model = nn.Sequential(CountingRMSNorm(8), CountingRMSNorm(8))
    entries = resolve_norm_params(model, ["0", "1"], (), (NormType.RMSNORM,))
    assert [entry.offset for entry in entries] == [0.0, 0.0]
    # Each module gets two controlled-gain forwards; instances may configure
    # different offsets even when they share a class.
    assert CountingRMSNorm.calls == 4
    resolve_norm_params(model, ["0"], (), (NormType.RMSNORM,))
    assert CountingRMSNorm.calls == 6


def test_tune_norm_params_requires_calibration_data():
    teacher = _NormLM(nn.RMSNorm(8))
    student = _NormLM(nn.RMSNorm(8))
    with pytest.raises(ValueError, match="requires calibration data"):
        tune_norm_params(student, teacher, [], ["norm"], NormConfig(teacher=teacher, steps=1))


def test_tune_norm_params_rejects_a_gain_that_receives_no_gradient():
    class _DualNormLM(nn.Module):
        """Only `used` sits on the forward path, so `unused.weight` never gets a gradient."""

        def __init__(self, dim: int = 8, vocab: int = 32):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.used = nn.RMSNorm(dim)
            self.unused = nn.RMSNorm(dim)
            self.proj = nn.Linear(dim, vocab, bias=False)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            del attention_mask, use_cache
            return SimpleNamespace(logits=self.proj(self.used(self.embed(input_ids))))

    teacher = _NormLM(nn.RMSNorm(8))
    student = _DualNormLM()
    batches = [{"input_ids": torch.randint(0, 32, (2, 4))}]
    with pytest.raises(RuntimeError, match="no gradient reached"):
        tune_norm_params(student, teacher, batches, ["used", "unused"], NormConfig(teacher=teacher, steps=1))


def test_tune_norm_params_skips_non_finite_steps():
    class NonFiniteLM(_NormLM):
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            out = super().forward(input_ids, attention_mask, use_cache)
            return SimpleNamespace(logits=out.logits * float("inf"))

    teacher = _NormLM(nn.RMSNorm(8))
    student = NonFiniteLM(nn.RMSNorm(8))
    before = student.norm.weight.detach().clone()
    batches = [{"input_ids": torch.randint(0, 32, (2, 4))}]

    stats = tune_norm_params(student, teacher, batches, ["norm"], NormConfig(teacher=teacher, steps=3, reg=0.0))

    # Every step produced non-finite gradients, so none may be applied.
    assert stats["skipped_steps"] == 3
    assert torch.equal(student.norm.weight.detach(), before)


def test_norm_module_names_skips_unsupported_head_norm():
    class HeadLinearQModel(BaseQModel):
        pre_lm_head_norm_module = "model.head_linear"
        module_tree = ["model", "layers", "#", {"input_layernorm": ("input_layernorm:!:norm",)}]

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            block = nn.Module()
            block.input_layernorm = nn.RMSNorm(8)
            self.model.layers = nn.ModuleList([block])
            self.model.head_linear = nn.Linear(8, 8, bias=False)

    # A non-norm head module is skipped rather than raising or being tuned.
    assert _bare(HeadLinearQModel, Wrapper()).norm_module_names() == ["model.layers.0.input_layernorm"]


def test_requantize_requires_a_target():
    model = _bare(BaseQModel, nn.Module())
    with pytest.raises(ValueError, match="requires"):
        model.requantize(calibration=["sample"])
