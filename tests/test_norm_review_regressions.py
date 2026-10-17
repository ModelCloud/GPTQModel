# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regressions found while reviewing merged PR #3306."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from gptqmodel import NormConfig, NormType, RMSNormConfig
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.norm_tuning import _probe_gain_offset, resolve_norm_params, tune_norm_params


class ConfigurableRMSNorm(nn.Module):
    def __init__(self, width=8, *, offset=0.0, multiplier=1.0, weight=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((width,), weight))
        self.offset = offset
        self.multiplier = multiplier
        self.eps = 1e-6

    def forward(self, x):
        unit = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (unit * (self.multiplier * self.weight + self.offset)).to(x.dtype)


_DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.mark.parametrize('device', _DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(3,), (7,), (64,), (3, 5)])
@pytest.mark.parametrize('family', [NormType.RMSNORM, NormType.LAYERNORM])
def test_probe_odd_and_multidimensional_norms_against_scale_oracle(device, dtype, shape, family):
    cls = nn.RMSNorm if family == NormType.RMSNORM else nn.LayerNorm
    module = cls(shape, eps=0.1).to(device=device, dtype=dtype)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-2, 3, module.weight.numel()).reshape(shape))
        if getattr(module, 'bias', None) is not None:
            module.bias.fill_(128)
    before = {name: value.clone() for name, value in module.state_dict().items()}
    param = module.weight
    entries = resolve_norm_params(nn.Sequential(module), ['0'], types=(family,))
    assert entries[0].offset == 0.0
    # Separate float64 arithmetic on the stored checkpoint; no probed offsets
    # or scales are reused to construct the expected value.
    expected = float(before['weight'].double().abs().mean().clamp_min(1e-6))
    assert entries[0].scale == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert module.weight is param
    assert all(torch.equal(value, before[name]) for name, value in module.state_dict().items())


@pytest.mark.parametrize('device', _DEVICES)
@pytest.mark.parametrize('width', [7, 128, 4096])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('cls, offset', [(LlamaRMSNorm, 0.0), (Qwen3RMSNorm, 0.0), (Gemma2RMSNorm, 1.0)])
@pytest.mark.parametrize('gain', [0.0, 16.0, 512.0])
def test_hf_probe_large_and_zero_gains(device, width, dtype, cls, offset, gain):
    module = cls(width).to(device=device, dtype=dtype)
    with torch.no_grad():
        module.weight.fill_(gain)
    entry = resolve_norm_params(nn.Sequential(module), ['0'])[0]
    assert entry.offset == offset
    expected = float((module.weight.detach().double() + offset).abs().mean().clamp_min(1e-6))
    assert entry.scale == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_offsets_are_resolved_per_instance_and_rechecked_after_configuration_changes():
    class InstanceRMSNorm(ConfigurableRMSNorm):
        pass

    plain = InstanceRMSNorm()
    offset = InstanceRMSNorm(offset=1.0, weight=0.0)
    assert _probe_gain_offset(plain, 'plain') == 0.0
    assert _probe_gain_offset(offset, 'offset') == 1.0
    plain.offset = 1.0
    assert _probe_gain_offset(plain, 'plain') == 1.0


@pytest.mark.parametrize('offset, multiplier, weight', [(0.5, 1.0, 100.0), (0.0, 0.25, 0.0)])
def test_unknown_parameterizations_cannot_hide_behind_large_or_zero_gains(offset, multiplier, weight):
    class UnknownRMSNorm(ConfigurableRMSNorm):
        pass

    module = UnknownRMSNorm(offset=offset, multiplier=multiplier, weight=weight)
    before = module.weight.detach().clone()
    with pytest.raises(ValueError, match='unrecognized gain parameterization'):
        _probe_gain_offset(module, 'unknown')
    assert torch.equal(module.weight, before)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_probe_output_is_rejected(bad):
    class BadRMSNorm(ConfigurableRMSNorm):
        def forward(self, x):
            return super().forward(x) * bad

    with pytest.raises(ValueError, match='non-finite'):
        _probe_gain_offset(BadRMSNorm(), 'bad')


@pytest.mark.parametrize('kind', ['weight', 'bias'])
def test_nonfinite_checkpoint_parameters_are_rejected(kind):
    module = nn.LayerNorm(8)
    with torch.no_grad():
        getattr(module, kind)[0] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        resolve_norm_params(nn.Sequential(module), ['0'], types=(NormType.LAYERNORM,))


def test_degenerate_layer_norm_cannot_be_mistaken_for_an_offset_gain():
    with pytest.raises(ValueError, match='ambiguous'):
        _probe_gain_offset(nn.LayerNorm(1), 'scalar_norm')


def test_failed_probe_does_not_mutate_the_live_module():
    class ExplodingRMSNorm(ConfigurableRMSNorm):
        def forward(self, x):
            self.weight.zero_()
            raise RuntimeError('probe failed')

    module = ExplodingRMSNorm().train()
    param = module.weight
    before = param.detach().clone()
    with pytest.raises(ValueError, match='cannot probe'):
        _probe_gain_offset(module, 'bad')
    assert module.training and module.weight is param and param.requires_grad
    assert torch.equal(param, before)


def test_legacy_config_positional_order_is_preserved():
    old = RMSNormConfig('teacher', 10, 0.01, 0.02, 42, ('q_norm',))
    new = NormConfig(teacher='teacher', steps=10, lr=0.01, reg=0.02, seed=42, skip=('q_norm',))
    assert old == new


@pytest.mark.parametrize('field', ['lr', 'reg'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_training_configuration_is_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        NormConfig(teacher='teacher', **{field: bad})


class HeadQModel(BaseQModel):
    pre_lm_head_norm_module = 'head'
    module_tree = ['layers', '#', {'norm': ('norm:!:norm',)}]


def _head_model(layer_norm):
    wrapper = HeadQModel.__new__(HeadQModel)
    nn.Module.__init__(wrapper)
    wrapper.model = nn.Module()
    block = nn.Module()
    block.norm = layer_norm
    wrapper.model.layers = nn.ModuleList([block])
    wrapper.model.head = nn.LayerNorm(8)
    wrapper.quantized = True
    wrapper.load_quantized_model = False
    return wrapper


def test_final_norm_is_selected_even_when_layer_norms_are_filtered_out():
    wrapper = _head_model(nn.RMSNorm(8))
    assert wrapper.norm_module_names((NormType.LAYERNORM,)) == ['head']


def test_skip_applies_before_unsupported_tag_validation_through_public_api():
    wrapper = _head_model(nn.Linear(8, 8))
    wrapper.prepare_dataset = lambda **kwargs: pytest.fail('selection should be empty before dataset preparation')
    with pytest.raises(ValueError, match='no `:norm` modules matching'):
        wrapper.requantize([], norm=NormConfig(teacher='must-not-load', types=(NormType.LAYERNORM,),
                                              skip=('layers.0.norm', 'head')))


def test_deprecated_requantize_preserves_result_key_and_positional_slot():
    wrapper = _head_model(nn.RMSNorm(8))
    wrapper._requantize_norm = lambda **kwargs: {'steps': kwargs['norm'].steps}
    config = RMSNormConfig(teacher='t', steps=1)
    assert wrapper.requantize([], rmsnorm=config) == {'rmsnorm': {'steps': 1}}
    assert wrapper.requantize([], None, None, 'desc', 1, None, None, None, None, 10, None, None, config) == {
        'rmsnorm': {'steps': 1}}
    assert wrapper.requantize([], norm=config) == {'norm': {'steps': 1}}


class SharedNormLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.norm = nn.LayerNorm(8)
        self.alias = self.norm
        self.proj = nn.Linear(8, 16)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.proj(self.norm(self.embed(input_ids))))


def test_shared_norm_parameter_is_optimized_once_but_both_paths_are_persisted():
    torch.manual_seed(11)
    teacher = SharedNormLM().eval()
    student = copy.deepcopy(teacher)
    with torch.no_grad():
        student.norm.weight.mul_(1.2)
    control = copy.deepcopy(student)
    batches = [{'input_ids': torch.tensor([[0, 1, 2, 3]])}]
    config = NormConfig(teacher=teacher, steps=3, reg=0.01, types=(NormType.LAYERNORM,))
    tune_norm_params(control, teacher, batches, ['norm'], config)
    stats = tune_norm_params(student, teacher, batches, ['norm', 'alias'], config)
    assert stats['modules'] == ['norm', 'alias']
    assert stats['skipped_steps'] == 0
    for name, tensor in control.state_dict().items():
        assert torch.equal(tensor, student.state_dict()[name]), name


def test_shared_parameter_with_incompatible_offsets_is_rejected_without_updates():
    plain = ConfigurableRMSNorm(offset=0.0)
    offset = ConfigurableRMSNorm(offset=1.0)
    offset.weight = plain.weight
    model = nn.Sequential(plain, offset)
    before = plain.weight.detach().clone()
    with pytest.raises(ValueError, match='incompatible gain scales'):
        tune_norm_params(model, nn.Module(), [{'input_ids': torch.tensor([[0, 1]])}], ['0', '1'],
                         NormConfig(teacher='unused', steps=1))
    assert torch.equal(plain.weight, before)
    assert plain.weight is offset.weight and plain.weight.requires_grad


@pytest.mark.parametrize('offset', [0.0, 1.0])
def test_tuning_updates_match_independent_torch_oracle(offset):
    """Check complete regularized updates without calling the norm in the oracle."""
    torch.manual_seed(17)
    student = SharedNormLM().eval()
    student.norm = ConfigurableRMSNorm(offset=offset, weight=0.2 if offset else 1.2)
    student.alias = student.norm
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher.norm.weight.add_(torch.linspace(-0.2, 0.3, 8))
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    config = NormConfig(teacher=teacher, steps=4, lr=0.01, reg=0.02)
    base = student.norm.weight.detach().clone()
    gain = base.clone().requires_grad_(True)
    # This scale uses float64 checkpoint data and the known parametrization,
    # independently of resolve_norm_params and its intermediate outputs.
    scale = (base.double() + offset).abs().mean().clamp_min(1e-6)
    optimizer = torch.optim.Adam([gain], lr=config.lr)
    with torch.no_grad():
        embedded = torch.nn.functional.embedding(ids, student.embed.weight)
        unit = embedded * torch.rsqrt(embedded.square().mean(-1, keepdim=True) + 1e-6)
        target_logits = torch.nn.functional.linear(
            unit * (teacher.norm.weight + offset), teacher.proj.weight, teacher.proj.bias)
        target = target_logits.log_softmax(-1)
    for _ in range(config.steps):
        optimizer.zero_grad()
        logits = torch.nn.functional.linear(unit * (gain + offset), student.proj.weight.detach(),
                                            student.proj.bias.detach())
        per_token = (target.exp() * (target - logits.log_softmax(-1))).sum(-1)
        kl = (per_token * mask).sum() / mask.sum()
        penalty = ((gain - base) / scale).square().mean()
        (kl + config.reg * penalty).backward()
        optimizer.step()
    stats = tune_norm_params(student, teacher, [{'input_ids': ids, 'attention_mask': mask}], ['norm'], config)
    assert stats['skipped_steps'] == 0
    torch.testing.assert_close(student.norm.weight.detach(), gain.detach(), rtol=1e-6, atol=1e-6)
