import json

import pytest
import torch

from gptqmodel.quantization import GSQTrainingConfig


def test_staged_config_roundtrip_and_distinct_defaults():
    config = GSQTrainingConfig()
    assert not config.enabled
    assert config.qk_steps == 2000 and config.epochs == 10
    assert GSQTrainingConfig(**json.loads(json.dumps(config.to_dict()))) == config
    assert config.training_kwargs()['qk_damp_percent'] == .01


@pytest.mark.parametrize('values', [
    {'optimizer': 'unknown'}, {'enabled': 1}, {'epochs': 0}, {'qk_steps': True}, {'seed': -1},
    {'assignment_lr': 0}, {'scale_lr': float('nan')}, {'weight_decay': -1},
    {'damp_percent': float('inf')}, {'betas': [.9, 1.]}, {'temperature': [2.]},
    {'multiplier': [100., 0.]}, {'warmup_steps': -1}, {'min_lr': 2.}, {'decay': 'unknown'},
    {'initializer': 'unknown'}, {'batch_size': 0}, {'microbatch_size': True},
    {'batch_size': 2, 'microbatch_size': 3},
])
def test_staged_config_rejects_invalid_settings(values):
    with pytest.raises((ValueError, TypeError)):
        GSQTrainingConfig(**values)


@pytest.mark.parametrize('optimizer', ['lion', 'adamw'])
@pytest.mark.parametrize('bits', [2, 3, 4])
@pytest.mark.parametrize('initializer', ['gptq', 'gptq_signed', 'rtn'])
def test_configured_block_entry_default_off_and_enabled_packed_reload(bits, initializer, optimizer, tmp_path, monkeypatch):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel.quantization import gsq_training

    torch.manual_seed(7)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    original = {key: value.clone() for key, value in layer.state_dict().items()}
    hidden = torch.randn(2, 16, 32)
    kwargs = dict(position_embeddings=LlamaRotaryEmbedding(config)(hidden, torch.arange(16)[None]),
                  attention_mask=torch.full((16, 16), -torch.inf).triu(1)[None, None], use_cache=False)
    batches = [(hidden, kwargs)]
    seeds, _ = gsq_training.initialize_llama_gptq(layer, batches, bits=bits, group_size=32,
                                                 damp_percent=.01, initializer=initializer)
    if initializer == 'gptq_signed':
        assert any((scales < 0).any() for _, scales in seeds.values())
    fitter = gsq_training.fit_llama_stages

    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled configuration must not train')
    monkeypatch.setattr(gsq_training, 'fit_llama_stages', forbidden)
    baseline, info = gsq_training.quantize_llama_gsq_block(
        layer, batches, bits=bits, group_size=32, pack=False,
        gsq=GSQTrainingConfig(initializer=initializer) if initializer != 'gptq' else None)
    assert not info['gsq_training']['enabled']
    for name, (weight, _) in seeds.items():
        torch.testing.assert_close(baseline.get_submodule(name).weight, weight, rtol=0, atol=0)
    monkeypatch.setattr(gsq_training, 'fit_llama_stages', fitter)
    constructed = []
    owner, attribute = (torch.optim, 'AdamW') if optimizer == 'adamw' else (gsq_training, 'GSQLion')
    constructor = getattr(owner, attribute)

    def record_optimizer(*args, **kwargs):
        result = constructor(*args, **kwargs)
        constructed.append(result)
        return result

    monkeypatch.setattr(owner, attribute, record_optimizer)
    settings = GSQTrainingConfig(enabled=True, epochs=2, qk_steps=2, initializer=initializer, optimizer=optimizer)
    exported, info = gsq_training.quantize_llama_gsq_block(
        layer, batches, bits=bits, group_size=32, gsq=json.loads(json.dumps(settings.to_dict())))
    assert len(constructed) == 4  # Q, K, attention, and MLP must all honor the selection.
    assert info['gsq_training'] == settings.to_dict()
    assert info['stages']['mlp']['initializer_timing'] == 'after_attention'
    assert len(info['stages']['attention']['history']) == 2
    before = exported(hidden, **kwargs)
    assert torch.isfinite(before).all()
    path = tmp_path/'block.pt'
    torch.save(exported.state_dict(), path)
    exported.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(exported(hidden, **kwargs), before, rtol=0, atol=0)
    assert all(torch.equal(value, original[name]) for name, value in layer.state_dict().items())
