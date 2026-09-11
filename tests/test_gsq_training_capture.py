import pytest
import torch

from gptqmodel.looper.gsq_training_capture import prepare_llama_gsq_capture
from gptqmodel.looper.input_cache import InputCache
from gptqmodel.nn_modules.hooked_linear import HookedLinear


def fixture():
    from transformers import LlamaConfig, LlamaModel

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    model = LlamaModel(config).eval()
    layer = model.layers[0]
    for name, module in list(layer.named_modules()):
        if isinstance(module, torch.nn.Linear):
            parent, leaf = name.rsplit('.', 1)
            setattr(layer.get_submodule(parent), leaf, HookedLinear.from_linear(module))
    captured = []
    handle = layer.register_forward_pre_hook(lambda module, args, kwargs: captured.append((args, kwargs)),
                                              with_kwargs=True)
    with torch.inference_mode():
        model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)
    handle.remove()
    args, kwargs = captured[0]
    hidden = args[0] if args else kwargs['hidden_states']
    kwargs = dict(kwargs)
    kwargs.pop('hidden_states', None)
    positions = kwargs.pop('position_ids', None)
    mask = kwargs.pop('attention_mask', None)
    cache = InputCache([[hidden]], [kwargs], [positions], [mask])
    return layer, cache


def test_real_model_capture_preserves_forward_and_enables_staged_gradients():
    from gptqmodel.quantization.gsq_training import reconstruction_stage_loss

    layer, cache = fixture()
    assert cache.layer_inputs[0][0].is_inference()
    with torch.inference_mode():
        prepared, batches = prepare_llama_gsq_capture(layer, cache)
    hidden, kwargs = batches[0]
    assert not hidden.is_inference()
    assert all(not value.is_inference() for value in kwargs['position_embeddings'])
    assert isinstance(layer.self_attn.q_proj, HookedLinear)
    assert type(prepared.self_attn.q_proj) is torch.nn.Linear
    with torch.no_grad():
        torch.testing.assert_close(prepared(hidden, **kwargs), layer(hidden, **kwargs), rtol=0, atol=0)
    weights = {name+'.weight': prepared.get_submodule(name).weight.detach().clone().requires_grad_()
               for name in ('mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')}
    with torch.no_grad():
        weights['mlp.down_proj.weight'].add_(.01)
    loss = reconstruction_stage_loss(prepared, (hidden,), kwargs, student_weights=weights)
    loss.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in weights.values())
    assert all(parameter.grad is None for parameter in layer.parameters())
    assert cache.layer_inputs[0][0].is_inference()


def test_capture_rejects_missing_rotary_and_unpreserved_transform():
    layer, cache = fixture()
    layer.self_attn.q_proj.online_full_had = True
    with pytest.raises(ValueError, match='Hadamard'):
        prepare_llama_gsq_capture(layer, cache)
    layer.self_attn.q_proj.online_full_had = False
    cache.layer_input_kwargs[0].pop('position_embeddings')
    with pytest.raises(ValueError, match='rotary'):
        prepare_llama_gsq_capture(layer, cache)


def test_captured_block_entry_trains_and_packs_under_inference_mode():
    from gptqmodel.looper.gsq_training_capture import quantize_llama_gsq_capture
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    layer, cache = fixture()
    original = {name: value.clone() for name, value in layer.state_dict().items()}
    with torch.inference_mode():
        exported, info = quantize_llama_gsq_capture(
            layer, cache, bits=2, group_size=32, gsq=GSQTrainingConfig(enabled=True, epochs=2, qk_steps=2))
        assert torch.is_inference_mode_enabled()
    assert isinstance(exported.self_attn.q_proj, TorchLinear)
    assert len(info['stages']['attention']['history']) == 2
    assert info['stages']['mlp']['initializer_timing'] == 'after_attention'
    assert all(torch.equal(value, original[name]) for name, value in layer.state_dict().items())
    assert isinstance(layer.self_attn.q_proj, HookedLinear)
    _, batches = prepare_llama_gsq_capture(layer, cache)
    hidden, kwargs = batches[0]
    assert torch.isfinite(exported(hidden, **kwargs)).all()


def test_model_capture_preserves_positions_uses_current_prefix_and_cleans_hooks():
    from transformers import LlamaConfig, LlamaForCausalLM
    from gptqmodel.looper.gsq_training_capture import capture_llama_gsq_inputs

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=4, num_hidden_layers=2)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval()
    documents = [{'input_ids': [1, 2, 3], 'position_ids': [10, 11, 12]}]
    first = capture_llama_gsq_inputs(model, documents, layer_index=1)
    assert torch.equal(first.position_ids[0], torch.tensor([[10, 11, 12]]))
    assert not model.model.layers[1]._forward_pre_hooks
    with torch.no_grad():
        model.model.layers[0].mlp.down_proj.weight.add_(.1)
    second = capture_llama_gsq_inputs(model, documents, layer_index=1)
    assert not torch.equal(first.layer_inputs[0][0], second.layer_inputs[0][0])
    with pytest.raises(ValueError, match='unpadded'):
        capture_llama_gsq_inputs(model, [{'input_ids': [1, 2], 'attention_mask': [1, 0]}], layer_index=1)
    assert not model.model.layers[1]._forward_pre_hooks


def test_model_capture_can_offload_equal_length_default_metadata():
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.looper.gsq_training_capture import capture_llama_gsq_inputs

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    documents = [{'input_ids': [1, 2, 3]}, {'input_ids': [4, 5, 6]}]
    cache = capture_llama_gsq_inputs(model, documents, offload_to_cpu=True)
    assert all(inputs[0].device.type == 'cpu' for inputs in cache.layer_inputs)
    assert cache.layer_input_kwargs[0]['position_embeddings'][0].data_ptr() == (
        cache.layer_input_kwargs[1]['position_embeddings'][0].data_ptr())
    with pytest.raises(ValueError, match='equal-length'):
        capture_llama_gsq_inputs(model, documents+[{'input_ids': [7]}], offload_to_cpu=True)


def test_affine_capture_preserves_scaled_teacher_and_inference_cache():
    from gptqmodel.looper.gsq_training_capture import fit_llama_awq_gsq_capture
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    torch.manual_seed(7)
    layer, cache = fixture()
    prepared, batches = prepare_llama_gsq_capture(layer, cache)
    hidden, kwargs = batches[0]
    with torch.no_grad():
        before = prepared(hidden, **kwargs)
        channel_scale = torch.linspace(.5, 2., 32)
        layer.input_layernorm.weight.div_(channel_scale)
        for name in ('q_proj', 'k_proj', 'v_proj'):
            getattr(layer.self_attn, name).weight.mul_(channel_scale)
        torch.testing.assert_close(layer(hidden, **kwargs), before, atol=1e-6, rtol=1e-5)
    pristine = {key: value.clone() for key, value in layer.state_dict().items()}
    initializers = {}
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            grouped = module.weight.detach().reshape(module.out_features, -1, 32)
            scales = (grouped.amax(-1)-grouped.amin(-1)).clamp_min(1e-6)/15
            zeros = (-grouped.amin(-1)/scales).round().clamp(0, 15)
            codes = (grouped/scales.unsqueeze(-1)+zeros.unsqueeze(-1)).round().clamp(0, 15)
            initializers[name] = (codes.reshape_as(module.weight), scales, zeros)
    with torch.inference_mode():
        packed, record = fit_llama_awq_gsq_capture(layer, cache, initializers,
                                                  group_size=32, epochs=1, qk_steps=2)
    assert record['initializer'] == 'provided_awq'
    assert sum(isinstance(m, AwqTorchLinear) for m in packed.modules()) == 7
    assert cache.layer_inputs[0][0].is_inference()
    assert all(torch.equal(value, pristine[name]) for name, value in layer.state_dict().items())
    for module in packed.modules():
        if isinstance(module, AwqTorchLinear):
            module.scales = module.scales.float()
    assert torch.isfinite(packed(hidden, **kwargs)).all()
