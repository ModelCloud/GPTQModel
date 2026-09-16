import pytest
import torch

from gptqmodel.quantization.gsq_training import relaxed_scalar_weights, sampling_schedule


def test_masked_reconstruction_matches_unpadded_loss_and_gradient():
    from gptqmodel.quantization.gsq_training import reconstruction_stage_loss

    generator = torch.Generator().manual_seed(7)
    module = torch.nn.Linear(4, 3, bias=False).double()
    inputs = torch.randn(2, 5, 4, generator=generator, dtype=torch.float64)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, True]])
    inputs[~mask] = 1000
    weight = (module.weight.detach()+.1).requires_grad_()
    loss = reconstruction_stage_loss(module, (inputs,), {}, student_weights={'weight': weight}, output_mask=mask)
    expected = torch.nn.functional.mse_loss(torch.nn.functional.linear(inputs[mask], weight),
                                           module(inputs[mask]).detach())
    torch.testing.assert_close(loss, expected, rtol=1e-12, atol=1e-12)
    actual_gradient, = torch.autograd.grad(loss, weight)
    expected_gradient, = torch.autograd.grad(expected, weight)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
    for invalid in (torch.zeros_like(mask), mask.float(), mask[:, :-1]):
        with pytest.raises(ValueError, match='output mask'):
            reconstruction_stage_loss(module, (inputs,), {}, student_weights={'weight': weight}, output_mask=invalid)


def test_author_schedule_endpoints_and_single_update():
    assert sampling_schedule(0, 5) == (2., 10.)
    assert sampling_schedule(2, 5) == (1.25, 30.)
    assert sampling_schedule(4, 5) == (.5, 50.)
    assert sampling_schedule(0, 1) == (2., 10.)
    with pytest.raises(ValueError):
        sampling_schedule(5, 5)


def test_scalar_relaxation_assignment_and_scale_gradients():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(4, 2, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    scales = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    values = torch.tensor([-2., -1., 0., 1.], dtype=torch.float64)[:, None, None].expand_as(logits)
    uniform = torch.rand(logits.shape, generator=generator, dtype=torch.float64)
    groups = torch.tensor([0, 0, 1, 1])

    def apply(logits, scales):
        return relaxed_scalar_weights(logits, scales, values, groups, uniform=uniform,
                                      temperature=.7, multiplier=12.)
    assert torch.autograd.gradcheck(apply, (logits, scales))


def test_lion_scale_group_and_zero_gradient_momentum():
    from gptqmodel.quantization.gsq_training import GSQLion

    logits = torch.nn.Parameter(torch.tensor([1., -1.], dtype=torch.float64))
    scales = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = GSQLion([{'params': [logits], 'lr': .1, 'weight_decay': 1.},
                         {'params': [scales], 'lr': .01, 'weight_decay': 0.}])
    logits.grad = torch.tensor([2., -2.], dtype=torch.float64)
    scales.grad = torch.tensor([-1., 1.], dtype=torch.float64)
    optimizer.step()
    torch.testing.assert_close(logits, torch.tensor([.8, -.8], dtype=torch.float64))
    torch.testing.assert_close(scales, torch.tensor([1.01, .99], dtype=torch.float64))
    logits.grad.zero_()
    scales.grad = None
    optimizer.step()
    torch.testing.assert_close(logits, torch.tensor([.62, -.62], dtype=torch.float64))
    torch.testing.assert_close(scales, torch.tensor([1.01, .99], dtype=torch.float64))


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_author_candidate_initialization_and_grid_edges(bits):
    from gptqmodel.quantization.gsq_training import scalar_training_candidates

    low, high = -(2**(bits-1)), 2**(bits-1)-1
    weight = torch.tensor([[low, high, 0., 1.]])
    scales = torch.ones(1, 2)
    count = 4 if bits == 2 else 5
    result = scalar_training_candidates(weight, scales, 2, bits=bits, noise=torch.zeros(count, 1, 4))
    assert result['candidates'].shape == (count, 1, 4)
    selected = result['logits'].masked_fill(~result['valid'], -torch.inf).argmax(0, keepdim=True)
    hard = result['candidates'].gather(0, selected).squeeze(0)
    torch.testing.assert_close(hard, weight, rtol=0, atol=0)
    assert ((result['candidates'][result['valid']] >= low)
            & (result['candidates'][result['valid']] <= high)).all()
    if bits > 2:
        assert result['valid'][:, 0, 0].tolist() == [False, False, True, True, True]
        assert result['valid'][:, 0, 1].tolist() == [True, True, True, False, False]


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_trainable_module_mask_gradient_and_hard_export(bits):
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule, GSQLion

    weight = torch.tensor([[-(2**(bits-1)), 2**(bits-1)-1, 0., 1.]])
    count = 4 if bits == 2 else 5
    module = GSQScalarTrainingModule(weight, torch.ones(1, 2), 2, bits=bits,
                                    noise=torch.zeros(count, 1, 4))
    torch.testing.assert_close(module.hard_weight(), weight, rtol=0, atol=0)
    optimizer = GSQLion(module.optimizer_groups(assignment_lr=.001, scale_lr=.0001, weight_decay=1.))
    output = module(uniform=torch.full_like(module.logits, .5), temperature=1., multiplier=10.)
    (output-weight*.9).square().mean().backward()
    assert torch.isfinite(module.logits.grad).all()
    assert torch.isfinite(module.scales.grad).all()
    assert torch.count_nonzero(module.logits.grad[~module.valid]) == 0
    optimizer.step()
    assert torch.isfinite(module.hard_weight()).all()
    with torch.no_grad():
        module.logits[~module.valid] = 1e8
    hard = module.hard_weight()/module.scales[:, module.group_index]
    assert (hard >= -(2**(bits-1))).all() and (hard <= 2**(bits-1)-1).all()


def test_stage_loss_retains_nonlinear_interaction_and_freezes_dense_teacher():
    from gptqmodel.quantization.gsq_training import reconstruction_stage_loss

    module = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.SiLU(), torch.nn.Linear(6, 4))
    inputs = torch.randn(2, 3, 4)
    original = {key: value.detach().clone() for key, value in module.state_dict().items()}
    first = (module[0].weight.detach()*.9).requires_grad_()
    last = (module[2].weight.detach()*1.1).requires_grad_()
    loss = reconstruction_stage_loss(module, (inputs,), {}, student_weights={'0.weight': first, '2.weight': last})
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(inputs, first, module[0].bias.detach())),
        last, module[2].bias.detach())
    torch.testing.assert_close(loss, (expected-module(inputs).detach()).square().mean())
    loss.backward()
    assert first.grad is not None and last.grad is not None
    assert all(parameter.grad is None for parameter in module.parameters())
    assert all(torch.equal(value, module.state_dict()[key]) for key, value in original.items())


def test_stage_update_consumes_remainder_and_keeps_global_rng_private():
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule, GSQLion, train_stage_update

    module = GSQScalarTrainingModule(torch.ones(2, 4), torch.ones(2, 2), 2, bits=2,
                                    noise=torch.zeros(4, 2, 4))
    optimizer = GSQLion(module.optimizer_groups(assignment_lr=.001, scale_lr=.001, weight_decay=1.))
    seen = []

    def objective(batch, weights):
        seen.append(batch.shape[0])
        return (batch @ weights['weight'].T).square().mean()
    state = torch.random.get_rng_state().clone()
    result = train_stage_update({'weight': module}, optimizer,
                               [(torch.ones(2, 4), 4), (torch.ones(1, 4), 2)], objective,
                               generator=torch.Generator().manual_seed(7), temperature=1., multiplier=10.)
    assert seen == [2, 1]
    assert result >= 0
    assert torch.equal(state, torch.random.get_rng_state())
    assert optimizer.state[module.scales]['exp_avg'].abs().sum() > 0


def test_llama_attention_stage_matches_real_decoder_boundary_and_vo_gradients():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel.quantization.gsq_training import LlamaGSQAttentionStage, reconstruction_stage_loss

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    inputs = torch.randn(2, 5, 32)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    embeddings = LlamaRotaryEmbedding(config)(inputs, positions)
    mask = torch.full((5, 5), -torch.inf).triu(1)[None, None]
    kwargs = dict(position_embeddings=embeddings, attention_mask=mask, use_cache=False)
    captured = []
    handle = layer.post_attention_layernorm.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
    layer(inputs, **kwargs)
    handle.remove()
    stage = LlamaGSQAttentionStage(layer)
    torch.testing.assert_close(stage(inputs, **kwargs), captured[0], rtol=0, atol=0)
    weights = {f'self_attn.{name}.weight': (getattr(layer.self_attn, name).weight.detach()*.9).requires_grad_()
               for name in ('v_proj', 'o_proj')}
    loss = reconstruction_stage_loss(stage, (inputs,), kwargs, student_weights=weights)
    loss.backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in weights.values())
    assert all(value.grad is None for value in layer.parameters())


def test_llama_full_block_mlp_objective_restores_teacher_attention():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel.quantization.gsq_training import reconstruction_stage_loss

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    inputs = torch.randn(2, 5, 32)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    kwargs = dict(position_embeddings=LlamaRotaryEmbedding(config)(inputs, positions),
                  attention_mask=torch.full((5, 5), -torch.inf).triu(1)[None, None], use_cache=False)
    teacher_attention = {name: value.detach().clone() for name, value in layer.named_parameters()
                         if name.startswith('self_attn.')}
    with torch.no_grad():
        teacher = layer(inputs, **kwargs)
        layer.self_attn.v_proj.weight.mul_(.7)
        layer.self_attn.o_proj.weight.mul_(.8)
    weights = {f'mlp.{name}.weight': (getattr(layer.mlp, name).weight.detach()*.9).requires_grad_()
               for name in ('gate_proj', 'up_proj', 'down_proj')}
    original = {name: value.detach().clone() for name, value in layer.named_parameters()}
    loss = reconstruction_stage_loss(layer, (inputs,), kwargs, student_weights=weights,
                                     teacher_weights=teacher_attention)
    student = torch.func.functional_call(layer, weights, (inputs,), kwargs)
    torch.testing.assert_close(loss, (student-teacher).square().mean())
    loss.backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in weights.values())
    assert all(value.grad is None for value in layer.parameters())
    assert all(torch.equal(value, original[name]) for name, value in layer.named_parameters())


def test_llama_mlp_stage_matches_full_block_from_post_attention_boundary():
    from gptqmodel.quantization.gsq_training import (
        LlamaGSQAttentionStage,
        LlamaGSQMLPStage,
    )
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaRotaryEmbedding,
    )

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    inputs = torch.randn(2, 5, 32)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    kwargs = {
        'position_embeddings': LlamaRotaryEmbedding(config)(inputs, positions),
        'attention_mask': torch.full((5, 5), -torch.inf).triu(1)[None, None],
        'use_cache': False,
    }
    with torch.no_grad():
        post_attention = LlamaGSQAttentionStage(layer)(inputs, **kwargs)
        staged = LlamaGSQMLPStage(layer)(post_attention)
        reference = layer(inputs, **kwargs)
    torch.testing.assert_close(staged, reference, rtol=0, atol=0)


@pytest.mark.parametrize('decay', ['linear', 'cosine', 'constant'])
def test_author_learning_rate_warmup_and_final_update(decay):
    from gptqmodel.quantization.gsq_training import stage_learning_rate

    kwargs = dict(base_lr=.001, warmup_steps=2, min_lr=.1, decay=decay)
    assert stage_learning_rate(0, 6, **kwargs) == pytest.approx(.0001)
    assert stage_learning_rate(1, 6, **kwargs) == pytest.approx(.00055)
    assert stage_learning_rate(2, 6, **kwargs) == pytest.approx(.001)
    assert stage_learning_rate(5, 6, **kwargs) == pytest.approx(.001 if decay == 'constant' else .0001)


def test_complete_stage_driver_schedule_export_and_determinism():
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule, fit_reconstruction_stage

    def run():
        quantizer = GSQScalarTrainingModule(torch.ones(2, 4), torch.ones(2, 2), 2, bits=2,
                                           noise=torch.zeros(4, 2, 4))
        batches = [[(torch.ones(2, 4), 4)], [(torch.ones(1, 4)*.5, 2)]]

        def objective(batch, weights):
            return (batch @ weights['weight'].T).square().mean()
        return fit_reconstruction_stage({'weight': quantizer}, batches, objective, epochs=2, seed=7)
    first, second = run(), run()
    assert first['history'] == second['history']
    assert len(first['history']) == 4
    assert first['hard_loss_before'] >= 0
    assert first['hard_loss_after'] >= 0
    assert first['hard_loss_delta'] == pytest.approx(first['hard_loss_after']-first['hard_loss_before'])
    assert first['history'][0]['temperature'] == 2.
    assert first['history'][-1]['temperature'] == .5
    assert first['history'][-1]['multiplier'] == 50.
    assert first['history'][-1]['learning_rates'] == [0., 0.]
    torch.testing.assert_close(first['weights']['weight'], second['weights']['weight'], rtol=0, atol=0)
    assert torch.isfinite(first['weights']['weight']).all()


def test_stage_driver_restores_nonregressing_disjoint_validation_checkpoint():
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule, fit_reconstruction_stage

    quantizer = GSQScalarTrainingModule(
        torch.ones(1, 1), torch.ones(1, 1), 1, bits=2,
        noise=torch.zeros(4, 1, 1),
    )
    train = [[(torch.tensor(-2.), 1)]]
    heldout = [[(torch.tensor(1.), 1)]]

    def objective(target, weights):
        return (weights['weight'].squeeze()-target).square()

    result = fit_reconstruction_stage(
        {'weight': quantizer}, train, objective, epochs=4, seed=7,
        assignment_lr=.1, scale_lr=.01, temperature=(2., .05),
        multiplier=(100., 500.), validation_batches=heldout, restore_best=True,
    )
    assert result['validation_hard_loss_after'] <= result['validation_hard_loss_before']
    assert result['best_validation_hard_loss'] == result['validation_hard_loss_after']
    assert result['restored_best_validation_checkpoint'] is True
    assert len(result['validation_history']) == 4
    with pytest.raises(ValueError, match='validation batches'):
        fit_reconstruction_stage(
            {'weight': quantizer}, train, objective, epochs=1, restore_best=True,
        )


def test_checkpoint_global_guard_requires_all_metrics_to_avoid_regression():
    from scripts.validate_qvq_gsq_checkpoint import strict_global_guard

    incumbent = {
        'forward_kld': 2., 'logit_mse': 4., 'cross_entropy': 3.,
        'perplexity': 20., 'top1_agreement': .8,
    }
    candidate = {
        'forward_kld': 1., 'logit_mse': 3., 'cross_entropy': 2.9,
        'perplexity': 19., 'top1_agreement': .81,
    }
    accepted, improvements = strict_global_guard(incumbent, candidate)
    assert accepted is True
    assert improvements['forward_kld'] == 50.
    candidate['top1_agreement'] = .79
    accepted, improvements = strict_global_guard(incumbent, candidate)
    assert accepted is False
    assert improvements['top1_agreement_points'] < 0.


def test_llama_staged_driver_executes_all_projections_without_mutating_teacher(monkeypatch):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    import gptqmodel.quantization.gsq_training as gsq_training

    stage_teacher_overrides = []
    mlp_teacher_overrides = []
    original_stage_loss = gsq_training.reconstruction_stage_loss

    def observe_mlp_teacher(module, args, kwargs, *, student_weights, teacher_weights=None, **options):
        stage_teacher_overrides.append(teacher_weights)
        if set(student_weights) == {
                'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'}:
            mlp_teacher_overrides.append(teacher_weights)
        return original_stage_loss(module, args, kwargs, student_weights=student_weights,
                                   teacher_weights=teacher_weights, **options)

    monkeypatch.setattr(gsq_training, 'reconstruction_stage_loss', observe_mlp_teacher)

    config = LlamaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2,
                         num_key_value_heads=1, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    original = {name: value.detach().clone() for name, value in layer.named_parameters()}
    initializers = {}
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            scales = torch.full((module.out_features, module.in_features//8), .1)
            initializers[name] = ((module.weight.detach()/.1).round().clamp(-2, 1)*.1, scales)
    hidden = torch.randn(1, 3, 16)
    kwargs = dict(position_embeddings=LlamaRotaryEmbedding(config)(hidden, torch.arange(3)[None]),
                  attention_mask=torch.full((3, 3), -torch.inf).triu(1)[None, None], use_cache=False)
    fitted, records = gsq_training.fit_llama_stages(layer, initializers, [(hidden, kwargs)], bits=2, group_size=8,
                                      epochs=2, qk_steps=2, reinitialize_mlp=False, decay='constant')
    assert set(records) == {'self_attn.q_proj', 'self_attn.k_proj', 'attention', 'mlp'}
    assert all(len(result['history']) == 2 for result in records.values())
    assert torch.isfinite(fitted(hidden, **kwargs)).all()
    assert all(torch.equal(value, original[name]) for name, value in layer.named_parameters())
    assert stage_teacher_overrides and all(value is None for value in stage_teacher_overrides)
    assert mlp_teacher_overrides and all(value is None for value in mlp_teacher_overrides)


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_actual_gptq_initializer_captures_llama_inputs_and_preserves_teacher(bits):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel.quantization.gsq_training import initialize_llama_gptq

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    original = {key: value.clone() for key, value in layer.state_dict().items()}
    hidden = torch.randn(2, 16, 32)
    kwargs = dict(position_embeddings=LlamaRotaryEmbedding(config)(hidden, torch.arange(16)[None]),
                  attention_mask=torch.full((16, 16), -torch.inf).triu(1)[None, None], use_cache=False)
    seeds, metadata = initialize_llama_gptq(layer, [(hidden, kwargs)], bits=bits, group_size=32)
    assert len(seeds) == len(metadata) == 7
    for name, (weight, scales) in seeds.items():
        assert weight.shape == layer.get_submodule(name).weight.shape
        assert torch.isfinite(weight).all() and torch.isfinite(scales).all()
    assert all(torch.equal(value, original[key]) for key, value in layer.state_dict().items())


def test_gptq_staged_training_packing_and_disk_reload(tmp_path):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization.gsq_training import initialize_llama_gptq, fit_llama_stages

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    hidden = torch.randn(2, 16, 32)
    kwargs = dict(position_embeddings=LlamaRotaryEmbedding(config)(hidden, torch.arange(16)[None]),
                  attention_mask=torch.full((16, 16), -torch.inf).triu(1)[None, None], use_cache=False)
    batches = [(hidden, kwargs)]
    seeds, _ = initialize_llama_gptq(layer, batches, bits=4, group_size=32)
    fitted, records = fit_llama_stages(
        layer, seeds, batches, bits=4, group_size=32, epochs=2, qk_steps=2, decay='constant',
    )
    assert records['mlp']['initializer_timing'] == 'after_attention'
    from gptqmodel.quantization.gsq_training import pack_llama_staged_block
    exported = pack_llama_staged_block(fitted, records, bits=4, group_size=32)
    assert all(isinstance(exported.get_submodule(name), TorchLinear) for name in seeds)
    assert torch.isfinite(exported(hidden, **kwargs)).all()

    for name in seeds:
        if name in records:
            scales = records[name]['scales']['weight']
        else:
            stage = 'attention' if name.startswith('self_attn') else 'mlp'
            scales = records[stage]['scales'][name+'.weight']
        linear = fitted.get_submodule(name)
        groups = torch.arange(linear.in_features, dtype=torch.int32)//32

        def container():
            return TorchLinear(bits=4, group_size=32, sym=True, desc_act=False,
                               in_features=linear.in_features, out_features=linear.out_features,
                               bias=False, backend=BACKEND.TORCH)
        packed = container()
        packed.pack_original(linear, scales, torch.full_like(scales, 8), groups)
        path = tmp_path / (name+'.pt')
        torch.save(packed.state_dict(), path)
        restored = container()
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        codes, zeros = restored._unpack_continuous_codes()
        decoded = (restored.scales.float()[groups]*(codes.float()-zeros.float()[groups])).T
        assignments = (linear.weight.detach()/scales[:, groups]).round()
        expected = assignments*restored.scales.float().T[:, groups]
        torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
        inputs = torch.randn(3, linear.in_features).half()
        torch.testing.assert_close(restored(inputs), packed(inputs), rtol=0, atol=0)


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_signed_learned_scale_preserves_scalar_packing(bits):
    from gptqmodel import BACKEND
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    columns, rows = 64, 32
    groups = torch.arange(columns, dtype=torch.int32)//32
    scales = torch.full((rows, 2), .015625)
    scales[:, 1] = -.0078125
    assignments = (torch.arange(rows*columns).reshape(rows, columns) % (2**bits))-2**(bits-1)
    weight = assignments*scales[:, groups]
    linear = torch.nn.Linear(columns, rows, bias=False)
    linear.weight.data.copy_(weight)
    packed = TorchLinear(bits=bits, group_size=32, sym=True, desc_act=False, in_features=columns,
                         out_features=rows, bias=False, backend=BACKEND.TORCH)
    packed.pack_original(linear, scales, torch.full_like(scales, 2**(bits-1)), groups)
    codes, zeros = packed._unpack_continuous_codes()
    decoded = (packed.scales.float()[groups]*(codes.float()-zeros.float()[groups])).T
    torch.testing.assert_close(decoded, weight, rtol=0, atol=0)
    inputs = torch.ones(2, columns, dtype=torch.float16)
    torch.testing.assert_close(packed(inputs).float(), inputs.float() @ weight.T, rtol=0, atol=0)


def test_qk_factor_matches_author_sequence_normalization_and_damping():
    from gptqmodel.quantization.gsq_training import prepare_qk_calibration_factor

    batches = [torch.randn(2, 5, 4), torch.randn(1, 3, 4)]
    for batch in batches:
        batch[..., -1] = 0
    factor, dead = prepare_qk_calibration_factor(batches, damp_percent=.1)
    gram = sum(batch.reshape(-1, 4).T @ batch.reshape(-1, 4) for batch in batches)*2/3
    gram[-1, -1] = 1
    gram += torch.eye(4)*(.1*gram.diagonal().mean())
    torch.testing.assert_close(factor @ factor.T, gram)
    assert dead.tolist() == [False, False, False, True]


def test_hard_export_rounds_scales_before_bf16_multiplication():
    from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule

    weight = torch.tensor([[-4., -3., 2., 3.]], dtype=torch.bfloat16)
    module = GSQScalarTrainingModule(weight, torch.ones(1, 2, dtype=torch.bfloat16), 2,
                                    bits=3, noise=torch.zeros(5, 1, 4, dtype=torch.bfloat16))
    with torch.no_grad():
        module.scales.copy_(torch.tensor([[.123456, .876543]]))
    expected = weight*module.scales[:, module.group_index].bfloat16()
    assert module.hard_weight().dtype == torch.bfloat16
    torch.testing.assert_close(module.hard_weight(), expected, rtol=0, atol=0)
