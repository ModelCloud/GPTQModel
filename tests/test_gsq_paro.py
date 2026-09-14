import pytest
import torch

from gptqmodel.quantization.gsq_paro import paro_gsq_basis


@pytest.mark.parametrize('group_size', [16, 32, -1])
@pytest.mark.parametrize('rotations', [0, 3])
def test_exported_paro_basis_preserves_dense_reconstruction(group_size, rotations):
    rng = torch.Generator().manual_seed(7)
    width = 64
    group = width if group_size == -1 else group_size
    pairs = torch.stack([torch.cat([torch.randperm(group, generator=rng) for _ in range(width // group)])
                         for _ in range(rotations)]) if rotations else torch.empty(0, width, dtype=torch.long)
    theta = torch.randn(rotations, width // 2, generator=rng, dtype=torch.float64)
    scales = torch.rand(width, generator=rng, dtype=torch.float64) + .5
    weight = torch.randn(13, width, generator=rng, dtype=torch.float64)
    inputs = torch.randn(23, width, generator=rng, dtype=torch.float64)
    teacher, features = paro_gsq_basis(weight, inputs, pairs, theta, scales, group_size=group_size)

    # Independent dense rotation product using the stored FP16 metadata.
    rotation = torch.eye(width, dtype=torch.float64)
    for stage in range(rotations):
        step = torch.eye(width, dtype=torch.float64)
        for index in range(width // 2):
            offset = (2 * index // group) * group
            i, j = pairs[stage, 2*index:2*index+2] + offset
            angle = theta[stage, index].half().double()
            step[i, i] = step[j, j] = angle.cos()
            step[i, j], step[j, i] = -angle.sin(), angle.sin()
        rotation = rotation @ step
    stored_scales = scales.half().double()
    torch.testing.assert_close(teacher, (weight / stored_scales) @ rotation, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(features, (inputs * stored_scales) @ rotation, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(features @ teacher.T, inputs @ weight.T, rtol=1e-12, atol=1e-12)

    error = torch.randn(weight.shape, generator=rng, dtype=torch.float64)
    original_error = (error @ rotation.T) * stored_scales
    torch.testing.assert_close((features @ error.T).square().sum(),
                               (inputs @ original_error.T).square().sum(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('bad_scale', [0., -1., 1e-12, float('inf')])
def test_paro_basis_rejects_invalid_export_scales(bad_scale):
    with pytest.raises(ValueError, match='finite|positive'):
        paro_gsq_basis(torch.ones(2, 16), torch.ones(3, 16), torch.arange(16).view(1, -1),
                       torch.zeros(1, 8), torch.full((16,), bad_scale), group_size=16)


def test_paro_basis_rejects_overlapping_pairs():
    with pytest.raises(ValueError, match='disjoint'):
        paro_gsq_basis(torch.ones(2, 16), torch.ones(3, 16), torch.zeros(1, 16, dtype=torch.long),
                       torch.zeros(1, 8), torch.ones(16), group_size=16)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_paro_basis_accumulates_low_precision_inputs_in_fp32(dtype):
    rng = torch.Generator().manual_seed(7)
    weight = torch.randn(13, 32, generator=rng).to(dtype)
    inputs = torch.randn(23, 32, generator=rng).to(dtype)
    pairs = torch.arange(16).repeat(2).reshape(1, 32).to(torch.int16)
    theta = torch.randn(1, 16, generator=rng)
    scales = torch.rand(32, generator=rng) + .5
    snapshots = [v.clone() for v in (weight, inputs, pairs, theta, scales)]
    teacher, features = paro_gsq_basis(weight, inputs, pairs, theta, scales, group_size=16)
    assert teacher.dtype == features.dtype == torch.float32
    torch.testing.assert_close(features @ teacher.T, inputs.float() @ weight.float().T, rtol=2e-5, atol=2e-5)
    for original, snapshot in zip((weight, inputs, pairs, theta, scales), snapshots, strict=True):
        assert torch.equal(original, snapshot)


@pytest.mark.parametrize('learn_scales', [False, True])
def test_paro_gsq_export_improves_and_matches_actual_awq_packing(learn_scales):
    from types import SimpleNamespace
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.config import GSQConfig
    from gptqmodel.quantization.gsq_paro import refine_paro_export
    from gptqmodel.quantization.paroquant.optimization import _apply_inverse_rotation
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm

    rng = torch.Generator().manual_seed(7)
    width, outputs, group = 32, 8, 16
    pairs = torch.arange(group).repeat(2).reshape(1, width).short()
    theta = torch.full((1, width//2), .2).half()
    channel = (torch.rand(width, generator=rng) + .5).half()
    target = torch.full((outputs, width), .125)
    teacher = _apply_inverse_rotation(target, pairs, theta.float(), group_size=group,
                                     fused_rotation=False) * channel.float()
    inputs = torch.randn(128, width, generator=rng)
    result = SimpleNamespace(pack_weight=torch.zeros_like(target).half(), pseudo_weight=torch.zeros_like(target),
                             q_scales=torch.full((outputs, 2), .125).half(),
                             q_zeros=torch.full((outputs, 2), 8.), pairs=pairs, theta=theta, channel_scales=channel)
    fitted = refine_paro_export(result, teacher=teacher, inputs=inputs, group_size=group,
                               config=GSQConfig(enabled=True, learn_scales=learn_scales, steps=80, learning_rate=.2))
    assert fitted['after'] < fitted['before']
    linear = torch.nn.Linear(width, outputs, bias=False, dtype=torch.float16)
    linear.weight.data.copy_(fitted['pack_weight'])
    packed = AwqTorchLinear(bits=4, group_size=group, sym=False, desc_act=False,
                            in_features=width, out_features=outputs, bias=False, register_buffers=True)
    packed.pack(linear, fitted['q_scales'], fitted['q_zeros'])
    decoded = dequantize_gemm(packed.qweight, packed.qzeros, packed.scales.float(), 4, group).T
    _, features = paro_gsq_basis(teacher, inputs, pairs, theta, channel, group_size=group)
    torch.testing.assert_close(inputs @ fitted['pseudo_weight'].T, features @ decoded.T, rtol=2e-5, atol=2e-5)
    explicit = ((inputs @ fitted['pseudo_weight'].T - inputs @ teacher.T).square().sum() /
                (inputs @ teacher.T).square().sum()).item()
    assert abs(explicit - fitted['after']) <= max(1e-7, abs(explicit) * 1e-4)
    assert torch.equal(result.pack_weight, torch.zeros_like(result.pack_weight))
    assert torch.equal(result.q_scales, torch.full_like(result.q_scales, .125))


@pytest.mark.parametrize('config', [None, {'enabled': False}])
def test_paro_gsq_disabled_keeps_export_exact_without_calibration(config):
    from types import SimpleNamespace
    from gptqmodel.quantization.gsq_paro import refine_paro_export

    state = SimpleNamespace(**{key: torch.randn(2, 3) for key in
                              ('pack_weight', 'pseudo_weight', 'q_scales', 'q_zeros')})
    fitted = refine_paro_export(state, teacher=None, inputs=None, group_size=16, config=config)
    for key in vars(state):
        assert torch.equal(fitted[key], getattr(state, key))
    assert fitted['before'] is fitted['after'] is None


def test_paro_gsq_exact_baseline_retains_original_export():
    from types import SimpleNamespace
    from gptqmodel.quantization.gsq_paro import refine_paro_export

    state = SimpleNamespace(pack_weight=torch.zeros(8, 32).half(), pseudo_weight=torch.zeros(8, 32),
                             q_scales=torch.ones(8, 2).half(), q_zeros=torch.full((8, 2), 8.),
                             pairs=torch.empty(0, 32, dtype=torch.int16), theta=torch.empty(0, 16),
                             channel_scales=torch.ones(32))
    snapshots = {key: value.clone() for key, value in vars(state).items()}
    fitted = refine_paro_export(state, teacher=torch.zeros(8, 32), inputs=torch.eye(32),
                               group_size=16, config={'enabled': True, 'steps': 3})
    assert fitted['before'] == fitted['after'] == 0
    for key in ('pack_weight', 'pseudo_weight', 'q_scales', 'q_zeros'):
        assert torch.equal(fitted[key], snapshots[key])
    for key, value in snapshots.items():
        assert torch.equal(getattr(state, key), value)


@pytest.mark.parametrize('gsq', [None, {'enabled': False}, {'enabled': True, 'learn_scales': True}])
def test_paro_gsq_config_roundtrip(gsq):
    from gptqmodel.quantization.config import ParoConfig, QuantizeConfig

    config = ParoConfig(gsq=gsq)
    restored = QuantizeConfig.from_quant_config(config.to_dict())
    assert isinstance(restored, ParoConfig)
    assert restored.gsq == config.gsq


@pytest.mark.parametrize('scope', ['layer', 'compute_block'])
def test_paro_gsq_grouped_config_roundtrip_with_paired_inputs(scope):
    from gptqmodel.quantization.config import ParoConfig

    from gptqmodel.quantization.config import QuantizeConfig

    cfg = ParoConfig(gsq={'enabled': True}, opt_scope=scope)
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert restored.opt_scope == scope and restored.gsq.enabled
    paired = ParoConfig(gsq={'enabled': True}, opt_scope=scope, opt_train_on_noisy_inputs=True)
    paired_restored = QuantizeConfig.from_quant_config(paired.to_dict())
    assert paired_restored.gsq.enabled and paired_restored.opt_train_on_noisy_inputs
    assert paired_restored.opt_scope == scope
    assert ParoConfig(gsq={'enabled': False}, opt_scope=scope).gsq.enabled is False


@pytest.mark.parametrize('explicit_validation', [False, True])
def test_paro_processor_gsq_uses_only_training_rows(monkeypatch, explicit_validation):
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig
    import gptqmodel.quantization.gsq_paro as adapter

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq={'enabled': True}, group_size=16,
                                opt_train_samples=4, opt_validation_samples=2)
    module = SimpleNamespace(full_name='model.layers.0.self_attn.q_proj', state={})
    rows = torch.arange(6).float().view(-1, 1).expand(-1, 16)
    validation = torch.full((2, 16), 999.) if explicit_validation else None
    result = SimpleNamespace(train_loss=3., val_loss=4.)
    observed = []

    def fit(export, **kwargs):
        observed.append(kwargs['inputs'].clone())
        return {'before': 1., 'after': 1., 'history': [1.]}

    monkeypatch.setattr(adapter, 'refine_paro_export', fit)
    returned = processor._refine_gsq_export(module, result, torch.ones(8, 16), rows, validation)
    assert returned is result
    expected = rows[[0, 2, 3, 5]] if explicit_validation else rows[:4]
    assert torch.equal(observed[0], expected)
    assert module.state['gsq_diagnostics']['initializer_val_loss'] == 4.
    assert module.state['gsq_diagnostics']['validation_rows'] == 2


def test_paro_processor_gsq_unmatched_module_bypasses_without_data():
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq={'enabled': True, 'modules': ['v_proj$']})
    module = SimpleNamespace(full_name='q_proj', state={})
    result = object()
    assert processor._refine_gsq_export(module, result, None, None, None) is result
    assert not module.state


def test_paro_processor_applies_improved_export_and_updates_replay_losses():
    import threading
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig
    from gptqmodel.quantization.paroquant.optimization import ParoQuantOptimizationResult

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(group_size=16, opt_train_samples=64, opt_validation_samples=16,
                                gsq={'enabled': True, 'steps': 80, 'learning_rate': .2})
    processor.lock = threading.Lock()
    processor.calculate_w_wq_diff = False
    layer = torch.nn.Linear(32, 8, bias=False, dtype=torch.float16)
    layer.weight.data.fill_(.125)
    module = NamedModule(layer, name='q_proj', full_name='model.layers.0.self_attn.q_proj', layer_index=0)
    original = layer.weight.detach().clone()
    result = ParoQuantOptimizationResult(
        pseudo_weight=torch.zeros(8, 32), pack_weight=torch.zeros(8, 32).half(),
        q_scales=torch.full((8, 2), .125).half(), q_zeros=torch.full((8, 2), 8.),
        pairs=torch.empty(0, 32, dtype=torch.int16), theta=torch.empty(0, 16),
        channel_scales=torch.ones(32), train_loss=3., val_loss=4., used_identity=True)
    rng = torch.Generator().manual_seed(7)
    train = torch.randn(64, 32, generator=rng)
    validation = torch.randn(16, 32, generator=rng)
    updated = processor._refine_gsq_export(module, result, original, train, validation)
    assert updated is not result
    processor._apply_optimization_result(module, updated, original)
    assert torch.equal(module.weight, updated.pseudo_weight.half())
    assert torch.equal(module.state['pack_weight'], updated.pack_weight.half())
    assert torch.equal(module.state['theta'], result.theta.half())
    assert torch.equal(module.state['channel_scales'], result.channel_scales.half())
    expected = torch.nn.functional.smooth_l1_loss(validation @ module.weight.float().T,
                                                 validation @ original.float().T).item()
    assert updated.val_loss == pytest.approx(expected)
    assert module.state['gsq_diagnostics']['initializer_val_loss'] == 4.
    assert module.state['gsq_diagnostics']['after'] < module.state['gsq_diagnostics']['before']


@pytest.mark.parametrize('tokens', [1, 17])
@pytest.mark.parametrize('learn_scales', [False, True])
@pytest.mark.parametrize('krot', [1, 8])
def test_paro_gsq_native_reload_and_graph(tokens, learn_scales, krot, tmp_path):
    import os
    from types import SimpleNamespace
    if not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
        pytest.skip('requires an exclusive GPU lease')
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
    from gptqmodel.quantization.gsq_paro import refine_paro_export
    from gptqmodel.quantization.paroquant.optimization import _apply_inverse_rotation

    rng = torch.Generator().manual_seed(7)
    width = 128
    pairs = torch.arange(width).view(1, -1).repeat(krot, 1).short()
    theta = torch.full((krot, width//2), .2).half()
    channel = (torch.rand(width, generator=rng) + .5).half()
    target = torch.full((width, width), .03125)
    teacher = _apply_inverse_rotation(target, pairs, theta.float(), group_size=width,
                                     fused_rotation=False) * channel.float()
    result = SimpleNamespace(pack_weight=torch.zeros_like(target).half(), pseudo_weight=torch.zeros_like(target),
                             q_scales=torch.full((width, 1), .03125).half(), q_zeros=torch.full((width, 1), 8.),
                             pairs=pairs, theta=theta, channel_scales=channel)
    fitted = refine_paro_export(result, teacher=teacher, inputs=torch.randn(128, width, generator=rng),
                               group_size=width, config={'enabled': True, 'steps': 80, 'learning_rate': .2,
                                                        'learn_scales': learn_scales})
    assert fitted['after'] < fitted['before']
    kwargs = dict(bits=4, group_size=width, sym=True, desc_act=False, in_features=width,
                  out_features=width, bias=False, register_buffers=True, krot=krot)
    packed = ParoLinear(**kwargs)
    linear = torch.nn.Linear(width, width, bias=False, dtype=torch.float16)
    linear.weight.data.copy_(fitted['pack_weight'])
    packed.pack(linear, fitted['q_scales'], fitted['q_zeros'])
    packed.pairs.copy_(pairs)
    packed.theta.copy_(theta)
    packed.channel_scales.copy_(channel.reshape_as(packed.channel_scales))
    path = tmp_path / 'paro.pt'
    torch.save(packed.state_dict(), path)
    restored = ParoLinear(**kwargs)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in packed.state_dict().items())
    restored = restored.cuda().eval()
    restored.post_init()
    x = torch.randn(tokens, width, generator=rng).half().cuda()
    reference_weight = fitted['pseudo_weight'].cuda().float()

    def check(actual, inputs):
        delta = (actual.float() - inputs.float() @ reference_weight.T).abs()
        assert torch.isfinite(actual).all()
        assert delta.mean() <= .002
        assert delta.max() <= .046875
        print('PARO_GSQ_NATIVE', tokens, learn_scales, krot, float(delta.mean()), float(delta.max()), flush=True)

    for _ in range(3):
        output = restored(x)
    check(output, x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = restored(x)
    for _ in range(3):
        x.copy_(torch.randn(tokens, width, generator=rng).half())
        graph.replay()
        torch.cuda.synchronize()
        check(captured, x)
        torch.testing.assert_close(captured, restored(x), rtol=0, atol=0)


@pytest.mark.parametrize('lengths', [(2, 3), (1,), (1, 2, 1)])
def test_paro_gsq_implicit_feature_streams_reserve_disjoint_sequences(lengths):
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq={'enabled': True}, opt_train_samples=2048, opt_validation_samples=64)
    offset = 0
    tensors = []
    for length in lengths:
        tensors.append(torch.arange(offset, offset + length).reshape(length, 1, 1).float())
        offset += length
    train, validation = processor._module_feature_streams(tensors, [])
    assert train.numel() > 0
    assert not set(train.flatten().tolist()) & set(validation.flatten().tolist())
    assert train.numel() + validation.numel() == offset
    if offset > 1:
        assert validation.numel() == 1


@pytest.mark.parametrize('config', [None, {'enabled': False}, {'enabled': True, 'modules': ['k_proj$']}])
def test_paro_implicit_stream_selection_unchanged_without_matching_gsq(config):
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq=config)
    tensors = [torch.arange(6).reshape(2, 3, 1).float()]
    train, validation = processor._module_feature_streams(tensors, [], 'model.layers.0.self_attn.q_proj')
    assert torch.equal(train, tensors[0])
    assert torch.equal(validation, tensors[0])


def test_paro_gsq_layer_capture_routes_full_module_filter():
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq={'enabled': True, 'modules': ['self_attn.q_proj$']})
    values = torch.arange(12).reshape(2, 3, 2).float()
    processor.tasks = {name: {'inputs': [values.clone()], 'batch_indices': [0]} for name in ('q', 'k')}
    state = SimpleNamespace(modules={
        name: SimpleNamespace(full_name=f'model.layers.0.self_attn.{name}_proj') for name in ('q', 'k')})
    processor._layer_input_features(state)
    q, k = processor.tasks['q'], processor.tasks['k']
    assert torch.equal(q['train_inputs'], values[:1])
    assert torch.equal(q['validation_inputs'], values[1:])
    assert torch.equal(k['train_inputs'], values)
    assert torch.equal(k['validation_inputs'], values)


@pytest.mark.parametrize('scope', ['layer', 'compute_block'])
def test_paro_group_gsq_preserves_initializer_loss_scope(monkeypatch, scope):
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq={'enabled': True}, opt_scope=scope)
    train, validation = torch.ones(3, 16), torch.full((2, 16), 9.)
    processor.tasks = {'q': {'train_inputs': train, 'validation_inputs': validation}}
    module = SimpleNamespace(name='q', full_name='q_proj', state={})
    original, updated = object(), object()

    def refine(module, result, weight, inputs, check_inputs):
        assert result is original
        assert inputs is train and check_inputs is validation
        module.state['gsq_diagnostics'] = {'before': 2., 'after': 1.}
        return updated

    monkeypatch.setattr(processor, '_refine_gsq_export', refine)
    assert processor._refine_group_gsq_export(module, original, None, 7.) is updated
    stats = module.state['gsq_diagnostics']
    assert stats['initializer_group_val_loss'] == 7.
    assert stats['initializer_scope'] == scope
    assert stats['refinement_scope'] == 'module'
    assert stats['group_loss_recomputed'] is False


@pytest.mark.parametrize('scope', ['layer', 'compute_block'])
def test_paro_group_lifecycle_applies_actual_gsq_export(scope, monkeypatch):
    import threading
    from types import SimpleNamespace
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig
    from gptqmodel.quantization.paroquant.optimization import ParoQuantOptimizationResult

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(group_size=16, opt_scope=scope, offload_to_disk=False,
                                opt_train_samples=64, opt_validation_samples=16,
                                gsq={'enabled': True, 'steps': 80, 'learning_rate': .2})
    processor.lock = threading.Lock()
    processor.fallback = False
    processor.calculate_w_wq_diff = False
    processor._has_explicit_validation_calibration = True
    processor._train_calibration_batch_count = 1
    processor._validation_calibration_batch_count = 1
    linear = torch.nn.Linear(32, 8, bias=False, dtype=torch.float16)
    linear.weight.data.fill_(.125)
    module = NamedModule(linear, 'self_attn.q_proj', 'model.layers.0.self_attn.q_proj', 0)
    module.state['module_tree_flags'] = frozenset({'q'})
    rng = torch.Generator().manual_seed(7)
    train, validation = torch.randn(1, 64, 32, generator=rng), torch.randn(1, 16, 32, generator=rng)
    processor.tasks = {module.name: {'inputs': [train, validation], 'batch_indices': [0, 1], 'layer_index': 0}}
    baseline = ParoQuantOptimizationResult(
        pseudo_weight=torch.zeros(8, 32), pack_weight=torch.zeros(8, 32).half(),
        q_scales=torch.full((8, 2), .125).half(), q_zeros=torch.full((8, 2), 8.),
        pairs=torch.empty(0, 32, dtype=torch.int16), theta=torch.empty(0, 16),
        channel_scales=torch.ones(32), train_loss=3., val_loss=4., used_identity=True)
    monkeypatch.setattr(processor, '_optimize_group', lambda state, modules: ({module.name: baseline}, 4.))
    logged = []
    monkeypatch.setattr(processor, '_log_quant_result', lambda *args: logged.append(args))
    state = SimpleNamespace(quantized=False, modules={module.name: module}, layer_inputs=[[train]],
                            layer_outputs=[[train]], pending_modules=set(), processed_subsets={0}, subset_total=1)
    processor._quantize_layer(0, state)
    stats = module.state['gsq_diagnostics']
    assert stats['after'] < stats['before']
    assert stats['train_rows'] == 64 and stats['validation_rows'] == 16
    assert stats['initializer_group_val_loss'] == 4.
    assert torch.count_nonzero(module.state['pack_weight']) > 0
    assert torch.count_nonzero(module.weight) > 0
    assert logged[0][2] == 4.
    assert state.quantized and not state.modules


@pytest.mark.parametrize('missing', ['train', 'validation', 'validation_none'])
def test_paro_group_gsq_does_not_replace_missing_explicit_calibration(missing):
    from types import SimpleNamespace
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.quantization.config import ParoConfig

    processor = object.__new__(ParoQuantProcessor)
    processor.qcfg = ParoConfig(gsq={'enabled': True}, opt_scope='layer')
    processor._has_explicit_validation_calibration = True
    entry = {'train_inputs': torch.ones(4, 16), 'validation_inputs': torch.ones(2, 16)}
    if missing == 'validation_none':
        entry['validation_inputs'] = None
    else:
        entry[missing + '_inputs'] = torch.empty(0)
    processor.tasks = {'q': entry}
    module = SimpleNamespace(name='q', full_name='q_proj', state={})
    with pytest.raises(RuntimeError, match='explicit training and validation'):
        processor._refine_group_gsq_export(module, object(), None, 1.)
    assert not module.state


def test_paro_paired_objective_matches_direct_clean_target_loss(monkeypatch):
    from types import SimpleNamespace
    from gptqmodel.quantization.gsq_paro import refine_paro_export
    import gptqmodel.quantization.gsq_scalar as scalar

    rng = torch.Generator().manual_seed(7)
    width = 16
    teacher = torch.randn(8, width, generator=rng)
    noisy = torch.randn(27, width, generator=rng)
    clean = noisy + .2 * torch.randn(27, width, generator=rng)
    result = SimpleNamespace(pack_weight=torch.zeros_like(teacher).half(), pseudo_weight=torch.zeros_like(teacher),
                             q_scales=torch.ones(8, 1).half(), q_zeros=torch.full((8, 1), 8.),
                             pairs=torch.arange(width).reshape(1, width), theta=torch.full((1, width // 2), .3),
                             channel_scales=torch.linspace(.5, 1.5, width))
    observed = []

    def fit(*args, **kwargs):
        target, x, cross = kwargs['target'], kwargs['inputs'], kwargs['cross_moment']
        _, y = paro_gsq_basis(teacher, clean, result.pairs, result.theta, result.channel_scales, group_size=width)
        candidate = target + torch.randn(target.shape, generator=rng) * .1
        error = candidate - target
        quadratic = (x @ error.T).square().sum() - 2 * (error * (target @ cross)).sum()
        direct = (x @ candidate.T - y @ target.T).square().sum()
        constant = ((x-y) @ target.T).square().sum()
        torch.testing.assert_close(quadratic, direct-constant, rtol=2e-5, atol=2e-5)
        observed.append(True)
        return SimpleNamespace(before=1., after=1., history=[1.])

    monkeypatch.setattr(scalar, 'refine_affine_scalar', fit)
    refine_paro_export(result, teacher=teacher, inputs=noisy, teacher_inputs=clean,
                       group_size=width, config={'enabled': True})
    assert observed == [True]
    with pytest.raises(ValueError, match='row-aligned'):
        refine_paro_export(result, teacher=teacher, inputs=noisy, teacher_inputs=clean[:-1],
                           group_size=width, config={'enabled': True})


@pytest.mark.parametrize('learn_scales', [False, True])
def test_paro_paired_fit_improves_actual_clean_target_reconstruction(learn_scales):
    from types import SimpleNamespace
    from gptqmodel.quantization.gsq_paro import refine_paro_export
    from gptqmodel.quantization.paroquant.optimization import _apply_inverse_rotation
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm

    width, group = 32, 16
    rng = torch.Generator().manual_seed(7)
    pairs = torch.arange(group).repeat(2).reshape(1, width).short()
    theta = torch.full((1, width // 2), .2).half()
    channel = torch.linspace(.5, 1.5, width).half()
    target = torch.full((8, width), .125)
    teacher = _apply_inverse_rotation(target, pairs, theta.float(), group_size=group,
                                      fused_rotation=False) * channel.float()
    noisy = torch.randn(128, width, generator=rng)
    clean = noisy * 1.2
    result = SimpleNamespace(pack_weight=torch.zeros_like(target).half(), pseudo_weight=torch.zeros_like(target),
                             q_scales=torch.full((8, 2), .125).half(), q_zeros=torch.full((8, 2), 8.),
                             pairs=pairs, theta=theta, channel_scales=channel)
    config = {'enabled': True, 'learn_scales': learn_scales, 'steps': 80, 'learning_rate': .2, 'seed': 7}
    fitted = refine_paro_export(result, teacher=teacher, inputs=noisy, teacher_inputs=clean,
                               group_size=group, config=config)
    assert fitted['after'] < fitted['before']
    linear = torch.nn.Linear(width, 8, bias=False, dtype=torch.float16)
    linear.weight.data.copy_(fitted['pack_weight'])
    packed = AwqTorchLinear(bits=4, group_size=group, sym=False, desc_act=False, in_features=width,
                            out_features=8, bias=False, register_buffers=True)
    packed.pack(linear, fitted['q_scales'], fitted['q_zeros'])
    decoded = dequantize_gemm(packed.qweight, packed.qzeros, packed.scales.float(), 4, group).T
    _, features = paro_gsq_basis(teacher, noisy, pairs, theta, channel, group_size=group)
    output = features @ decoded.T
    reference = clean @ teacher.T
    actual_loss = (output-reference).square().sum()
    assert actual_loss < reference.square().sum()
    constant = ((clean-noisy) @ teacher.T).square().sum()
    denominator = (noisy @ teacher.T).square().sum()
    torch.testing.assert_close((actual_loss-constant)/denominator, torch.tensor(fitted['after']),
                               rtol=1e-4, atol=1e-6)
    regular = refine_paro_export(result, teacher=teacher, inputs=noisy, group_size=group, config=config)
    identical = refine_paro_export(result, teacher=teacher, inputs=noisy, teacher_inputs=noisy,
                                  group_size=group, config=config)
    for key in ('pack_weight', 'pseudo_weight', 'q_scales', 'q_zeros'):
        assert torch.equal(regular[key], identical[key])
    assert regular['history'] == identical['history']
