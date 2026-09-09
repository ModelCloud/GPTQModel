import pytest
import torch

from gptqmodel.quantization.gsq_fp8 import fp8_payload_candidates


@pytest.mark.parametrize('name', ['float8_e4m3fn', 'float8_e5m2', 'float8_e4m3fnuz', 'float8_e5m2fnuz'])
def test_candidates_preserve_every_finite_encoding_and_follow_actual_grid(name):
    dtype = getattr(torch, name)
    raw = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    decoded = raw.view(dtype).float()
    finite = raw[torch.isfinite(decoded)]
    weight = finite.view(dtype).reshape(1, -1)
    payloads, values = fp8_payload_candidates(weight, 5)
    assert torch.equal(payloads[0], finite.reshape(1, -1))
    assert torch.equal(values, payloads.view(dtype).float())
    assert torch.isfinite(values).all()
    grid = sorted(set(decoded[torch.isfinite(decoded)].tolist()))
    for column, value in enumerate(weight.float().flatten().tolist()):
        center = grid.index(value)
        for candidate, offset in enumerate([0, -1, 1, -2, 2]):
            expected = grid[min(max(center + offset, 0), len(grid)-1)]
            assert values[candidate, 0, column].item() == expected
    nonfinite = raw[~torch.isfinite(decoded)][:1].view(dtype).reshape(1, 1)
    with pytest.raises(ValueError, match='finite'):
        fp8_payload_candidates(nonfinite)


@pytest.mark.parametrize('count', [0, -1, True, 257, 1.5])
def test_invalid_candidate_counts_rejected(count):
    with pytest.raises(ValueError, match='count'):
        fp8_payload_candidates(torch.zeros(2, 3).to(torch.float8_e4m3fn), count)


def test_noncontiguous_fp8_payload_keeps_logical_order():
    weight = torch.arange(12).reshape(3, 4).to(torch.float8_e4m3fn).T
    payload, values = fp8_payload_candidates(weight, 1)
    assert torch.equal(payload[0], weight.contiguous().view(torch.uint8))
    assert torch.equal(values[0], weight.float())


@pytest.mark.parametrize('method', ['tensor', 'row', 'block'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_runtime_honors_inverse_scale_below_one(method, dtype):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    layer = TorchFP8Linear(bits=8, group_size=-1, sym=True, desc_act=False, in_features=4, out_features=2,
                           weight_scale_method=method, weight_block_size=(1, 2) if method == 'block' else None)
    layer.weight.copy_(torch.ones(2, 4).to(layer.weight.dtype))
    layer.weight_scale_inv.fill_(.5)
    assert torch.equal(layer.dequantize_weight(dtype=dtype), torch.full((4, 2), 2., dtype=dtype))


@pytest.mark.parametrize('method', ['tensor', 'row', 'block'])
@pytest.mark.parametrize('inverse', [.5, 2.])
def test_decoded_candidates_match_runtime_scale_geometry(method, inverse):
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.quantization.gsq_fp8 import fp8_decoded_candidates

    block = (2, 3) if method == 'block' else None
    layer = TorchFP8Linear(bits=8, group_size=-1, sym=True, desc_act=False, in_features=6, out_features=4,
                           weight_scale_method=method, weight_block_size=block)
    layer.weight.copy_((torch.arange(24).reshape(4, 6).float()-12).to(layer.weight.dtype))
    layer.weight_scale_inv.fill_(inverse)
    if method != 'tensor':
        layer.weight_scale_inv.reshape(-1)[::2] *= 1.25
    payloads, decoded = fp8_decoded_candidates(layer.weight, layer.weight_scale_inv,
                                              method=method, block_size=block, count=5)
    for index in range(len(payloads)):
        layer.weight.copy_(payloads[index].view(layer.weight.dtype))
        torch.testing.assert_close(decoded[index], layer.dequantize_weight(dtype=torch.float32).T,
                                   rtol=0, atol=0)


@pytest.mark.parametrize('scales,method,block', [
    (torch.tensor([0., 1.]), 'row', None),
    (torch.tensor([float('nan'), 1.]), 'row', None),
    (torch.ones(2, 1), 'row', None),
    (torch.ones(2), 'tensor', None),
    (torch.ones(1, 1), 'block', (3, 4)),
    (torch.ones(1, 1), 'block', (True, 4)),
])
def test_invalid_scale_metadata_rejected(scales, method, block):
    from gptqmodel.quantization.gsq_fp8 import fp8_decoded_candidates

    with pytest.raises(ValueError):
        fp8_decoded_candidates(torch.ones(2, 4).to(torch.float8_e4m3fn), scales, method=method, block_size=block)


@pytest.mark.parametrize('calibrated', [False, True])
def test_fp8_fitter_improves_hard_payload_and_preserves_global_rng(calibrated):
    from gptqmodel.quantization.gsq_fp8 import refine_fp8_weight

    weight = torch.ones(4, 8).to(torch.float8_e4m3fn)
    scales = torch.ones(4)
    target = torch.full((4, 8), 1.125)
    inputs = torch.randn(32, 8, generator=torch.Generator().manual_seed(7)) if calibrated else None
    rng = torch.random.get_rng_state().clone()
    result = refine_fp8_weight(weight, scales, target=target, inputs=inputs,
                               config={'enabled': True, 'steps': 80, 'learning_rate': .2, 'seed': 7})
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result['after'] < result['before']
    assert torch.equal(result['scale_inv'], scales)
    decoded = result['weight'].float()
    error, teacher = decoded-target, target
    if inputs is not None:
        error, teacher = error @ inputs.T, teacher @ inputs.T
    assert result['after'] == pytest.approx(float(error.square().sum()/teacher.square().sum()), abs=1e-7)
    assert all(a >= b for a, b in zip(result['history'], result['history'][1:]))


def test_fp8_fitter_exact_zero_and_disabled_retention():
    from gptqmodel.quantization.gsq_fp8 import refine_fp8_weight

    weight = torch.zeros(2, 4).to(torch.float8_e4m3fn)
    scales = torch.ones(2)
    exact = refine_fp8_weight(weight, scales, target=weight.float(), config={'enabled': True})
    assert exact['before'] == exact['after'] == 0
    disabled = refine_fp8_weight(weight, scales, target=None)
    assert torch.equal(disabled['weight'].view(torch.uint8), weight.view(torch.uint8))
    with pytest.raises(ValueError, match='scale learning'):
        refine_fp8_weight(weight, scales, target=weight.float(), config={'enabled': True, 'learn_scales': True})


@pytest.mark.parametrize('shape', [(8,), (0, 8), (2, 0), (1, 2, 4)])
def test_fp8_fitter_rejects_invalid_weight_shape_before_input_access(shape):
    from gptqmodel.quantization.gsq_fp8 import refine_fp8_weight

    weight = torch.zeros(shape).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match='rank-2'):
        refine_fp8_weight(weight, torch.ones(1), target=weight.float(), inputs=torch.ones(2, 8),
                          config={'enabled': True})


def test_fp8_fitter_rejects_overflowing_teacher_energy():
    from gptqmodel.quantization.gsq_fp8 import refine_fp8_weight

    weight = torch.ones(2, 4).to(torch.float8_e4m3fn)
    # Finite entries can still overflow the squared normalization. In particular,
    # an infinite denominator must not turn a finite residual into false zero loss.
    target = torch.full((2, 4), 1e20)
    with pytest.raises(ValueError, match='teacher energy'):
        refine_fp8_weight(weight, torch.ones(2), target=target, config={'enabled': True})


def test_fp8_gsq_config_roundtrip_and_dynamic_override():
    from gptqmodel.quantization.config import FP8Config, QuantizeConfig, clone_weight_only_config_for_module

    cfg = FP8Config(gsq={'enabled': True, 'steps': 3}, dynamic={'+:proj': {'gsq': {'enabled': False}}})
    restored = QuantizeConfig.from_quant_config(cfg.to_dict())
    assert restored.gsq == cfg.gsq
    assert not clone_weight_only_config_for_module(restored, 'proj').gsq.enabled
    assert clone_weight_only_config_for_module(restored, 'other').gsq.enabled
    assert FP8Config().gsq is None
    with pytest.raises(ValueError, match='scale learning'):
        FP8Config(gsq={'enabled': True, 'learn_scales': True})


@pytest.mark.parametrize('enabled,matched', [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize('force_change', [False, True])
def test_fp8_processor_pack_refine_reload(enabled, matched, force_change, monkeypatch, tmp_path):
    import threading
    from types import SimpleNamespace

    import gptqmodel.looper.weight_only_processor as implementation
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.quantization.config import FP8Config

    cfg = FP8Config(device='cpu', gsq={'enabled': enabled, 'steps': 80, 'candidates': 3,
                                      'learning_rate': .2, 'modules': ['proj$' if matched else 'absent$']})
    dense = torch.nn.Linear(8, 4, bias=False)
    dense.weight.data.copy_(torch.randn(4, 8, generator=torch.Generator().manual_seed(7)))
    if force_change:
        dense.weight.data.fill_(1.125)
    teacher = dense.weight.detach().clone()
    baseline_bytes = []
    named = NamedModule(dense, name='proj', full_name='model.layers.0.proj', layer_index=0)
    kwargs = dict(bits=8, group_size=-1, sym=True, desc_act=False, in_features=8, out_features=4, bias=False)
    packed = TorchFP8Linear(**kwargs)
    monkeypatch.setattr(implementation, 'create_quant_module', lambda **kw: packed)

    def pack(**kw):
        packed.pack_original(dense, None, None)
        if force_change:
            # Controlled non-optimal initializer: prove that the actual fitter's
            # changed bytes reach export, rather than only testing baseline retention.
            with torch.inference_mode():
                packed.weight.fill_(1.)
                packed.weight_scale_inv.fill_(1.)
        baseline_bytes.append(packed.weight.view(torch.uint8).clone())
        return 'fp8'

    monkeypatch.setattr(implementation, 'pack_module', pack)
    processor = object.__new__(implementation.WeightOnlyProcessor)
    processor.qcfg = cfg
    processor.lock = threading.Lock()
    processor.log = []
    model = SimpleNamespace(qlinear_kernel=TorchFP8Linear, model=torch.nn.Module(), lm_head='lm_head')
    result = processor.submodule_finalize(named, model, qcfg=cfg)
    restored = TorchFP8Linear(**kwargs)
    checkpoint = tmp_path / 'fp8.pt'
    torch.save(result.state_dict(), checkpoint)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert torch.equal(restored.weight.view(torch.uint8), result.weight.view(torch.uint8))
    decoded = restored.dequantize_weight(dtype=torch.float32).T
    features = torch.randn(7, 8, generator=torch.Generator().manual_seed(9))
    torch.testing.assert_close(restored(features), features @ decoded.T, rtol=0, atol=0)
    if enabled and matched:
        stats = named.state['gsq_diagnostics']
        expected = float((decoded-teacher).square().sum()/teacher.square().sum())
        assert stats['after'] == pytest.approx(expected, rel=1e-5)
        assert stats['after'] <= stats['before']
        assert stats['objective'] == 'weight_reconstruction'
        if force_change:
            assert stats['after'] < stats['before']
            assert not torch.equal(restored.weight.view(torch.uint8), baseline_bytes[0])
    else:
        assert torch.equal(restored.weight.view(torch.uint8), baseline_bytes[0])
        assert 'gsq_diagnostics' not in named.state
