import pytest
import torch

from gptqmodel.quantization.gsq_mxfp4 import mxfp4_candidates, pack_mxfp4_codes
from gptqmodel.utils.mxfp4_cpu import dequantize_mxfp4


@pytest.mark.parametrize('exponent', [0, 120, 127, 130])
def test_every_candidate_roundtrips_through_existing_decoder(exponent):
    # All 256 byte patterns exercise both nibble positions and signed zeros.
    weight = torch.arange(256).to(torch.uint8).reshape(8, 32)
    scales = torch.full((8, 2), exponent, dtype=torch.uint8)
    codes, decoded = mxfp4_candidates(weight, scales, count=16)
    assert torch.equal(pack_mxfp4_codes(codes[0]), weight)
    for index in range(16):
        reference = dequantize_mxfp4(pack_mxfp4_codes(codes[index]), scales)
        torch.testing.assert_close(decoded[index], reference, rtol=0, atol=0)


@pytest.mark.parametrize('count', [0, 17, True, 1.5])
def test_invalid_candidate_count(count):
    with pytest.raises(ValueError, match='count'):
        mxfp4_candidates(torch.zeros(1, 16, dtype=torch.uint8), torch.ones(1, 1, dtype=torch.uint8), count)


def test_invalid_scale_encoding_and_overflow_rejected():
    weight = torch.full((1, 16), 0x77, dtype=torch.uint8)
    for exponent in (254, 255):
        with pytest.raises(ValueError):
            mxfp4_candidates(weight, torch.full((1, 1), exponent, dtype=torch.uint8))


def test_mxfp4_fitter_exports_improving_hard_payload():
    from gptqmodel.quantization.gsq_mxfp4 import refine_mxfp4_weight

    # Every baseline nibble is 2 => 1.0; the next numeric level is 1.5.
    weight = torch.full((2, 16), 0x22, dtype=torch.uint8)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    teacher = torch.full((2, 32), 1.5)
    rng = torch.random.get_rng_state().clone()
    result = refine_mxfp4_weight(weight, scales, target=teacher,
        config={'enabled': True, 'steps': 80, 'learning_rate': .2, 'candidates': 3})
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result['after'] < result['before']
    assert torch.equal(result['scales'], scales)
    decoded = dequantize_mxfp4(result['weight'], result['scales'])
    expected = float((decoded-teacher).square().sum()/teacher.square().sum())
    assert result['after'] == pytest.approx(expected, abs=1e-8)
    assert all(a >= b for a, b in zip(result['history'], result['history'][1:]))


def test_mxfp4_disabled_and_exact_baseline_retention():
    from gptqmodel.quantization.gsq_mxfp4 import refine_mxfp4_weight

    weight = torch.zeros(2, 16, dtype=torch.uint8)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    disabled = refine_mxfp4_weight(weight, scales, target=None)
    assert torch.equal(disabled['weight'], weight)
    exact = refine_mxfp4_weight(weight, scales, target=torch.zeros(2, 32), config={'enabled': True})
    assert exact['before'] == exact['after'] == 0


@pytest.mark.parametrize('tokens', [8, 48])
def test_calibrated_objective_matches_direct_output_reconstruction(tokens):
    from gptqmodel.quantization.gsq_mxfp4 import refine_mxfp4_weight

    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(tokens, 32, generator=generator)
    teacher = torch.randn(2, 32, generator=generator)
    weight = torch.full((2, 16), 0x22, dtype=torch.uint8)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    config = {'enabled': True, 'steps': 20, 'learning_rate': .2, 'seed': 7}
    baseline = dequantize_mxfp4(weight, scales)
    energy = (inputs @ teacher.T).square().sum()
    expected_before = float((inputs @ (baseline-teacher).T).square().sum()/energy)
    for metric in ({'inputs': inputs}, {'hessian': inputs.T @ inputs}):
        result = refine_mxfp4_weight(weight, scales, target=teacher, config=config, **metric)
        decoded = dequantize_mxfp4(result['weight'], result['scales'])
        expected_after = float((inputs @ (decoded-teacher).T).square().sum()/energy)
        assert result['before'] == pytest.approx(expected_before, rel=2e-5)
        assert result['after'] == pytest.approx(expected_after, rel=2e-5)
        assert result['after'] <= result['before']
        assert torch.equal(result['scales'], scales)


@pytest.mark.parametrize('kind', ['both', 'asymmetric', 'negative', 'empty', 'nan', 'scales', 'budget'])
def test_fitter_rejects_invalid_calibration_and_unsupported_requests(kind):
    from gptqmodel.quantization.gsq_mxfp4 import refine_mxfp4_weight

    weight = torch.full((2, 16), 0x22, dtype=torch.uint8)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    config = {'enabled': True}
    metric = {}
    if kind == 'both':
        metric = {'inputs': torch.ones(2, 32), 'hessian': torch.eye(32)}
    elif kind == 'asymmetric':
        hessian = torch.eye(32)
        hessian[0, 1] = 1
        metric = {'hessian': hessian}
    elif kind == 'negative':
        metric = {'hessian': -torch.eye(32)}
    elif kind == 'empty':
        metric = {'inputs': torch.empty(0, 32)}
    elif kind == 'nan':
        metric = {'inputs': torch.full((2, 32), float('nan'))}
    elif kind == 'scales':
        config['learn_scales'] = True
    else:
        config['max_candidate_bytes'] = 1
    with pytest.raises(ValueError):
        refine_mxfp4_weight(weight, scales, target=torch.ones(2, 32), config=config, **metric)


def test_payload_replacement_invalidates_vnni_cache_and_owns_bytes(monkeypatch):
    from gptqmodel.nn_modules.qlinear import mxfp4_cpu

    calls = []

    class Extension:
        def mxfp4_prepack_vnni(self, weight, scales):
            calls.append(weight.clone())
            return weight.clone(), scales.clone()

    monkeypatch.setattr(mxfp4_cpu, 'load_mxfp4_cpu_kernel', lambda: Extension())
    module = mxfp4_cpu.Mxfp4CpuLinear(4, -1, False, True, 32, 2, use_vnni=True)
    module._maybe_prepack_vnni()
    weight = torch.full((2, 16), 0x33, dtype=torch.uint8)
    scales = torch.full((2, 1), 127, dtype=torch.uint8)
    module.replace_payload(weight, scales)
    assert not hasattr(module, 'qpack') and not hasattr(module, 'spack')
    weight.zero_()
    scales.zero_()
    module._maybe_prepack_vnni()
    assert len(calls) == 2
    assert torch.all(module.qpack == 0x33)
    assert torch.all(module.spack == 127)
    torch.testing.assert_close(module.dequantize_weight(), torch.full((32, 2), 1.5))
    saved = {key: value.clone() for key, value in module.state_dict().items()}
    with pytest.raises(ValueError):
        module.replace_payload(torch.zeros(2, 16, dtype=torch.uint8), torch.full((2, 1), 255, dtype=torch.uint8))
    assert all(torch.equal(value, module.state_dict()[key]) for key, value in saved.items())


def test_module_pack_refines_actual_initializer_and_retains_export_objective():
    from gptqmodel.nn_modules.qlinear.mxfp4_cpu import Mxfp4CpuLinear
    from gptqmodel.utils.mxfp4_cpu import quantize_mxfp4

    generator = torch.Generator().manual_seed(7)
    layer = torch.nn.Linear(32, 2, bias=False)
    layer.weight.data.copy_(torch.randn(2, 32, generator=generator))
    inputs = torch.randn(48, 32, generator=generator)
    module = Mxfp4CpuLinear(4, -1, False, True, 32, 2, use_vnni=False)
    baseline, scales = quantize_mxfp4(layer.weight.detach())
    module.pack_original(layer, None, None, gsq={'enabled': True, 'steps': 20}, gsq_inputs=inputs)
    expected = ((inputs @ (module.dequantize_weight().T-layer.weight).T).square().sum()
                / (inputs @ layer.weight.T).square().sum())
    assert module.gsq_diagnostics['after'] == pytest.approx(float(expected.detach()), rel=1e-5)
    assert module.gsq_diagnostics['after'] <= module.gsq_diagnostics['before']
    assert module.gsq_diagnostics['objective'] == 'activation_reconstruction'
    assert torch.equal(module.scales, scales)
    module.pack_original(layer, None, None)
    assert module.gsq_diagnostics is None
    assert torch.equal(module.qweight, baseline)


def test_mxfp4_config_roundtrip_and_module_override():
    from gptqmodel.quantization.config import MXFP4Config, clone_weight_only_config_for_module
    from gptqmodel.looper.weight_only_processor import WeightOnlyProcessor

    assert MXFP4Config().gsq is None
    cfg = MXFP4Config(gsq={'enabled': True, 'seed': 7}, dynamic={r'+:q_proj$': {'gsq': {'enabled': False}}})
    loaded = MXFP4Config.from_quant_config(cfg.to_dict())
    assert loaded.gsq.enabled and loaded.gsq.seed == 7
    assert not clone_weight_only_config_for_module(loaded, 'q_proj').gsq.enabled
    assert clone_weight_only_config_for_module(loaded, 'k_proj').gsq.enabled
    assert WeightOnlyProcessor._uses_direct_pack(loaded)
    with pytest.raises(ValueError, match='scale learning'):
        MXFP4Config(gsq={'enabled': True, 'learn_scales': True})
