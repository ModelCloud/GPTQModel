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
