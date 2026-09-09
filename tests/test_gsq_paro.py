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
