# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear, _swizzle_scales, nvfp4_input
from gptqmodel.nn_modules.qlinear.w4a_nvfp4_triton import nvfp4_pack_and_swizzle
from gptqmodel.quantization.activation_floatx import (
    NVFP4ActivationHeadroom,
    nvfp4_block_qdq,
    nvfp4_global_scale,
)
from gptqmodel.quantization.config import QuantizeConfig


@pytest.mark.parametrize("legacy,canonical", [
    ("lsq", "least_squares"),
    ("lsq_headroom", "least_squares_headroom"),
    ("lsq_grid", "least_squares_grid"),
])
def test_legacy_recipe_names_load_and_export_expanded_names(tmp_path, legacy, canonical):
    import json

    config = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False,
                           activation={"version": 3, "mode": "w4a_nvfp4", "recipe": canonical})
    config.save_pretrained(str(tmp_path))
    path = tmp_path / "quantize_config.json"
    saved = json.loads(path.read_text())
    saved["activation"]["recipe"] = legacy
    path.write_text(json.dumps(saved))
    loaded = QuantizeConfig.from_pretrained(str(tmp_path))
    assert loaded.activation_recipe == canonical
    loaded.save_pretrained(str(tmp_path))
    assert json.loads(path.read_text())["activation"]["recipe"] == canonical

    x = torch.linspace(-4, 4, 256).reshape(2, 128)
    global_scale = torch.tensor(.01)
    torch.testing.assert_close(nvfp4_block_qdq(x, global_scale, legacy),
                               nvfp4_block_qdq(x, global_scale, canonical), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("legacy,canonical", [
    ("lsq", "least_squares"),
    ("lsq_headroom", "least_squares_headroom"),
    ("lsq_grid", "least_squares_grid"),
])
def test_legacy_recipe_names_preserve_gpu_packed_codes_and_scales(legacy, canonical):
    from gptqmodel.nn_modules.qlinear.w4a_activation import pack_activation

    generator = torch.Generator(device="cuda").manual_seed(1893)
    x = torch.randn((33, 256), dtype=torch.bfloat16, device="cuda", generator=generator)
    scale = torch.tensor(.01, device="cuda")
    old = pack_activation(x, "w4a_nvfp4", global_scale=scale, recipe=legacy)
    new = pack_activation(x, "w4a_nvfp4", global_scale=scale, recipe=canonical)
    assert old.recipe == new.recipe == canonical
    assert torch.equal(old.codes.view(torch.uint8), new.codes.view(torch.uint8))
    assert torch.equal(old.scales.view(torch.uint8), new.scales.view(torch.uint8))


def _independent_nvfp4_qdq(x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Numeric codebook oracle for hardware-scale least-squares refinement."""
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    maxima = blocks.abs().amax(dim=-1)
    # Put even E2M1 codes first so argmin independently implements ties-even.
    codebook = torch.tensor(
        (0., -1., 1., -2., 2., -4., 4., -.5, .5, -1.5, 1.5, -3., 3., -6., 6.),
        device=x.device,
    )
    best_error = torch.full_like(maxima, torch.inf)
    best_reconstructed = torch.zeros_like(blocks)
    best_values = torch.zeros_like(blocks)
    seeds = []

    def evaluate(local):
        scale = local[..., None] * global_scale
        nearest = (blocks[..., None] / scale[..., None] - codebook).abs().argmin(dim=-1)
        values = codebook[nearest]
        reconstructed = values * scale
        error = (reconstructed - blocks).square().sum(dim=-1)
        return values, reconstructed, error

    def refine(values):
        denominator = values.square().sum(dim=-1)
        optimal = (blocks * values).sum(dim=-1) / denominator.clamp_min(1.0)
        inverse = optimal / global_scale
        return torch.where(
            denominator > 0,
            inverse.clamp(min=2.0**-9, max=448.0),
            torch.ones_like(inverse),
        ).to(torch.float8_e4m3fn).float()

    for bound in (4.0, 6.0):
        local = torch.where(
            maxima > 0,
            (maxima / (bound * global_scale)).clamp(min=2.0**-9, max=448.0),
            torch.ones_like(maxima),
        ).to(torch.float8_e4m3fn).float()
        values, reconstructed, error = evaluate(local)
        seeds.append(values)
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_reconstructed = torch.where(use[..., None], reconstructed, best_reconstructed)
        best_values = torch.where(use[..., None], values, best_values)
    for values in seeds:
        local = refine(values)
        values, reconstructed, error = evaluate(local)
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_reconstructed = torch.where(use[..., None], reconstructed, best_reconstructed)
        best_values = torch.where(use[..., None], values, best_values)
    local = refine(best_values)
    _, reconstructed, error = evaluate(local)
    use = error < best_error
    best_reconstructed = torch.where(use[..., None], reconstructed, best_reconstructed)
    return best_reconstructed.reshape_as(x)


def _four_over_six_qdq(x: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    blocks = x.float().reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    maxima = blocks.abs().amax(dim=-1)
    codebook = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=x.device)
    candidates = []
    errors = []
    for bound in (4.0, 6.0):
        local = torch.where(
            maxima > 0,
            (maxima / (bound * global_scale)).clamp(min=2.0**-9, max=448.0),
            torch.ones_like(maxima),
        ).to(torch.float8_e4m3fn).float()
        normalized = blocks / (local * global_scale)[..., None]
        indices = (normalized.abs()[..., None] - codebook).abs().argmin(dim=-1)
        values = codebook[indices].copysign(normalized)
        reconstructed = values * (local * global_scale)[..., None]
        candidates.append(reconstructed)
        errors.append((reconstructed - blocks).square().sum(dim=-1))
    return torch.where((errors[0] <= errors[1])[..., None], candidates[0], candidates[1]).reshape_as(x)


def _module(k=128, n=128, device="cpu"):
    module = W4ANVFP4Linear(
        bits=4, group_size=128, sym=True, desc_act=False,
        in_features=k, out_features=n, bias=False,
    ).to(device)
    codes = torch.arange(k, device=device, dtype=torch.int32).remainder(16)
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    words = (codes.reshape(k // 8, 8) << shifts).sum(dim=1).to(torch.int32)
    module.qweight.copy_(words[:, None].expand(k // 8, n))
    module.qzeros.fill_(0x77777777)
    module.scales.copy_(
        torch.arange(1, k // 128 + 1, device=device, dtype=torch.float16)[:, None].expand(k // 128, n) / 8
    )
    module.activation_global_scale.fill_(0.03125)
    module.post_init()
    return module


def test_nvfp4_config_and_exact_weight_planes(tmp_path):
    config = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation="w4a_nvfp4", offload_to_disk=False,
    )
    config.save_pretrained(str(tmp_path))
    loaded = QuantizeConfig.from_pretrained(str(tmp_path))
    assert loaded.activation == {"version": 3, "mode": "w4a_nvfp4", "recipe": "least_squares"}
    assert loaded.activation_version == 3
    assert loaded.activation_recipe == "least_squares"
    legacy = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={"version": 2, "mode": "w4a_nvfp4"}, offload_to_disk=False,
    )
    assert legacy.activation == {"version": 2, "mode": "w4a_nvfp4", "recipe": "four_six"}
    assert legacy.activation_recipe == "four_six"
    nvidia = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={"version": 2, "mode": "w4a_nvfp4", "recipe": "nvidia"},
        offload_to_disk=False,
    )
    assert nvidia.activation_recipe == "nvidia"
    headroom = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={"version": 2, "mode": "w4a_nvfp4", "recipe": "nvidia_headroom"},
        offload_to_disk=False,
    )
    assert headroom.activation_recipe == "nvidia_headroom"
    least_squares_headroom = QuantizeConfig(
        bits=4, group_size=128, sym=True, desc_act=False,
        activation={"version": 2, "mode": "w4a_nvfp4", "recipe": "least_squares_headroom"},
        offload_to_disk=False,
    )
    assert least_squares_headroom.activation_recipe == "least_squares_headroom"
    module = _module()
    # Decode the actual packed FP4 operands through the format's independent
    # numeric codebook, then reconstruct every centered INT4 value.
    values = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6))

    def decode(plane):
        data = plane.view(torch.uint8)
        codes = torch.stack((data & 15, data >> 4), dim=1).reshape(128, 128)
        return values[(codes & 7).long()] * torch.where((codes & 8) == 0, 1, -1)

    centered = decode(module._weight_both[:, :128]) + 4 * decode(module._weight_both[:, 128:])
    expected = torch.arange(128).remainder(16).sub(8).float()[:, None].expand(128, 128)
    torch.testing.assert_close(centered, expected, rtol=0, atol=0)
    assert module.qweight.dtype == torch.int32
    # Mixed-precision attention consumes FP8 carriers, so the module stages a
    # derived E4M3 plane in post_init. It is non-persistent and never saved.
    assert module._weight_e4m3.numel() == module.in_features * module.out_features
    assert module._weight_e4m3.dtype == torch.float8_e4m3fn
    assert set(module.state_dict()) == {"qweight", "qzeros", "scales", "g_idx", "activation_global_scale_bits"}
    assert nvfp4_global_scale(4 * 448).item() == 1.0
    assert nvfp4_global_scale(6 * 448, recipe="nvidia").item() == 1.0
    assert nvfp4_global_scale(6 * 448, recipe="nvidia_headroom").item() == 1.0
    assert nvfp4_global_scale(6 * 448, recipe="least_squares_headroom").item() == 1.0


def test_nvfp4_headroom_scale_matches_nvidia_percentile_definition():
    collector = NVFP4ActivationHeadroom(rho=1024.0)
    collector.collect(torch.full((256, 64), 0.5))
    assert float(collector.compute_amax()) == pytest.approx(512.0, rel=0.06)


def test_nvfp4_headroom_default_clips_a_single_extreme_block():
    values = torch.ones((4096, 64), dtype=torch.float32)
    values[0, :16] = 3e7
    clipped = NVFP4ActivationHeadroom()
    literal = NVFP4ActivationHeadroom(upper_percentile=100.0)
    clipped.collect(values)
    literal.collect(values)
    assert float(clipped.compute_amax()) < 3e7
    assert float(literal.compute_amax()) >= 3e7


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_nvfp4_replay_matches_packed_runtime_input():
    generator = torch.Generator(device="cuda").manual_seed(510)
    x = torch.randn((3, 256), generator=generator, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1, :16] = 0.25
    global_scale = torch.tensor(0.03125, device="cuda")
    packed, local = nvfp4_input(x, global_scale)
    bytes_ = packed.view(torch.uint8)
    codes = torch.stack((bytes_ & 15, bytes_ >> 4), dim=-1).reshape_as(x)
    values = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device="cuda")
    decoded = values[(codes & 7).long()] * torch.where((codes & 8) == 0, 1., -1.)
    from_packed = decoded.reshape(3, 16, 16) * local.float()[..., None] * global_scale
    oracle = nvfp4_block_qdq(x, global_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(from_packed.reshape_as(x), oracle.float(), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_nvfp4_refined_scale_never_increases_block_sse():
    generator = torch.Generator(device="cuda").manual_seed(2037)
    x = torch.randn((32, 256), generator=generator, device="cuda", dtype=torch.float32)
    x[0].zero_()
    x[1, 0] = 20
    global_scale = nvfp4_global_scale(x.abs().amax()).to("cuda")
    refined = nvfp4_block_qdq(x, global_scale).reshape(-1, 16)
    baseline = _four_over_six_qdq(x, global_scale).reshape(-1, 16)
    source = x.reshape(-1, 16)
    refined_sse = (refined - source).square().sum(dim=-1)
    baseline_sse = (baseline - source).square().sum(dim=-1)
    torch.cuda.synchronize()
    assert torch.all(refined_sse <= baseline_sse + 1e-6)
    assert torch.count_nonzero(refined_sse < baseline_sse).item() > refined_sse.numel() // 4


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_nvfp4_grid_search_never_increases_least_squares_block_sse():
    generator = torch.Generator(device="cuda").manual_seed(2041)
    x = torch.randn((32, 256), generator=generator, device="cuda", dtype=torch.float32)
    x[0].zero_()
    global_scale = nvfp4_global_scale(x.abs().amax()).to("cuda")
    refined = nvfp4_block_qdq(x, global_scale, recipe="least_squares").reshape(-1, 16)
    grid = nvfp4_block_qdq(x, global_scale, recipe="least_squares_grid").reshape(-1, 16)
    source = x.reshape(-1, 16)
    refined_sse = (refined - source).square().sum(dim=-1)
    grid_sse = (grid - source).square().sum(dim=-1)
    assert torch.all(grid_sse <= refined_sse + 1e-6)
    assert torch.count_nonzero(grid_sse < refined_sse).item() > 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize(
    "recipe", ["nvidia", "nvidia_headroom", "four_six", "least_squares", "least_squares_headroom", "least_squares_grid"]
)
@pytest.mark.parametrize("rows,width", [(1, 128), (7, 256), (129, 128)])
def test_fused_nvfp4_pack_matches_torch_codes_and_scale_layout(rows, width, recipe):
    generator = torch.Generator(device="cuda").manual_seed(538 + rows)
    x = torch.randn((rows, width), generator=generator, device="cuda", dtype=torch.bfloat16)
    x[0, :16] = 0
    global_scale = torch.tensor(0.03125, device="cuda")
    packed, swizzled = nvfp4_pack_and_swizzle(x, global_scale, recipe=recipe)
    reference_codes, reference_scales = nvfp4_input(x, global_scale, recipe=recipe)
    expected_scales = torch.stack([
        _swizzle_scales(reference_scales[:, group * 8:(group + 1) * 8])
        for group in range(width // 128)
    ])
    torch.cuda.synchronize()
    assert torch.equal(packed.view(torch.uint8), reference_codes.view(torch.uint8))
    assert torch.equal(swizzled.view(torch.uint8), expected_scales.view(torch.uint8))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
def test_nvfp4_forward_can_be_captured_in_cuda_graph():
    module = _module(k=128, n=128, device="cuda")
    x = torch.randn((1, 128), device="cuda", dtype=torch.bfloat16)
    expected = module(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = module(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("rows", [1, 16, 128])
def test_nvfp4_native_output_against_independent_oracle(rows):
    module = _module(k=256, n=128, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(509)
    x = torch.randn((rows, 256), generator=generator, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    actual = module(x)

    # Independent FP4 QDQ oracle: derive block scales and choose the nearest
    # finite E2M1 value from the numeric codebook, without using backend codes.
    global_scale = module.activation_global_scale.float()
    xq = _independent_nvfp4_qdq(x, global_scale)
    weights = torch.arange(256, device="cuda", dtype=torch.float32).remainder(16).sub(8)
    oracle = torch.zeros((rows, 128), device="cuda", dtype=torch.float32)
    for group in range(2):
        sl = slice(group * 128, (group + 1) * 128)
        oracle += (xq[:, sl] @ weights[sl, None].expand(128, 128)) * module.scales[group].float()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, oracle.to(actual.dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("n", [96, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nvfp4_random_columns_and_scales(n, dtype):
    generator = torch.Generator(device="cuda").manual_seed(2031)
    k, rows = 256, 7
    module = _module(k=k, n=n, device="cuda")
    logical_codes = torch.randint(0, 16, (k, n), generator=generator, device="cuda", dtype=torch.int32)
    shifts = torch.arange(8, device="cuda", dtype=torch.int32) * 4
    module.qweight.copy_((logical_codes.reshape(k // 8, 8, n) << shifts[None, :, None]).sum(dim=1))
    module.scales.copy_((0.02 + 0.18 * torch.rand((k // 128, n), generator=generator, device="cuda")).half())
    module.post_init()
    x = torch.randn((rows, k), generator=generator, device="cuda", dtype=dtype)
    actual = module(x)

    global_scale = module.activation_global_scale.float()
    xq = _independent_nvfp4_qdq(x, global_scale)
    weight = logical_codes.float() - 8
    oracle = torch.zeros((rows, n), device="cuda", dtype=torch.float32)
    for group in range(k // 128):
        sl = slice(group * 128, (group + 1) * 128)
        oracle += (xq[:, sl] @ weight[sl]) * module.scales[group].float()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, oracle.to(actual.dtype), rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1),
    reason="GB10 / SM121 required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nvfp4_norm_preserves_codes_and_scales_against_oracle(dtype):
    from types import SimpleNamespace

    from gptqmodel.nn_modules.qlinear.w4a_activation import pack_activation
    from gptqmodel.nn_modules.qlinear.w4a_llama_stream import _norm_forward

    generator = torch.Generator(device="cuda").manual_seed(9771)
    x = torch.randn((3, 256), device="cuda", dtype=dtype, generator=generator)
    x[0].zero_()
    global_scale = torch.tensor(0.03125, device="cuda")
    encoded = pack_activation(x, "w4a_nvfp4", global_scale=global_scale, recipe="least_squares")
    reference = _independent_nvfp4_qdq(x, global_scale)
    norm = SimpleNamespace(
        variance_epsilon=1e-5, _w4a_preserve_norm_codes=True,
        _w4a_norm_mode="w4a_nvfp4", _w4a_norm_recipe="least_squares",
    )
    actual = _norm_forward(norm, encoded)
    expected = reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + 1e-5)
    assert actual.codes.data_ptr() == encoded.codes.data_ptr()
    assert actual.scales.data_ptr() == encoded.scales.data_ptr()
    assert actual.token_scale.dtype == torch.float32
    assert actual.token_scale.shape == (3,)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.decode(torch.float32), expected.float(), rtol=1e-6, atol=1e-6)

    layer = _module(k=256, n=128, device="cuda")
    layer._w4a_output_encoded = False
    layer.bias = torch.linspace(-0.25, 0.25, 128, device="cuda", dtype=dtype)
    weights = torch.arange(256, device="cuda", dtype=torch.float32).remainder(16).sub(8)
    output_reference = torch.zeros((3, 128), device="cuda", dtype=torch.float32)
    for group in range(2):
        sl = slice(group * 128, (group + 1) * 128)
        output_reference += (expected[:, sl] @ weights[sl, None].expand(128, 128)) * layer.scales[group].float()
    output_reference += layer.bias.float()
    output = layer(actual)
    torch.cuda.synchronize()
    assert output.dtype == dtype
    torch.testing.assert_close(output, output_reference.to(dtype), rtol=2e-3, atol=2e-3)


def test_nvfp4_v4_config_requires_fused_rotation(tmp_path):
    with pytest.raises(ValueError, match="requires NVFP4 and rotation"):
        QuantizeConfig(bits=4, group_size=128,
                       activation={"version": 4, "mode": "w4a_nvfp4", "recipe": "least_squares"})
    cfg = QuantizeConfig(bits=4, group_size=128, rotation="hadamard",
                         activation={"version": 4, "mode": "w4a_nvfp4", "recipe": "least_squares"})
    cfg.save_pretrained(str(tmp_path))
    assert QuantizeConfig.from_pretrained(str(tmp_path)).activation_version == 4
