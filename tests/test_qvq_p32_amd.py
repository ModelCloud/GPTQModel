# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Accuracy, repeatability, and dispatch coverage for the gfx950 P32 kernel."""

from unittest.mock import patch

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    reconstruct_qvq_inner_weight,
    repack_p32_planar_to_window,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    pgc16_levels_for_version,
)
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from gptqmodel.utils.qvq_amd import (
    _launch_config,
    _use_gemv,
    qvq_p32_amd,
    qvq_p32_amd_supported,
)

P32_RATES = (2.0, 2.5, 3.0, 3.5)
REQUESTED_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _gfx950_available() -> bool:
    return torch.cuda.is_available() and torch.version.hip is not None and qvq_p32_amd_supported("cuda:0")


def _case(bits: float, m: int, *, k: int = 256, n: int = 256, seed: int = 950) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed + int(bits * 10) + m)
    tile_count = (k // 16) * (n // 16)
    planar = torch.randint(
        -(1 << 31),
        1 << 31,
        (tile_count, qvq_words_per_tile(bits, vector_size=2)),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    bank_ids = pack_qvq_binary_bank_ids(
        torch.randint(0, 2, (tile_count * 8,), dtype=torch.uint8, device="cuda", generator=generator)
    )
    bank_alt_id = torch.tensor([3], dtype=torch.uint8, device="cuda")
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).to("cuda")
    x = (torch.randn((m, k), dtype=torch.float16, device="cuda", generator=generator) * 0.1).contiguous()
    return x, planar, repack_p32_planar_to_window(planar, bits=bits), levels, bank_ids, bank_alt_id


def test_qvq_p32_amd_support_is_rocm_gfx950_only():
    properties = type("Properties", (), {"gcnArchName": "gfx950:sramecc+:xnack-"})()
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.version, "hip", "10.0"),
        patch.object(torch.cuda, "get_device_properties", return_value=properties),
    ):
        assert qvq_p32_amd_supported("cuda:0")
        properties.gcnArchName = "gfx942:sramecc+:xnack-"
        assert not qvq_p32_amd_supported("cuda:0")
    with patch.object(torch.version, "hip", None):
        assert not qvq_p32_amd_supported("cuda:0")
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert not qvq_p32_amd_supported("cuda:0")
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.version, "hip", "10.0"),
        patch.object(torch.cuda, "get_device_properties", side_effect=RuntimeError),
    ):
        assert not qvq_p32_amd_supported("cuda:0")
    assert not qvq_p32_amd_supported("cpu")


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    (
        (1, 4096, (16, 64, 8)),
        (16, 4096, (16, 64, 8)),
        (32, 4096, (32, 64, 8)),
        (64, 4096, (32, 64, 8)),
        (128, 4096, (128, 64, 8)),
        (256, 4096, (128, 64, 8)),
        (2048, 1024, (128, 64, 8)),
        (2048, 4096, (128, 128, 8)),
        (4096, 1024, (128, 128, 8)),
    ),
)
def test_qvq_p32_amd_launch_config_covers_requested_regimes(m, n, expected):
    assert _launch_config(m, n) == expected


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    (
        (1, 17408, True),
        (2, 6144, True),
        (2, 12288, False),
        (4, 4096, True),
        (4, 5120, False),
        (8, 1024, False),
    ),
)
def test_qvq_p32_amd_gemv_dispatch_limits_program_count(m, n, expected):
    assert _use_gemv(m, n) is expected


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("bits", P32_RATES)
@pytest.mark.parametrize("m", REQUESTED_M)
@pytest.mark.parametrize("seed", (950, 1950, 2950))
def test_qvq_p32_amd_all_requested_m_matches_fp32_reference(bits, m, seed):
    x, planar, window, levels, bank_ids, bank_alt_id = _case(bits, m, seed=seed)
    dense = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=x.shape[1],
        out_features=256,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    reference = x.float() @ dense
    outputs = [
        qvq_p32_amd(x, window, levels, bank_ids, bits, out_features=256, bank_alt_id=3)
        for _ in range(10)
    ]
    torch.cuda.synchronize()

    difference = outputs[0] - reference
    max_absolute = difference.abs().max().item()
    mean_absolute = difference.abs().mean().item()
    relative_l2 = difference.norm().div(reference.norm().clamp_min(1e-12)).item()
    assert outputs[0].shape == reference.shape
    assert outputs[0].dtype == torch.float32
    assert torch.isfinite(outputs[0]).all()
    assert max_absolute <= 2e-3, (
        f"W{bits:g} M{m} max_abs={max_absolute:.7g} "
        f"mean_abs={mean_absolute:.7g} relative_l2={relative_l2:.7g}"
    )
    assert all(torch.equal(outputs[0], repeated) for repeated in outputs[1:])


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("bits", P32_RATES)
def test_qvq_p32_amd_adversarial_activation_patterns(bits):
    x, planar, window, levels, bank_ids, bank_alt_id = _case(bits, 17, seed=3950)
    dense = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=x.shape[1],
        out_features=256,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    patterns = {
        "zero": torch.zeros_like(x),
        "alternating_sign": torch.where(
            torch.arange(x.numel(), device=x.device).reshape_as(x) % 2 == 0,
            torch.tensor(0.125, dtype=x.dtype, device=x.device),
            torch.tensor(-0.125, dtype=x.dtype, device=x.device),
        ),
        "mixed_magnitude": x.mul(1e-3),
    }
    patterns["mixed_magnitude"][0, 0] = 32.0
    patterns["mixed_magnitude"][-1, -1] = -32.0

    for name, activation in patterns.items():
        actual = qvq_p32_amd(
            activation,
            window,
            levels,
            bank_ids,
            bits,
            out_features=256,
            bank_alt_id=3,
        )
        reference = activation.float() @ dense
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=2e-3, msg=name)


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("bits", P32_RATES)
@pytest.mark.parametrize("bank_alt_id", (1, 2, 3))
def test_qvq_p32_amd_all_alternate_banks_match_reference(bits, bank_alt_id):
    x, planar, window, levels, bank_ids, _ = _case(bits, 7, seed=4950 + bank_alt_id)
    alternate = torch.tensor([bank_alt_id], dtype=torch.uint8, device="cuda")
    dense = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=x.shape[1],
        out_features=256,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=alternate,
    )
    actual = qvq_p32_amd(
        x,
        window,
        levels,
        bank_ids,
        bits,
        out_features=256,
        bank_alt_id=bank_alt_id,
        output_fp32=False,
    )
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual.float(), x.float() @ dense, rtol=0.0, atol=2e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
def test_qvq_p32_amd_rejects_invalid_contracts():
    x, _, window, levels, bank_ids, _ = _case(3.0, 7, seed=5950)

    with pytest.raises(ValueError, match="supports rates"):
        qvq_p32_amd(x, window, levels, bank_ids, 4.0, out_features=256, bank_alt_id=3)
    with pytest.raises(TypeError, match="out_features must be an integer"):
        qvq_p32_amd(x, window, levels, bank_ids, 3.0, out_features=True, bank_alt_id=3)
    with pytest.raises(TypeError, match="bank_alt_id must be an integer"):
        qvq_p32_amd(x, window, levels, bank_ids, 3.0, out_features=256, bank_alt_id="3")
    with pytest.raises(TypeError, match="output_fp32 must be boolean"):
        qvq_p32_amd(
            x,
            window,
            levels,
            bank_ids,
            3.0,
            out_features=256,
            bank_alt_id=3,
            output_fp32=1,
        )
    with pytest.raises(RuntimeError, match="ROCm gfx950"):
        qvq_p32_amd(x.cpu(), window, levels, bank_ids, 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(ValueError, match="expects 2D"):
        qvq_p32_amd(x.unsqueeze(0), window, levels, bank_ids, 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(TypeError, match="float16 input"):
        qvq_p32_amd(x.float(), window, levels, bank_ids, 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(TypeError, match="int32 continuous-window"):
        qvq_p32_amd(x, window.long(), levels, bank_ids, 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(TypeError, match="canonical 256-entry"):
        qvq_p32_amd(x, window, levels.float(), bank_ids, 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(TypeError, match="packed uint8"):
        qvq_p32_amd(x, window, levels, bank_ids.int(), 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(ValueError, match="share one device"):
        qvq_p32_amd(x, window.cpu(), levels, bank_ids, 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(ValueError, match="contiguous"):
        qvq_p32_amd(
            x,
            window.t().contiguous().t(),
            levels,
            bank_ids,
            3.0,
            out_features=256,
            bank_alt_id=3,
        )
    with pytest.raises(ValueError, match="alternative bank ID"):
        qvq_p32_amd(x, window, levels, bank_ids, 3.0, out_features=256, bank_alt_id=0)
    with pytest.raises(ValueError, match="positive M"):
        qvq_p32_amd(x[:0], window, levels, bank_ids, 3.0, out_features=256, bank_alt_id=3)
    with pytest.raises(ValueError, match="window must have shape"):
        qvq_p32_amd(
            x,
            window[:, :-1].contiguous(),
            levels,
            bank_ids,
            3.0,
            out_features=256,
            bank_alt_id=3,
        )
    with pytest.raises(ValueError, match="selectors must have shape"):
        qvq_p32_amd(x, window, levels, bank_ids[:-1], 3.0, out_features=256, bank_alt_id=3)


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("m", (1, 64, 4096))
def test_qvq_linear_dispatches_gfx950_p32_and_matches_reference(m):
    bits = 3.0
    x, planar, _, _, bank_ids, bank_alt_id = _case(bits, m, seed=1950)
    layer = QVQLinear(
        bits=bits,
        in_features=256,
        out_features=256,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=False,
        output_hadamard=False,
        tensors={
            "trellis": planar,
            "SU": torch.ones(256, dtype=torch.float32, device="cuda"),
            "SV": torch.ones(256, dtype=torch.float32, device="cuda"),
            "bank_ids": bank_ids,
            "bank_alt_id": bank_alt_id,
        },
    ).eval()
    reference = x.float() @ layer.get_inner_weight_tensor()
    actual = layer(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.float(), reference, rtol=0.0, atol=2e-3)
    assert layer._qvq_cuda_window_cache is not None


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
def test_qvq_p32_amd_uses_current_non_default_stream():
    bits = 3.5
    x, planar, window, levels, bank_ids, bank_alt_id = _case(bits, 32, k=2048, n=80, seed=2950)
    dense = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=2048,
        out_features=80,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    reference = x.float() @ dense
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_p32_amd(x, window, levels, bank_ids, bits, out_features=80, bank_alt_id=3)
    stream.synchronize()
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=2e-3)
