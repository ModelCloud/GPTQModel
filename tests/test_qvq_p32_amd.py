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
from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
from gptqmodel.utils.qvq_amd import (
    _launch_config,
    _use_gemv,
    qvq_p32_amd,
    qvq_p32_amd_folded,
    qvq_p32_amd_folded_case_supported,
    qvq_p32_amd_folded_prefers_fp32_output,
    qvq_p32_amd_folded_shape_supported,
    qvq_p32_amd_supported,
)
from scripts.benchmark_qvq_p32_amd import _target_process_ids

P32_RATES = (2.0, 2.5, 3.0, 3.5)
REQUESTED_M = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def test_qvq_p32_amd_benchmark_filters_rocm_processes_to_target_gpu():
    system = {
        "Driver version": "7.1.3",
        "PID101": "candidate, 0, 4096, 0, 0",
        "PID102": "other-gpu, 1, 8192, 0, 0",
        "PID103": "stale, 0, 0, 0, 0",
        "PID104": "multi-gpu, 0 1, 16384, 0, 0",
    }

    assert _target_process_ids(system, 0) == [101, 104]
    assert _target_process_ids(system, 1) == [102, 104]


def test_qvq_p32_amd_benchmark_rejects_malformed_rocm_process_data():
    with pytest.raises(ValueError, match="malformed KFD process fields"):
        _target_process_ids({"PID101": "missing-fields"}, 0)


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


def test_qvq_p32_amd_folded_shape_gate_is_fail_closed():
    for shape in (
        (5120, 12288),
        (5120, 1024),
        (5120, 10240),
        (5120, 6144),
        (6144, 5120),
        (5120, 17408),
        (17408, 5120),
    ):
        assert qvq_p32_amd_folded_shape_supported(*shape)
    for shape in ((256, 256), (8192, 2048)):
        assert not qvq_p32_amd_folded_shape_supported(*shape)


def test_qvq_p32_amd_folded_case_gate_enforces_accuracy_boundaries():
    assert qvq_p32_amd_folded_case_supported(4096, 5120, 12288)
    assert qvq_p32_amd_folded_case_supported(32, 6144, 5120)
    assert qvq_p32_amd_folded_case_supported(64, 6144, 5120)
    assert qvq_p32_amd_folded_case_supported(4096, 6144, 5120)
    assert not qvq_p32_amd_folded_case_supported(4097, 6144, 5120)
    assert qvq_p32_amd_folded_case_supported(512, 5120, 17408)
    assert not qvq_p32_amd_folded_case_supported(1024, 5120, 17408)
    assert qvq_p32_amd_folded_case_supported(512, 17408, 5120)
    assert qvq_p32_amd_folded_case_supported(1024, 17408, 5120)
    assert qvq_p32_amd_folded_case_supported(2048, 17408, 5120)
    assert qvq_p32_amd_folded_case_supported(4096, 17408, 5120)
    assert not qvq_p32_amd_folded_case_supported(4097, 17408, 5120)


def test_qvq_p32_amd_folded_output_dtype_gate_covers_measured_regressions():
    assert qvq_p32_amd_folded_prefers_fp32_output(4096, 5120, 1024)
    assert qvq_p32_amd_folded_prefers_fp32_output(512, 5120, 6144)
    assert not qvq_p32_amd_folded_prefers_fp32_output(2048, 5120, 1024)
    assert not qvq_p32_amd_folded_prefers_fp32_output(4096, 5120, 12288)


@pytest.mark.parametrize(
    ("m", "n", "k", "expected"),
    (
        (1, 4096, 4096, (16, 64, 8)),
        (16, 4096, 4096, (16, 64, 8)),
        (32, 4096, 4096, (32, 64, 8)),
        (64, 4096, 4096, (32, 64, 8)),
        (64, 12288, 5120, (64, 64, 8)),
        (128, 6144, 5120, (64, 64, 8)),
        (128, 12288, 5120, (128, 64, 8)),
        (256, 1024, 5120, (32, 64, 8)),
        (256, 17408, 5120, (256, 64, 8)),
        (512, 1024, 5120, (64, 64, 8)),
        (512, 12288, 5120, (512, 64, 8)),
        (512, 17408, 5120, (256, 64, 8)),
        (1024, 12288, 5120, (1024, 64, 8)),
        (1024, 17408, 5120, (512, 64, 8)),
        (2048, 1024, 5120, (128, 64, 8)),
        (2048, 4096, 4096, (512, 64, 8)),
        (4096, 1024, 5120, (256, 64, 8)),
        (4096, 4096, 4096, (512, 64, 8)),
    ),
)
def test_qvq_p32_amd_launch_config_covers_requested_regimes(m, n, k, expected):
    assert _launch_config(m, n, k) == expected


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
    with pytest.raises(TypeError, match="cache_weight must be boolean"):
        qvq_p32_amd(
            x,
            window,
            levels,
            bank_ids,
            3.0,
            out_features=256,
            bank_alt_id=3,
            cache_weight=1,
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


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("bits", P32_RATES)
def test_qvq_p32_amd_fused_opt_out_matches_reference(bits):
    x, planar, window, levels, bank_ids, bank_alt_id = _case(bits, 64, seed=6950)
    dense = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=256,
        out_features=256,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    actual = qvq_p32_amd(
        x,
        window,
        levels,
        bank_ids,
        bits,
        out_features=256,
        bank_alt_id=3,
        cache_weight=False,
    )
    torch.testing.assert_close(actual, x.float() @ dense, rtol=0.0, atol=2e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
def test_qvq_p32_amd_cache_reuses_and_invalidates_on_selector_mutation():
    bits = 3.0
    x, _, window, levels, bank_ids, _ = _case(bits, 17, seed=7950)
    first = qvq_p32_amd(x, window, levels, bank_ids, bits, out_features=256, bank_alt_id=3)
    first_dense = window._qvq_p32_amd_dense_cache[1]
    repeated = qvq_p32_amd(x, window, levels, bank_ids, bits, out_features=256, bank_alt_id=3)
    assert window._qvq_p32_amd_dense_cache[1] is first_dense
    assert torch.equal(first, repeated)

    bank_ids[0] ^= 1
    changed = qvq_p32_amd(x, window, levels, bank_ids, bits, out_features=256, bank_alt_id=3)
    second_dense = window._qvq_p32_amd_dense_cache[1]
    assert second_dense is not first_dense
    reference = x.float() @ second_dense.float().T
    torch.testing.assert_close(changed, reference, rtol=0.0, atol=2e-3)


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("bits", P32_RATES)
@pytest.mark.parametrize(
    ("input_hadamard", "output_hadamard"),
    ((True, True), (True, False), (False, True)),
)
@pytest.mark.parametrize("m", (1, 32))
def test_qvq_p32_amd_folded_full_layer_matches_fp32_oracle(
    bits,
    input_hadamard,
    output_hadamard,
    m,
):
    x, planar, window, levels, bank_ids, bank_alt_id = _case(bits, m, seed=8950)
    x = x * 0.1
    su = torch.linspace(0.75, 1.25, 256, dtype=torch.float32, device="cuda")
    sv = torch.linspace(1.25, 0.75, 256, dtype=torch.float32, device="cuda")
    su_half = su.half()
    sv_half = sv.half()
    inner = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=256,
        out_features=256,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    reference = x.float() * su
    if input_hadamard:
        reference = matmul_hadU(reference)
    reference = reference @ inner
    if output_hadamard:
        reference = matmul_hadU(reference)
    reference = reference * sv

    actual = qvq_p32_amd_folded(
        x,
        window,
        levels,
        bank_ids,
        su_half,
        sv_half,
        bits,
        out_features=256,
        bank_alt_id=3,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    repeated = qvq_p32_amd_folded(
        x,
        window,
        levels,
        bank_ids,
        su_half,
        sv_half,
        bits,
        out_features=256,
        bank_alt_id=3,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    narrowed = qvq_p32_amd_folded(
        x,
        window,
        levels,
        bank_ids,
        su_half,
        sv_half,
        bits,
        out_features=256,
        bank_alt_id=3,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
        output_fp32=False,
    )
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=2e-3)
    torch.testing.assert_close(repeated, reference, rtol=0.0, atol=2e-3)
    torch.testing.assert_close(narrowed.float(), reference, rtol=0.0, atol=2e-3)
    assert narrowed.dtype == torch.float16
    assert window._qvq_p32_amd_dense_cache is None

    qvq_p32_amd(
        x,
        window,
        levels,
        bank_ids,
        bits,
        out_features=256,
        bank_alt_id=3,
    )
    assert window._qvq_p32_amd_dense_cache is not None
    qvq_p32_amd_folded(
        x,
        window,
        levels,
        bank_ids,
        su_half,
        sv_half,
        bits,
        out_features=256,
        bank_alt_id=3,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    assert window._qvq_p32_amd_dense_cache is None


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("m", (1, 32))
def test_qvq_p32_amd_folded_residual_cache_reuses_and_matches_fp32_oracle(m):
    bits = 3.0
    x, planar, window, levels, bank_ids, bank_alt_id = _case(bits, m, seed=9450)
    su = torch.linspace(0.75, 1.25, 256, dtype=torch.float16, device="cuda")
    sv = torch.linspace(1.25, 0.75, 256, dtype=torch.float16, device="cuda")
    inner = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=256,
        out_features=256,
        bank_ids=bank_ids,
        v2b2_p32=True,
        bank_alt_id=bank_alt_id,
    )
    reference = matmul_hadU(x.float() * su.float()) @ inner
    reference = matmul_hadU(reference) * sv.float()

    with patch(
        "gptqmodel.utils.qvq_amd._QWEN38_27B_RESIDUAL_FOLDED_SHAPES",
        frozenset({(256, 256)}),
    ):
        actual = qvq_p32_amd_folded(
            x,
            window,
            levels,
            bank_ids,
            su,
            sv,
            bits,
            out_features=256,
            bank_alt_id=3,
            input_hadamard=True,
            output_hadamard=True,
        )
        cache = window._qvq_p32_amd_folded_cache
        folded, operand, residual, residual_operand, composite_recovery = cache[1:]
        repeated = qvq_p32_amd_folded(
            x,
            window,
            levels,
            bank_ids,
            su,
            sv,
            bits,
            out_features=256,
            bank_alt_id=3,
            input_hadamard=True,
            output_hadamard=True,
        )

    assert residual is not None
    assert residual_operand is not None
    assert composite_recovery is None
    assert operand.untyped_storage().data_ptr() == folded.untyped_storage().data_ptr()
    assert residual_operand.untyped_storage().data_ptr() == residual.untyped_storage().data_ptr()
    assert window._qvq_p32_amd_folded_cache is cache
    torch.testing.assert_close(actual, reference, rtol=0.0, atol=2e-3)
    assert torch.equal(actual, repeated)


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("buffer_name", ["trellis", "bank_ids", "bank_alt_id", "SU", "SV", "bias"])
@pytest.mark.parametrize("replacement", [False, True])
def test_qvq_linear_amd_folded_cache_reuses_and_invalidates_auxiliary_mutation(buffer_name, replacement):
    bits = 3.0
    k, n = 5120, 1024
    x, planar, _, _, bank_ids, bank_alt_id = _case(bits, 7, k=k, n=n, seed=9950)
    x = x * 0.1
    layer = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=True,
        output_hadamard=True,
        tensors={
            "trellis": planar,
            "SU": torch.ones(k, dtype=torch.float32, device="cuda"),
            "SV": torch.ones(n, dtype=torch.float32, device="cuda"),
            "bias": torch.linspace(-0.01, 0.01, n, dtype=torch.float16, device="cuda"),
            "bank_ids": bank_ids,
            "bank_alt_id": bank_alt_id,
        },
    ).eval()

    first = layer(x)
    window = layer._qvq_cuda_window_cache[3]
    first_folded = window._qvq_p32_amd_folded_cache[1]
    first_hot = layer._qvq_amd_folded_hot_cache
    assert first_hot is not None
    with patch(
        "gptqmodel.nn_modules.qlinear.qvq.qvq_transition_bits",
        side_effect=AssertionError("hot hit should not normalize an unchanged rate"),
    ):
        repeated = layer(x)
    assert window._qvq_p32_amd_folded_cache[1] is first_folded
    assert layer._qvq_amd_folded_hot_cache is first_hot
    assert torch.equal(first, repeated)

    source = getattr(layer, buffer_name)
    if replacement:
        replacement_tensor = source.clone()
        if buffer_name == "SU":
            # Parameter replacement must still use normal module resolution.
            replacement_tensor = torch.nn.Parameter(replacement_tensor, requires_grad=False)
        setattr(layer, buffer_name, replacement_tensor)
    elif buffer_name == "bank_alt_id":
        source.fill_(1)
    elif buffer_name in ("trellis", "bank_ids"):
        source.bitwise_xor_(1)
    elif buffer_name == "bias":
        source.add_(0.001)
    else:
        source.mul_(0.875)
    changed = layer(x)
    window = layer._qvq_cuda_window_cache[3]
    second_folded = window._qvq_p32_amd_folded_cache[1]
    if buffer_name != "bias":
        assert second_folded is not first_folded
    assert layer._qvq_amd_folded_hot_cache is not first_hot
    inner = layer.get_inner_weight_tensor()
    reference = matmul_hadU(x.float() * layer.SU) @ inner
    reference = matmul_hadU(reference) * layer.SV + layer.bias.float()
    torch.testing.assert_close(changed.float(), reference, rtol=0.0, atol=2e-3)
    layer.bits = 2.5
    with patch(
        "gptqmodel.nn_modules.qlinear.qvq.qvq_transition_bits",
        side_effect=RuntimeError("changed rate must be validated"),
    ), pytest.raises(RuntimeError, match="changed rate must be validated"):
        layer._qvq_amd_folded_forward(x, torch.float16)


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("bits", P32_RATES)
@pytest.mark.parametrize("has_bias", [False, True])
def test_attention_residual_cache_preserves_decode_prefill_transitions(bits, has_bias):
    from gptqmodel.utils.qvq_amd import _qvq_p32_folded_execute

    k, n = 6144, 5120
    _, planar, _, _, bank_ids, bank_alt_id = _case(bits, 1, k=k, n=n, seed=9964)
    layer = QVQLinear(
        bits=bits, in_features=k, out_features=n, bank_count=2, v2b2_p32=True,
        input_hadamard=True, output_hadamard=True,
        tensors={"trellis": planar, "bank_ids": bank_ids, "bank_alt_id": bank_alt_id,
                 "SU": torch.full((k,), .875, device="cuda"),
                 "SV": torch.full((n,), .75, device="cuda"),
                 "bias": torch.full((n,), .001, device="cuda", dtype=torch.float16) if has_bias else None},
    ).eval()
    inner = layer.get_inner_weight_tensor()
    generator = torch.Generator(device="cuda").manual_seed(9965)
    cache = None
    for m in (1, 64, 1, 4096, 32, 128):
        x = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator) * .01
        actual = layer(x)
        hot = layer._qvq_amd_folded_hot_cache
        assert hot[25] is not None
        assert hot[25].numel() * hot[25].element_size() == 2 * k * n
        if cache is not None:
            assert hot is cache
        cache = hot
        reference = matmul_hadU(matmul_hadU(x.float() * layer.SU) @ inner) * layer.SV
        if has_bias:
            reference = reference + layer.bias.float()
        torch.testing.assert_close(actual.float(), reference, atol=2e-3, rtol=0)
        if m <= 32:
            expected = _qvq_p32_folded_execute(
                x, hot[24], None, None, out_features=n, output_fp32=has_bias,
            )
            if has_bias:
                expected = expected + hot[23]
            assert torch.equal(actual, expected.to(torch.float16))


@pytest.mark.cuda
@pytest.mark.skipif(not _gfx950_available(), reason="requires a ROCm gfx950 GPU")
@pytest.mark.parametrize("bits", P32_RATES)
def test_down_composite_cache_large_m_transitions(bits):
    k, n = 17408, 5120
    _, planar, _, _, bank_ids, bank_alt_id = _case(bits, 1, k=k, n=n, seed=9970)
    layer = QVQLinear(
        bits=bits, in_features=k, out_features=n, bank_count=2, v2b2_p32=True,
        input_hadamard=False, output_hadamard=True,
        tensors={"trellis": planar, "bank_ids": bank_ids, "bank_alt_id": bank_alt_id,
                 "SU": torch.ones(k, device="cuda"), "SV": torch.full((n,), .75, device="cuda")},
    ).eval()
    inner = layer.get_inner_weight_tensor()
    generator = torch.Generator(device="cuda").manual_seed(9971)
    cache = None
    for m in (32, 2048, 4096, 32):
        x = torch.randn((m, k), device="cuda", dtype=torch.float16, generator=generator) * .01
        actual = layer(x)
        hot = layer._qvq_amd_folded_hot_cache
        assert hot[26] is not None
        if cache is not None:
            assert hot is cache
        cache = hot
        reference = matmul_hadU(x.float() @ inner) * layer.SV
        torch.testing.assert_close(actual.float(), reference, atol=2e-3, rtol=0)
