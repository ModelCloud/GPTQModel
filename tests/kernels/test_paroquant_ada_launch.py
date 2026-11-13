# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for every table/dispatch case; the kernel-oracle cases carry
# ``pytest.mark.cuda`` and skip unless the device is an Ada (SM 8.9) GPU.
"""Ada (RTX 4090, SM 8.9) fused ParoQuant GEMM: launch table + Torch-oracle A/B.

The oracle decodes the packed int32 words with Torch integer shifts, multiplies
by the FP16 group scales in FP64 and runs an FP64 matmul; it never calls the
Triton kernel or reuses its intermediates.
"""

import pytest
import torch

from gptqmodel.quantization.paroquant.modules.triton import gemm as paro_gemm
from gptqmodel.utils import arch


TOL = 2e-3
GROUP_SIZE = 128
ADA_TILES = {
    "decode": (16, 64, 64, 4, 4),
    "prefill": (32, 64, 64, 8, 4),
    "prefill_large": (64, 64, 64, 8, 4),
}


@pytest.fixture(autouse=True)
def _arch_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


def _is_ada_host() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and arch.is_ada(0)
    )


requires_ada = pytest.mark.skipif(not _is_ada_host(), reason="Ada (SM 8.9) CUDA GPU required")


# --------------------------------------------------------------------------- #
# Hardware-free: the table itself, its routing and the row bands.
# --------------------------------------------------------------------------- #

def test_ada_entry_is_present_and_measured():
    table = paro_gemm._PAROQUANT_GEMM_LAUNCH_CONFIGS
    assert "ada" in table
    # Every band must be recorded explicitly: a family that "wins with the
    # historical default" still gets an entry rather than the fallback.
    assert set(table["ada"]) == {"decode", "prefill", "prefill_large"}
    assert dict(table["ada"]) == ADA_TILES


def test_rdna4_entry_is_untouched_by_the_ada_addition():
    table = paro_gemm._PAROQUANT_GEMM_LAUNCH_CONFIGS
    assert table["rdna4"] == {
        "decode": (4, 64, 64, 4, 1),
        "prefill": (32, 64, 64, 4, 2),
        "prefill_large": (64, 128, 64, 4, 2),
    }
    assert paro_gemm._PAROQUANT_GEMM_DEFAULT_CONFIGS == {
        "decode": (4, 128, 32, 4, 2),
        "prefill": (32, 128, 32, 8, 4),
    }


@pytest.mark.parametrize("target", [(8, 9), "sm_89", "sm_89a"])
def test_ada_is_selected_for_every_ada_spelling(target):
    assert arch.tuning_family(target) == "ada"
    # Without rows the label's tile is returned; the label still means
    # "decode rows" vs "prefill rows".
    assert (
        paro_gemm.paroquant_gemm_launch_config("decode", target)
        == ADA_TILES["decode"]
    )
    assert (
        paro_gemm.paroquant_gemm_launch_config("prefill", target)
        == ADA_TILES["prefill"]
    )


@pytest.mark.parametrize(
    ("rows", "band"),
    [(1, "decode"), (8, "decode"), (9, "prefill"), (64, "prefill"),
     (65, "prefill_large"), (1024, "prefill_large")],
)
def test_ada_row_bands(rows, band):
    kind = "decode" if rows <= 8 else "prefill"
    assert (
        paro_gemm.paroquant_gemm_launch_config(kind, (8, 9), rows=rows)
        == ADA_TILES[band]
    )


@pytest.mark.parametrize(
    "target", [(7, 5), (8, 0), (8, 6), (9, 0), (10, 0), (12, 0), "sm_80", "sm_90",
               "gfx942", "gfx1100", ""]
)
def test_non_ada_families_keep_the_historical_default(target):
    # ``None`` is deliberately absent: it means "the current device", which on
    # this host is the Ada GPU and must route to the Ada tile.
    assert arch.tuning_family(target) != "ada"
    for kind in ("decode", "prefill"):
        assert (
            paro_gemm.paroquant_gemm_launch_config(kind, target)
            == paro_gemm._PAROQUANT_GEMM_DEFAULT_CONFIGS[kind]
        )
        for rows in (1, 8, 9, 64, 65, 4096):
            assert (
                paro_gemm.paroquant_gemm_launch_config(kind, target, rows=rows)
                == paro_gemm._PAROQUANT_GEMM_DEFAULT_CONFIGS[kind]
            )


def test_current_device_routes_to_ada_on_an_ada_host():
    if not _is_ada_host():
        pytest.skip("Ada (SM 8.9) CUDA GPU required")
    arch.clear_arch_cache()
    assert arch.tuning_family(None) == "ada"
    assert paro_gemm.paroquant_gemm_launch_config("decode", None, rows=1) == ADA_TILES["decode"]


def test_rdna4_routing_is_unchanged():
    table = paro_gemm._PAROQUANT_GEMM_LAUNCH_CONFIGS
    for rows, band in ((8, "decode"), (9, "prefill"), (65, "prefill_large")):
        kind = "decode" if rows <= 8 else "prefill"
        assert (
            paro_gemm.paroquant_gemm_launch_config(kind, "gfx1201", rows=rows)
            == table["rdna4"][band]
        )
    # The medium/large boundary constant still means what it did.
    assert paro_gemm._RDNA4_MEDIUM_MAX_ROWS == 64


def test_ada_tiles_differ_from_the_historical_fallback():
    # The whole point of the entry: every Ada band is a measured tile, not the
    # historical default.
    for band, kind in (("decode", "decode"), ("prefill", "prefill"),
                       ("prefill_large", "prefill")):
        assert ADA_TILES[band] != paro_gemm._PAROQUANT_GEMM_DEFAULT_CONFIGS[kind]


# --------------------------------------------------------------------------- #
# Hardware-free oracle helpers (independently decode the packing).
# --------------------------------------------------------------------------- #

def _awq_pack(codes: torch.Tensor) -> torch.Tensor:
    """AWQ [0,2,4,6,1,3,5,7] nibble order over the fast axis (Torch only)."""

    packed = torch.zeros(
        (codes.shape[0], codes.shape[1] // 8), dtype=torch.int32, device=codes.device
    )
    for i, order in enumerate([0, 2, 4, 6, 1, 3, 5, 7]):
        packed |= (codes[:, order::8].to(torch.int32) << (i * 4))
    return packed


def _awq_unpack(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=packed.device)
    shifts = torch.arange(0, 32, bits, device=packed.device, dtype=torch.int64)
    codes = (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF
    return codes.index_select(-1, order).flatten(-2)


def _paro_oracle(codes, zeros, scales, group_size, inputs):
    """FP64 ``inputs @ W`` oracle straight from the unpacked codes."""

    k = codes.shape[0]
    group = torch.arange(k, device=codes.device) // group_size
    dense = (codes.to(torch.float64) - zeros.to(torch.float64)[group]) * scales.to(
        torch.float64
    )[group]
    return inputs.to(torch.float64) @ dense


def _packed_inputs(k, n, group_size, seed):
    torch.manual_seed(seed)
    codes = torch.randint(0, 16, (k, n), dtype=torch.int32, device="cuda:0")
    zeros = torch.randint(0, 16, (k // group_size, n), dtype=torch.int32, device="cuda:0")
    scales = (
        torch.rand((k // group_size, n), dtype=torch.float16, device="cuda:0") * 0.05
        + 0.005
    )
    return codes, zeros, scales, _awq_pack(codes), _awq_pack(zeros)


# --------------------------------------------------------------------------- #
# Real forward on this 4090 vs the independent FP64 oracle.
# --------------------------------------------------------------------------- #

@pytest.mark.cuda
@requires_ada
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize(
    "rows,k,n,band",
    [
        (1, 512, 512, "decode"),
        (8, 2048, 2048, "decode"),
        (16, 2048, 2048, "prefill"),
        (64, 4096, 4096, "prefill"),
        (65, 4096, 4096, "prefill_large"),
        (128, 14336, 4096, "prefill_large"),
    ],
)
def test_ada_paroquant_gemm_matches_fp64_oracle(rows, k, n, band, dtype):
    codes, zeros, scales, qweight, qzeros = _packed_inputs(k, n, GROUP_SIZE, 7 + rows)
    inputs = torch.randn(rows, k, dtype=dtype, device="cuda:0")

    assert (
        paro_gemm.paroquant_gemm_launch_config("decode" if rows <= 8 else "prefill",
                                               inputs.device, rows=rows)
        == ADA_TILES[band]
    )

    with torch.inference_mode():
        first = paro_gemm.paroquant_gemm_triton_decode(
            inputs, qweight, scales, qzeros
        ) if rows <= 8 else paro_gemm.paroquant_gemm_triton_prefill(
            inputs, qweight, scales, qzeros
        )
        repeated = paro_gemm.paroquant_gemm_triton_decode(
            inputs, qweight, scales, qzeros
        ) if rows <= 8 else paro_gemm.paroquant_gemm_triton_prefill(
            inputs, qweight, scales, qzeros
        )
    torch.cuda.synchronize()

    truth = _paro_oracle(codes, zeros, scales, GROUP_SIZE, inputs)
    assert first.dtype == dtype
    assert torch.equal(first, repeated)
    scale = truth.abs().max().item()
    err = (first.to(torch.float64) - truth).abs().max().item()
    assert err / scale <= TOL, f"max rel err {err / scale:.3e} > {TOL}"
    rounded = truth.to(dtype).to(torch.float64)
    assert (first.to(torch.float64) - rounded).abs().max().item() <= TOL * scale


@pytest.mark.cuda
@requires_ada
def test_ada_paroquant_gemm_keeps_no_dense_weight_after_forward():
    from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear

    k = n = 512
    codes, zeros, scales, qweight, qzeros = _packed_inputs(k, n, GROUP_SIZE, 11)
    module = ParoQuantTritonLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=k,
        out_features=n,
        bias=False,
        register_buffers=True,
    ).to("cuda:0").eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.post_init()

    inputs = torch.randn(4, k, dtype=torch.float16, device="cuda:0")
    with torch.inference_mode():
        module(inputs)
        torch.cuda.synchronize()
        after_one = _dense_weight_attrs(module)
        for _ in range(3):
            module(inputs)
        torch.cuda.synchronize()
        after_many = _dense_weight_attrs(module)

    assert after_one == set(), f"dense weight retained after first forward: {after_one}"
    assert after_many == set(), f"dense weight retained after repeated forwards: {after_many}"
    # The packed representation stays quantized.
    assert module.qweight.dtype == torch.int32 and module.qweight.shape == (k, n // 8)


def _dense_weight_attrs(module) -> set:
    """Names of buffers/attributes holding a dense (K, N) float weight."""

    found = set()
    for name, value in list(module.__dict__.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if value.dim() == 2 and value.shape[0] == module.in_features:
                found.add(name)
    for name, value in module._buffers.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if value.dim() == 2 and value.shape[0] == module.in_features:
                found.add(name)
    return found
