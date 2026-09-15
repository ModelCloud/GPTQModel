# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear import marlin as marlin_module
from gptqmodel.nn_modules.qlinear.marlin import (
    MarlinLinear,
    get_marlin_packed_prefill_route_stats,
    reset_marlin_packed_prefill_route_stats,
)
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.marlin import marlin_import_exception, marlin_runtime_available, marlin_runtime_error


BITS = 4
GROUP_SIZE = 128
IN_FEATURES = 288
OUT_FEATURES = 288


@pytest.fixture
def available_marlin_contract(monkeypatch):
    monkeypatch.setattr(marlin_module, "marlin_import_exception", None)
    monkeypatch.setattr(MarlinLinear, "cached_validate_once", classmethod(lambda _: (True, None)))


def _validation_kwargs(**overrides):
    kwargs = {
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "desc_act": False,
        "sym": True,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "pack_dtype": torch.int32,
        "dtype": torch.float16,
    }
    kwargs.update(overrides)
    return kwargs


def test_marlin_accepts_packable_32_aligned_padding_shape(available_marlin_contract):
    ok, error = MarlinLinear.validate(**_validation_kwargs())

    assert ok
    assert error is None


def test_marlin_padding_preserves_serialized_checkpoint_shapes_before_post_init(available_marlin_contract):
    module = MarlinLinear(**_validation_kwargs(), bias=True)

    assert module.in_features == 288
    assert module.out_features == 288
    assert module.padded_in_features == 384
    assert module.padded_out_features == 320
    assert module.qweight.shape == (36, 288)
    assert module.qzeros.shape == (3, 36)
    assert module.scales.shape == (3, 288)
    assert module.g_idx.shape == (288,)
    assert module.bias.shape == (288,)


def test_marlin_padding_rejects_unexpected_input_width_before_launch(available_marlin_contract):
    module = MarlinLinear(**_validation_kwargs(), bias=False)

    with pytest.raises(ValueError, match="expected input width 288, got 256"):
        module(torch.zeros((2, 256), dtype=torch.float16))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"out_features": 290}, "out_features"),
        ({"desc_act": True}, "desc_act=True"),
        ({"adapter": Lora(rank=16, path="unused")}, "adapters"),
    ],
)
def test_marlin_padding_rejects_unsupported_contracts(available_marlin_contract, overrides, match):
    ok, error = MarlinLinear.validate(**_validation_kwargs(**overrides))

    assert not ok
    assert error is not None
    assert match in str(error)


def _build_padded_marlin(
    device: torch.device,
    in_features: int,
    out_features: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[MarlinLinear, nn.Linear]:
    generator = torch.Generator(device="cpu").manual_seed(898)
    groups = (in_features + GROUP_SIZE - 1) // GROUP_SIZE
    scales = 0.01 + 0.04 * torch.rand((out_features, groups), generator=generator)
    zeros = torch.full_like(scales, 8.0)
    codes = torch.randint(0, 16, (out_features, in_features), generator=generator)
    g_idx = torch.arange(in_features, dtype=torch.int32) // GROUP_SIZE
    weight = (codes.to(torch.float32) - 8.0) * scales[:, g_idx.long()]
    bias = 0.02 * torch.randn(out_features, generator=generator)

    dense = nn.Linear(in_features, out_features, bias=True, dtype=dtype)
    dense.weight.data.copy_(weight)
    dense.bias.data.copy_(bias)

    packed = TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        dtype=dtype,
    )
    packed.pack(dense, scales, zeros, g_idx, workers=1)

    marlin = MarlinLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=True,
        pack_dtype=torch.int32,
        dtype=dtype,
    )
    for name in ("qweight", "qzeros", "scales", "g_idx", "bias"):
        getattr(marlin, name).data.copy_(getattr(packed, name).data)
    marlin = marlin.to(device)
    marlin.post_init()
    return marlin, dense.to(device)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("in_features", "out_features", "input_prefix"),
    [
        (288, 320, (3,)),
        (256, 288, (3,)),
        (288, 288, (3,)),
        (288, 288, (2, 5)),
    ],
)
def test_marlin_runtime_padding_matches_dense_and_restores_shape(in_features, out_features, input_prefix):
    if marlin_import_exception is not None:
        pytest.skip(f"Marlin kernel unavailable: {marlin_import_exception}")
    if not marlin_runtime_available(torch.float16):
        pytest.skip(marlin_runtime_error(torch.float16))

    device = torch.device("cuda:0")
    marlin, dense = _build_padded_marlin(device, in_features, out_features)
    generator = torch.Generator(device=device).manual_seed(898)
    inputs = torch.randn(input_prefix + (in_features,), generator=generator, device=device, dtype=torch.float16)

    expected = dense(inputs)
    actual = marlin(inputs)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert marlin.padded_in_features == ((in_features + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    assert marlin.padded_out_features == ((out_features + 63) // 64) * 64


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("rows", (64, 65, 513))
@pytest.mark.parametrize("seed", (898, 1729))
def test_marlin_packed_prefill_kn_and_m_tails_match_dense(dtype, rows, seed):
    if marlin_import_exception is not None:
        pytest.skip(f"Marlin kernel unavailable: {marlin_import_exception}")
    if not marlin_runtime_available(dtype):
        pytest.skip(marlin_runtime_error(dtype))

    device = torch.device("cuda:0")
    marlin, dense = _build_padded_marlin(device, IN_FEATURES, OUT_FEATURES, dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    inputs = torch.randn((rows, IN_FEATURES), generator=generator, device=device, dtype=dtype)
    expected = dense(inputs)

    marlin.packed_prefill_min_rows = 1
    marlin.packed_prefill_stats = True
    reset_marlin_packed_prefill_route_stats()
    auto_output = marlin(inputs)
    auto_stats = get_marlin_packed_prefill_route_stats(reset=True)
    assert auto_stats["auto_hits"] == 0
    assert auto_stats["by_reason"] == {"contract_miss": 1}

    marlin.packed_prefill_stats = False
    marlin.packed_prefill_config = 1
    packed_output = marlin(inputs)
    marlin.packed_prefill = False
    ordinary_output = marlin(inputs)

    assert packed_output.shape == expected.shape
    assert packed_output.dtype == expected.dtype
    assert torch.isfinite(packed_output).all()
    torch.testing.assert_close(auto_output, ordinary_output, rtol=0, atol=0)
    torch.testing.assert_close(packed_output, ordinary_output, rtol=0.005, atol=0.05)
    torch.testing.assert_close(packed_output, expected, rtol=0.03, atol=0.125)
