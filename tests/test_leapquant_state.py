# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Numerical tests for the LeapQuant state quantizer."""

import pytest
import torch

from gptqmodel.utils.leapquant import (
    LeapQuantConfig,
    LeapQuantState,
    mask_is_unpadded,
)


def _devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _fixture(device, sequences=2, heads=2, key_dim=8, value_dim=8, tokens=12, seed=7):
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def draw(*shape):
        return torch.randn(*shape, generator=generator).to(device)

    query = draw(sequences, tokens, heads, key_dim)
    key = draw(sequences, tokens, heads, key_dim)
    value = draw(sequences, tokens, heads, value_dim)
    alpha = (torch.rand(sequences, tokens, heads, generator=generator) * 0.3 + 0.7).to(device)
    beta = (torch.rand(sequences, tokens, heads, generator=generator) * 0.8 + 0.1).to(device)
    return query, key, value, alpha, beta


def _oracle_gdn(query, key, value, alpha, beta):
    state = torch.zeros(
        query.shape[0], query.shape[2], query.shape[3], value.shape[3],
        dtype=torch.float32, device=query.device,
    )
    outputs = []
    for index in range(query.shape[1]):
        state = state * alpha[:, index][..., None, None]
        prediction = (state * key[:, index][..., None]).sum(dim=-2)
        update = beta[:, index][..., None] * (value[:, index] - prediction)
        state = state + key[:, index][..., None] * update[..., None, :]
        outputs.append((state * query[:, index][..., None]).sum(dim=-2))
    return torch.stack(outputs, dim=1), state


def _oracle_kda(query, key, value, alpha, beta):
    state = torch.zeros(
        query.shape[0], query.shape[2], query.shape[3], value.shape[3],
        dtype=torch.float32, device=query.device,
    )
    outputs = []
    for index in range(query.shape[1]):
        state = state * alpha[:, index][..., None]
        prediction = (state * key[:, index][..., None]).sum(dim=-2)
        update = beta[:, index][..., None] * (value[:, index] - prediction)
        state = state + key[:, index][..., None] * update[..., None, :]
        outputs.append((state * query[:, index][..., None]).sum(dim=-2))
    return torch.stack(outputs, dim=1), state


def _relative(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("bits", [4, 6, 8])
def test_gdn_step_tracks_fp32_oracle(device, bits):
    query, key, value, alpha, beta = _fixture(device)
    expected, expected_state = _oracle_gdn(query, key, value, alpha, beta)
    state = LeapQuantState(
        2, 2, 8, 8,
        LeapQuantConfig(bits=bits, window=4, rank=2, buffer_dtype=torch.float32),
        gate_dim=1,
        device=device,
    )
    actual = torch.cat(
        [
            state.step(query[:, index], key[:, index], value[:, index], alpha[:, index], beta[:, index]).unsqueeze(1)
            for index in range(query.shape[1])
        ],
        dim=1,
    )
    limit = {4: 8e-2, 6: 2e-2, 8: 5e-3}[bits]
    assert _relative(actual, expected) < limit
    assert _relative(state.materialize(), expected_state) < limit * 2


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("bits", [4, 6, 8])
def test_kda_step_tracks_fp32_oracle(device, bits):
    query, key, value, alpha_scalar, beta = _fixture(device)
    alpha = alpha_scalar[..., None].expand(-1, -1, -1, query.shape[-1])
    expected, expected_state = _oracle_kda(query, key, value, alpha, beta)
    state = LeapQuantState(
        2, 2, 8, 8,
        LeapQuantConfig(bits=bits, window=4, rank=2, buffer_dtype=torch.float32),
        gate_dim=8,
        device=device,
    )
    actual = torch.cat(
        [
            state.step(query[:, index], key[:, index], value[:, index], alpha[:, index], beta[:, index]).unsqueeze(1)
            for index in range(query.shape[1])
        ],
        dim=1,
    )
    limit = {4: 8e-2, 6: 2e-2, 8: 5e-3}[bits]
    assert _relative(actual, expected) < limit
    assert _relative(state.materialize(), expected_state) < limit * 2


@pytest.mark.parametrize("device", _devices())
def test_window_flushes_only_at_the_boundary(device):
    query, key, value, alpha, beta = _fixture(device, tokens=6)
    state = LeapQuantState(
        2, 2, 8, 8,
        LeapQuantConfig(bits=8, window=3, rank=2, buffer_dtype=torch.float32),
        gate_dim=1,
        device=device,
    )
    for index in range(2):
        state.step(query[:, index], key[:, index], value[:, index], alpha[:, index], beta[:, index])
        assert state.window_count == index + 1
        assert torch.count_nonzero(state.buffer_keys[:, :, index]) > 0
    state.step(query[:, 2], key[:, 2], value[:, 2], alpha[:, 2], beta[:, 2])
    assert state.window_count == 0
    assert torch.count_nonzero(state.buffer_keys) == 0
    assert torch.count_nonzero(state.codes) > 0


@pytest.mark.parametrize("device", _devices())
def test_full_rank_compensator_reduces_residual(device):
    generator = torch.Generator(device="cpu").manual_seed(17)
    dense = torch.randn(1, 2, 4, 4, generator=generator).to(device)
    full = LeapQuantState(1, 2, 4, 4, LeapQuantConfig(bits=8, window=2, rank=4), gate_dim=1, device=device)
    none = LeapQuantState(1, 2, 4, 4, LeapQuantConfig(bits=8, window=2, rank=0), gate_dim=1, device=device)
    full.quantize_(dense)
    none.quantize_(dense)
    full_error = (full.dequantize() - dense).norm()
    none_error = (none.dequantize() - dense).norm()
    assert full_error < none_error
    assert full.factors_key.shape[-1] == 4
    assert none.factors_key.shape[-1] == 0


@pytest.mark.parametrize(
    "buffer_dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_buffer_dtypes_are_finite(buffer_dtype):
    query, key, value, alpha, _beta = _fixture("cpu", tokens=3)
    state = LeapQuantState(
        2, 2, 8, 8,
        LeapQuantConfig(bits=8, window=4, rank=2, buffer_dtype=buffer_dtype),
        gate_dim=1,
        device="cpu",
    )
    for index in range(3):
        output = state.step(
            query[:, index], key[:, index], value[:, index], alpha[:, index],
            torch.full((2, 2), 0.5),
        )
        assert torch.isfinite(output).all()
    assert torch.isfinite(state.materialize()).all()


def test_packed_state_roundtrip_and_storage_accounting():
    query, key, value, alpha, _beta = _fixture("cpu", tokens=3)
    config = LeapQuantConfig(bits=8, window=4, rank=2)
    state = LeapQuantState(2, 2, 8, 8, config, gate_dim=1, device="cpu")
    for index in range(3):
        state.step(query[:, index], key[:, index], value[:, index], alpha[:, index], torch.full((2, 2), 0.5))
    packed = state.packed_state()
    restored = LeapQuantState(2, 2, 8, 8, config, gate_dim=1, device="cpu")
    restored.load_packed_state(packed)
    assert torch.equal(restored.codes, state.codes)
    assert torch.equal(restored.buffer_keys, state.buffer_keys)
    assert restored.window_count == state.window_count
    assert torch.equal(restored.materialize(), state.materialize())
    assert state.storage_bytes() == state.checkpoint_bytes() + state.buffer_bytes()
    assert state.buffer_bytes() > 0


def test_reorder_is_atomic():
    query, key, value, alpha, _beta = _fixture("cpu", tokens=2)
    state = LeapQuantState(2, 2, 8, 8, LeapQuantConfig(bits=8, window=4, rank=2), gate_dim=1, device="cpu")
    for index in range(2):
        state.step(query[:, index], key[:, index], value[:, index], alpha[:, index], torch.full((2, 2), 0.5))
    before = state.packed_state()
    index = torch.tensor([1, 0])
    state.reorder_(index)
    for name in ("codes", "key_scale", "value_scale", "factors_key", "factors_value", "buffer_keys"):
        assert torch.equal(getattr(state, name), before[name].index_select(0, index))


def test_nonfinite_values_fail_closed():
    state = LeapQuantState(1, 1, 4, 4, LeapQuantConfig(bits=8, window=2, rank=1), gate_dim=1, device="cpu")
    bad = torch.full((1, 1, 4, 4), float("nan"))
    with pytest.raises(ValueError):
        state.quantize_(bad)
    with pytest.raises(ValueError):
        state.step(
            torch.full((1, 1, 4), float("inf")),
            torch.ones(1, 1, 4),
            torch.ones(1, 1, 4),
            torch.ones(1, 1),
            torch.ones(1, 1),
        )


def test_config_and_mask_validation():
    with pytest.raises(ValueError):
        LeapQuantConfig(bits=5)
    with pytest.raises(ValueError):
        LeapQuantConfig(window=0)
    with pytest.raises(ValueError):
        LeapQuantState(1, 1, 4, 4, LeapQuantConfig(rank=5), gate_dim=1, device="cpu")
    assert mask_is_unpadded(None)
    assert mask_is_unpadded(torch.ones(1, 4, dtype=torch.bool))
    assert not mask_is_unpadded(torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    assert not mask_is_unpadded(torch.tensor([[1.0, 1.0, float("-inf")]]))
