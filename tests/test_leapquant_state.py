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


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("window", [2, 3])
def test_failed_step_preserves_live_window_for_retry(device, window):
    state = LeapQuantState(
        1, 1, 1, 1,
        LeapQuantConfig(window=window, rank=1, buffer_dtype=torch.float16),
        device=device,
    )
    unit = torch.ones((1, 1, 1), device=device)
    state.step(unit, unit, unit * 2, unit, unit)
    before = state.packed_state()

    # Window 2 fails while encoding the compensator. Window 3 would store an overflowing update.
    value = unit * (2 ** 34)
    with pytest.raises(ValueError, match="non-finite"):
        state.step(unit, unit, value, unit, unit)

    after = state.packed_state()
    for name, previous in before.items():
        current = after[name]
        if isinstance(previous, torch.Tensor):
            torch.testing.assert_close(current, previous, atol=0, rtol=0)
        else:
            assert current == previous
    torch.testing.assert_close(state.step(unit, unit, unit * 4, unit, unit), unit * 4, atol=0, rtol=0)
    torch.testing.assert_close(state.materialize(), (unit * 4).unsqueeze(-1), atol=0, rtol=0)
    assert state.window_count == (2 % window)


@pytest.mark.parametrize("device", _devices())
def test_compensator_stays_fp32_under_autocast(device):
    state = LeapQuantState(1, 1, 2, 2, LeapQuantConfig(rank=2), device=device)
    expected = torch.tensor([[[[65536.0, 0.0], [0.0, 4.0]]]], device=device)

    with torch.autocast(device_type=expected.device.type, dtype=torch.float16):
        state.quantize_(expected)
        actual = state.materialize()

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("invalid_field", ["buffer_dtype", "buffer_alpha", "window_count", "codes", "key_scale"])
def test_packed_restore_validates_before_replacing_live_state(invalid_field):
    config = LeapQuantConfig(window=3, rank=0, smooth=False, buffer_dtype=torch.float32)
    state = LeapQuantState(1, 1, 1, 1, config)
    unit = torch.ones(1, 1, 1)
    state.step(unit, unit, unit * 2, unit, unit)
    donor = LeapQuantState(1, 1, 1, 1, config)
    donor.quantize_((unit * 8).unsqueeze(-1))
    packed = donor.packed_state()
    valid = packed[invalid_field]
    packed[invalid_field] = {
        "buffer_dtype": str(torch.float16),
        "buffer_alpha": torch.empty(1),
        "window_count": config.window,
        "codes": packed["codes"].float(),
        "key_scale": torch.full_like(packed["key_scale"], float("nan")),
    }[invalid_field]

    with pytest.raises(ValueError, match=invalid_field):
        state.load_packed_state(packed)

    torch.testing.assert_close(state.materialize(), (unit * 2).unsqueeze(-1), atol=0, rtol=0)
    assert state.window_count == 1
    packed[invalid_field] = valid
    state.load_packed_state(packed)
    torch.testing.assert_close(state.materialize(), (unit * 8).unsqueeze(-1), atol=0, rtol=0)
    actual = state.step(unit, unit, unit * 4, unit, unit * 0.5)
    torch.testing.assert_close(actual, unit * 6, atol=0, rtol=0)


@pytest.mark.parametrize("field", ["key", "value", "alpha", "beta"])
@pytest.mark.parametrize("length", [1, 3])
def test_prefill_rejects_mismatched_lengths_before_consuming_tokens(field, length):
    inputs = dict(zip(("query", "key", "value", "alpha", "beta"), _fixture("cpu", tokens=3), strict=True))
    prompt = {name: tensor[:, 1:] for name, tensor in inputs.items()}
    state = LeapQuantState(2, 2, 8, 8, LeapQuantConfig(window=4, rank=0, buffer_dtype=torch.float32), gate_dim=1)
    state.step(*(tensor[:, 0] for tensor in inputs.values()))
    previous = state.materialize()

    with pytest.raises(ValueError, match="sequence length"):
        state.prefill(**{**prompt, field: inputs[field][:, :length]})

    assert state.window_count == 1
    torch.testing.assert_close(state.materialize(), previous, rtol=0, atol=0)
    expected, _ = _oracle_gdn(**inputs)
    torch.testing.assert_close(state.prefill(**prompt), expected[:, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("bad_value", [float("nan"), 2 ** 20])
def test_failed_prefill_preserves_checkpoint_and_window_for_retry(device, bad_value):
    state = LeapQuantState(
        1, 1, 1, 1, LeapQuantConfig(rank=0, window=2, smooth=False, buffer_dtype=torch.float16), device=device,
    )
    unit = torch.ones(1, 1, 1, device=device)
    state.step(unit, unit, unit * 2, unit, unit)
    before = state.packed_state()
    prompt = unit.unsqueeze(1).expand(-1, 2, -1, -1)
    values = prompt.clone()
    values[:, 1] = bad_value
    with pytest.raises(ValueError, match="non-finite"):
        state.prefill(prompt, prompt, values, prompt, prompt * 0.5)

    after = state.packed_state()
    for name, previous in before.items():
        current = after[name]
        if isinstance(previous, torch.Tensor):
            torch.testing.assert_close(current, previous, rtol=0, atol=0)
        else:
            assert current == previous
    # Starting from 2, beta=0.5 moves the state halfway toward 1 at each token.
    expected = torch.tensor([1.5, 1.25], device=device).reshape_as(prompt)
    torch.testing.assert_close(state.prefill(prompt, prompt, prompt, prompt, prompt * 0.5), expected, rtol=0, atol=0)
    torch.testing.assert_close(state.materialize(), expected[:, -1:], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a non-CPU default device")
def test_randomized_compensator_respects_explicit_generator_device():
    state = LeapQuantState(1, 1, 4, 4, LeapQuantConfig(rank=1, window=1), device="cuda")
    key = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda").reshape(1, 1, 4)
    value = key.flip(-1)
    gate = torch.ones(1, 1, device="cuda")
    with torch.device(key.device):
        state.step(key, key, value, gate, gate)
    expected = key.unsqueeze(-1) * value.unsqueeze(-2)
    torch.testing.assert_close(state.materialize(), expected, rtol=2e-3, atol=2e-3)
