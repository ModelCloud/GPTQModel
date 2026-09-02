# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear, qvq_dense_oracle_forward
from gptqmodel.quantization.qvq import (
    pack_qvq_binary_bank_ids,
    pack_trellis_states,
    qvq_transition_bits,
)
from gptqmodel.quantization.qvq_grouped import (
    QVQExecutionDescriptor,
    QVQGroupedP32Spec,
    QVQGroupedRuntimeOracle,
    canonical_child_p32_payload,
    group_canonical_p32_payloads,
    qvq_torch_child_oracle,
    qvq_torch_group_oracle,
    ungroup_canonical_p32_payload,
    validate_group,
)


def _tail_biting_states(bits: float, *, tiles: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shift = qvq_transition_bits(bits)
    edges = torch.randint(0, 1 << shift, (tiles, 128), generator=generator, dtype=torch.int64)
    stream = []
    for bit in range(shift - 1, -1, -1):
        stream.append((edges >> bit) & 1)
    stream = torch.stack(stream, dim=-1).reshape(tiles, -1)
    states = torch.empty((tiles, 128), dtype=torch.int64)
    for step in range(128):
        end = (step + 1) * shift
        offsets = torch.arange(end - 16, end) % stream.shape[1]
        state = torch.zeros(tiles, dtype=torch.int64)
        for offset in offsets:
            state = (state << 1) | stream[:, offset]
        states[:, step] = state
    return states


def _child(
    name: str,
    *,
    out_features: int,
    bits: float = 2.0,
    in_features: int = 32,
    su: torch.Tensor | None = None,
    alt_id: int = 1,
    seed: int = 0,
) -> QVQLinear:
    tiles = (in_features // 16) * (out_features // 16)
    states = _tail_biting_states(bits, tiles=tiles, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    if su is None:
        su = torch.randn(in_features, generator=generator)
    tensors = {
        "trellis": pack_trellis_states(states, bits=bits),
        "SU": su.clone(),
        "SV": torch.randn(out_features, generator=generator),
        "bias": torch.randn(out_features, generator=generator),
        "bank_ids": pack_qvq_binary_bank_ids(
            torch.randint(0, 2, (tiles * 8,), generator=generator, dtype=torch.uint8)
        ),
        "bank_alt_id": torch.tensor([alt_id], dtype=torch.uint8),
    }
    return QVQLinear.from_tensors(
        bits=bits,
        in_features=in_features,
        out_features=out_features,
        name=name,
        tensors=tensors,
        bank_count=2,
        v2b2_p32=True,
    )


def _descriptors(*names: str, basis: str = "attention.input"):
    return tuple(QVQExecutionDescriptor(name, basis) for name in names)


@pytest.mark.parametrize("bits", (2.0, 2.5, 3.0, 3.5))
def test_grouped_p32_payload_round_trip_is_byte_exact_and_storage_neutral(bits):
    su = torch.tensor([(-1.0) ** index for index in range(32)])
    children = (
        _child("q_proj", out_features=32, bits=bits, su=su, alt_id=1, seed=10),
        _child("k_proj", out_features=48, bits=bits, su=su, alt_id=2, seed=11),
        _child("v_proj", out_features=16, bits=bits, su=su, alt_id=3, seed=12),
    )
    descriptors = _descriptors("q_proj", "k_proj", "v_proj")
    grouped = group_canonical_p32_payloads(children, descriptors)

    assert grouped.trellis.shape == (12, children[0].trellis.shape[1])
    assert grouped.bank_ids.shape == (12,)
    assert [segment.output_tile_start for segment in grouped.segments] == [0, 2, 5]
    assert [segment.output_tile_count for segment in grouped.segments] == [2, 3, 1]
    assert grouped.out_features == 96

    for index, child in enumerate(children):
        original = canonical_child_p32_payload(child)
        recovered = ungroup_canonical_p32_payload(grouped, index)
        assert torch.equal(original.trellis, recovered.trellis)
        assert torch.equal(original.bank_ids, recovered.bank_ids)
        assert torch.equal(original.bank_alt_id, recovered.bank_alt_id)
        assert original.bits == recovered.bits
        assert original.in_features == recovered.in_features
        assert original.out_features == recovered.out_features


@pytest.mark.parametrize("bits", (2.0, 2.5, 3.0, 3.5))
def test_grouped_p32_oracle_is_exactly_equal_to_independent_dense_oracles(bits):
    su = torch.randn(32, generator=torch.Generator().manual_seed(20))
    children = (
        _child("q_proj", out_features=32, bits=bits, su=su, alt_id=1, seed=21),
        _child("k_proj", out_features=48, bits=bits, su=su, alt_id=2, seed=22),
        _child("v_proj", out_features=16, bits=bits, su=su, alt_id=3, seed=23),
    )
    descriptors = _descriptors("q_proj", "k_proj", "v_proj")
    x = torch.randn((2, 3, 32), generator=torch.Generator().manual_seed(24))

    grouped = qvq_torch_group_oracle(children, x, descriptors)
    independent = tuple(qvq_dense_oracle_forward(child, x) for child in children)

    assert all(output.dtype == torch.float32 for output in grouped)
    for actual, expected in zip(grouped, independent, strict=True):
        assert torch.equal(actual, expected)


def test_grouped_p32_oracle_keeps_output_recovery_child_local():
    su = torch.randn(32, generator=torch.Generator().manual_seed(30))
    child = _child("v_proj", out_features=16, su=su, alt_id=3, seed=31)
    descriptor = QVQExecutionDescriptor(
        "v_proj", "attention.input", input_hadamard=True, output_hadamard=False
    )
    x = torch.randn((4, 32), generator=torch.Generator().manual_seed(32))

    actual = qvq_torch_child_oracle(child, x, descriptor)
    inner = child.get_inner_weight_tensor(dtype=torch.float32)
    expected = matmul_reference(x, child.SU, inner, child.SV, child.bias)
    assert torch.equal(actual, expected)

    grouped = qvq_torch_group_oracle((child,), x, (descriptor,))
    assert torch.equal(grouped[0], expected)


def matmul_reference(
    x: torch.Tensor,
    su: torch.Tensor,
    inner: torch.Tensor,
    sv: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    # Keep the independent formula explicit in the test: input scale/H, inner
    # product, no output H for V, then the child-local SV/bias epilogue.
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU

    transformed = matmul_hadU(x.float() * su.float())
    return transformed @ inner * sv.float() + bias.float()


def test_grouped_p32_validation_fails_closed_for_unsafe_groups():
    shared = torch.ones(32)
    left = _child("left", out_features=32, su=shared, seed=40)
    right = _child("right", out_features=32, su=shared, seed=41)
    descriptors = _descriptors("left", "right")

    with pytest.raises(ValueError, match="bit-identical SU"):
        validate_group((left, _child("different", out_features=32, su=shared + 1, seed=42)), descriptors)
    with pytest.raises(ValueError, match="input_basis_id"):
        validate_group(
            (left, right),
            (
                QVQExecutionDescriptor("left", "attention.input"),
                QVQExecutionDescriptor("right", "other.input"),
            ),
        )
    with pytest.raises(ValueError, match="input_hadamard"):
        validate_group(
            (left, right),
            (
                QVQExecutionDescriptor("left", "attention.input", input_hadamard=True),
                QVQExecutionDescriptor("right", "attention.input", input_hadamard=False),
            ),
        )
    with pytest.raises(ValueError, match="spec does not match"):
        validate_group(
            (left, right), descriptors,
            QVQGroupedP32Spec("attention.input", ("right", "left")),
        )

    unsupported = QVQLinear(
        bits=2,
        in_features=32,
        out_features=32,
        name="unsupported",
        tensors={
            "trellis": left.trellis.clone(),
            "SU": shared.clone(),
            "SV": torch.ones(32),
        },
    )
    with pytest.raises(ValueError, match="only V2B2-P32"):
        validate_group((left, unsupported), descriptors)


def test_grouped_p32_oracle_rejects_grad_inputs_without_touching_children():
    child = _child("proj", out_features=32, seed=50)
    descriptor = QVQExecutionDescriptor("proj", "input")
    x = torch.randn((2, 32), requires_grad=True)
    before = {key: value for key, value in vars(child).items() if key.startswith("_qvq")}
    with pytest.raises(RuntimeError, match="requires gradients"):
        qvq_torch_group_oracle((child,), x, (descriptor,))
    with pytest.raises(RuntimeError, match="requires gradients"):
        qvq_torch_child_oracle(child, x, descriptor)
    after = {key: value for key, value in vars(child).items() if key.startswith("_qvq")}
    assert before == after


def test_grouped_runtime_oracle_requires_order_and_complete_consumption():
    su = torch.randn(32, generator=torch.Generator().manual_seed(60))
    children = (
        _child("q_proj", out_features=32, su=su, alt_id=1, seed=61),
        _child("k_proj", out_features=32, su=su, alt_id=2, seed=62),
        _child("v_proj", out_features=16, su=su, alt_id=3, seed=63),
    )
    descriptors = _descriptors("q_proj", "k_proj", "v_proj")
    runtime = QVQGroupedRuntimeOracle(children, descriptors)
    x = torch.randn((2, 32), generator=torch.Generator().manual_seed(64))

    with pytest.raises(RuntimeError, match="out of order"):
        runtime.consume("k_proj", x)
    assert not runtime.active

    runtime.consume("q_proj", x)
    with pytest.raises(RuntimeError, match="duplicated"):
        runtime.consume("q_proj", x)
    assert not runtime.active

    runtime = QVQGroupedRuntimeOracle(children, descriptors)
    runtime.consume("q_proj", x)
    with pytest.raises(RuntimeError, match="incomplete"):
        runtime.finish_cycle()
    assert not runtime.active

    runtime = QVQGroupedRuntimeOracle(children, descriptors)
    runtime.consume("q_proj", x)
    with pytest.raises(RuntimeError, match="tensor object changed"):
        runtime.consume("k_proj", x.clone())
    assert not runtime.active

    runtime = QVQGroupedRuntimeOracle(children, descriptors)
    runtime.consume("q_proj", x)
    x.add_(1.0)
    with pytest.raises(RuntimeError, match="mutated"):
        runtime.consume("k_proj", x)
    assert not runtime.active

    runtime = QVQGroupedRuntimeOracle(children, descriptors)
    outputs = tuple(runtime.consume(name, x) for name in ("q_proj", "k_proj", "v_proj"))
    runtime.finish_cycle()
    expected = tuple(qvq_dense_oracle_forward(child, x) for child in children)
    for actual, reference in zip(outputs, expected, strict=True):
        # The input mutation is intentionally after the previous failed cycle;
        # this successful cycle uses the same current tensor for all siblings.
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert not runtime.active
