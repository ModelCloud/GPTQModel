# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq_transform_planner import (
    ModuleTransformDescriptor,
    ProjectionRole,
    QVQTransformPlan,
    TransformKind,
    TransformPlacement,
    TransformSpec,
)
from gptqmodel.quantization.qvq_transform_runtime import (
    QVQGroupedP32Linear,
    QVQSharedInputLinear,
    install_qvq_grouped_p32_input_transforms,
    install_qvq_shared_input_transforms,
)


def _packed_layer(
    *, seed: int, su: torch.Tensor, input_hadamard=True, output_hadamard=True
):
    generator = torch.Generator().manual_seed(seed)
    return QVQLinear(
        bits=2,
        in_features=32,
        out_features=32,
        bank_count=2,
        v2b2_p32=True,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
        tensors={
            "trellis": torch.randint(
                -(2**31),
                2**31 - 1,
                (4, 16),
                generator=generator,
                dtype=torch.int32,
            ),
            "SU": su.clone(),
            "SV": torch.randn(32, generator=generator) * 0.1,
            "bank_ids": torch.zeros(4, dtype=torch.uint8),
            "bank_alt_id": torch.ones(1, dtype=torch.uint8),
        },
        dtype=torch.float32,
    ).eval()


def _shared_plan(*names):
    shared = TransformSpec(
        TransformKind.HADAMARD,
        TransformPlacement.SHARED,
        "test.shared.input",
    )
    online = TransformSpec(TransformKind.HADAMARD, TransformPlacement.ONLINE)
    return QVQTransformPlan(
        arm="TEST",
        description="test shared input",
        modules=tuple(
            ModuleTransformDescriptor(
                module_name=name,
                role=ProjectionRole.ATTENTION_Q,
                input_transform=shared,
                output_transform=online,
            )
            for name in names
        ),
    )


@pytest.mark.parametrize(
    ("input_hadamard", "output_hadamard"),
    ((True, True), (False, True), (True, False), (False, False)),
)
def test_qvq_pretransformed_forward_matches_regular_forward(
    input_hadamard, output_hadamard
):
    generator = torch.Generator().manual_seed(100)
    su = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    layer = _packed_layer(
        seed=101,
        su=su,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    x = torch.randn((2, 3, 32), generator=generator)

    expected = layer(x)
    actual = layer.forward_pretransformed(
        layer.transform_input(x), output_dtype=x.dtype
    )
    transformed = layer.transform_input(x)
    inner = layer._inner_forward(transformed.reshape(-1, layer.in_features))
    recovered = layer.recover_output(
        inner.reshape(*x.shape[:-1], layer.out_features), output_dtype=x.dtype
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(recovered, expected, rtol=0, atol=0)


def test_qvq_shared_input_runtime_reuses_once_and_preserves_outputs():
    generator = torch.Generator().manual_seed(200)
    su = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    root = torch.nn.Module()
    root.first = _packed_layer(seed=201, su=su)
    root.second = _packed_layer(seed=202, su=su)
    baseline_first = copy.deepcopy(root.first)
    baseline_second = copy.deepcopy(root.second)
    x = torch.randn((4, 32), generator=generator)
    expected = (baseline_first(x), baseline_second(x))

    states = install_qvq_shared_input_transforms(root, _shared_plan("first", "second"))
    actual = (root.first(x), root.second(x))

    assert isinstance(root.first, QVQSharedInputLinear)
    assert isinstance(root.second, QVQSharedInputLinear)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    state = states["test.shared.input"]
    assert state.transform_invocations == 1
    assert state.completed_cycles == 1
    assert state.pending_consumers == ("first", "second")

    root.first(x)
    root.second(x)
    assert state.transform_invocations == 2
    assert state.completed_cycles == 2


def test_qvq_shared_input_runtime_rejects_different_stored_su():
    su = torch.ones(32)
    root = torch.nn.Module()
    root.first = _packed_layer(seed=301, su=su)
    root.second = _packed_layer(seed=302, su=-su)

    with pytest.raises(ValueError, match="bit-identical stored SU"):
        install_qvq_shared_input_transforms(root, _shared_plan("first", "second"))


def test_qvq_shared_input_runtime_rejects_incomplete_or_duplicate_cycles():
    su = torch.ones(32)
    root = torch.nn.Module()
    root.first = _packed_layer(seed=401, su=su)
    root.second = _packed_layer(seed=402, su=su)
    install_qvq_shared_input_transforms(root, _shared_plan("first", "second"))
    x = torch.randn((1, 32), generator=torch.Generator().manual_seed(403))

    root.first(x)
    with pytest.raises(RuntimeError, match="duplicate consumer"):
        root.first(x)

    state = root.first._shared_input_state
    state.reset()
    root.first(x)
    with pytest.raises(RuntimeError, match="new or mutated activation"):
        root.second(x.clone())


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires NVIDIA CUDA compute capability >= 8.0",
)
def test_qvq_shared_input_runtime_uses_packed_cuda_p32_path():
    generator = torch.Generator().manual_seed(500)
    su = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    root = torch.nn.Module()
    root.first = _packed_layer(seed=501, su=su).half().cuda()
    root.second = _packed_layer(seed=502, su=su).half().cuda()
    with torch.inference_mode():
        x = torch.randn((8, 32), generator=generator, dtype=torch.float16).cuda()
        expected = (root.first(x), root.second(x))
        states = install_qvq_shared_input_transforms(
            root, _shared_plan("first", "second")
        )
        actual = (root.first(x), root.second(x))

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert states["test.shared.input"].transform_invocations == 1


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires NVIDIA CUDA compute capability >= 8.0",
)
def test_qvq_grouped_p32_runtime_runs_one_decode_and_releases_child_payloads():
    generator = torch.Generator().manual_seed(600)
    su = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    root = torch.nn.Module()
    root.first = _packed_layer(seed=601, su=su).half().cuda()
    root.second = _packed_layer(seed=602, su=su).half().cuda()
    root.first.bank_ids.fill_(0xAA)
    root.second.bank_ids.fill_(0x55)
    root.first.bank_alt_id.fill_(1)
    root.second.bank_alt_id.fill_(3)
    baseline_first = copy.deepcopy(root.first)
    baseline_second = copy.deepcopy(root.second)

    with torch.inference_mode():
        x = torch.randn((8, 32), generator=generator, dtype=torch.float16).cuda()
        expected = (baseline_first(x), baseline_second(x))
        states = install_qvq_grouped_p32_input_transforms(
            root, _shared_plan("first", "second")
        )
        actual = (root.first(x), root.second(x))

    assert isinstance(root.first, QVQGroupedP32Linear)
    assert isinstance(root.second, QVQGroupedP32Linear)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=2e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=2e-3)
    state = states["test.shared.input"]
    assert state.transform_invocations == 1
    assert state.grouped_gemv_invocations == 1
    assert state.completed_cycles == 1
    assert state.metadata_overhead_bytes == 2
    assert root.first.linear.trellis.numel() == 0
    assert root.second.linear.trellis.numel() == 0
    with pytest.raises(RuntimeError, match="not checkpoint-serializable"):
        root.state_dict()
