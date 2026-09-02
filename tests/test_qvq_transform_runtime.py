# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
import io
import json
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.qvq_transform_planner import (
    ModuleTransformDescriptor,
    ProjectionRole,
    QVQTransformPlan,
    TransformKind,
    TransformPlacement,
    TransformSpec,
)
from gptqmodel.quantization.qvq_transform_runtime import (
    QVQ_GROUPED_P32_PAYLOAD_LAYOUT,
    QVQ_GROUPED_P32_RUNTIME_META_KEY,
    QVQ_GROUPED_P32_RUNTIME_SCHEMA,
    QVQGroupedP32Linear,
    QVQSharedInputLinear,
    install_qvq_checkpointed_p32_runtime,
    install_qvq_grouped_p32_input_transforms,
    install_qvq_grouped_p32_runtime_from_config,
    install_qvq_refactored_p32_runtime,
    install_qvq_shared_input_transforms,
    qvq_grouped_p32_checkpoint_metadata,
    qvq_grouped_p32_checkpoint_metadata_bytes,
    set_qvq_grouped_p32_checkpoint_metadata,
)
from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv


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


def test_qvq_grouped_p32_checkpoint_metadata_is_versioned_and_storage_counted():
    plan = _shared_plan("first", "second")
    metadata = qvq_grouped_p32_checkpoint_metadata(plan)
    expected = {
        "schema": QVQ_GROUPED_P32_RUNTIME_SCHEMA,
        "payload_layout": QVQ_GROUPED_P32_PAYLOAD_LAYOUT,
        "groups": [
            {
                "basis_id": "test.shared.input",
                "module_names": ["first", "second"],
            }
        ],
    }
    assert metadata == expected
    assert qvq_grouped_p32_checkpoint_metadata_bytes(metadata) == len(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    config = SimpleNamespace(meta={"existing": "preserved"})
    returned = set_qvq_grouped_p32_checkpoint_metadata(config, plan)
    assert returned == expected
    assert config.meta == {
        "existing": "preserved",
        QVQ_GROUPED_P32_RUNTIME_META_KEY: expected,
    }
    returned["groups"][0]["module_names"].append("mutation")
    assert config.meta[QVQ_GROUPED_P32_RUNTIME_META_KEY] == expected

    qvq_config = QuantizeConfig(
        method="qvq",
        format="qvq_v2b2_p32",
        bits=2,
        offload_to_disk=False,
    )
    set_qvq_grouped_p32_checkpoint_metadata(qvq_config, plan)
    reloaded = QuantizeConfig.from_quant_config(qvq_config.to_dict())
    assert reloaded.meta[QVQ_GROUPED_P32_RUNTIME_META_KEY] == expected


def test_qvq_grouped_p32_checkpoint_absence_preserves_legacy_runtime():
    root = torch.nn.Module()
    root.first = _packed_layer(seed=10, su=torch.ones(32))

    assert (
        install_qvq_grouped_p32_runtime_from_config(
            root, SimpleNamespace(meta={"unrelated": True})
        )
        is None
    )
    assert isinstance(root.first, QVQLinear)
    assert not hasattr(root, "_qvq_grouped_p32_runtime_states")


@pytest.mark.parametrize(
    "mutate",
    (
        lambda metadata: metadata.update(schema="future"),
        lambda metadata: metadata.update(payload_layout="grouped_only"),
        lambda metadata: metadata.update(extra=True),
        lambda metadata: metadata["groups"].append(copy.deepcopy(metadata["groups"][0])),
        lambda metadata: metadata["groups"][0].update(module_names=["first"]),
        lambda metadata: metadata["groups"][0].update(module_names=["first", "first"]),
    ),
)
def test_qvq_grouped_p32_checkpoint_metadata_fails_closed(mutate):
    metadata = qvq_grouped_p32_checkpoint_metadata(
        _shared_plan("first", "second")
    )
    mutate(metadata)

    with pytest.raises((TypeError, ValueError)):
        qvq_grouped_p32_checkpoint_metadata_bytes(metadata)


def test_qvq_grouped_p32_checkpoint_rejects_model_manifest_mismatch():
    root = torch.nn.Module()
    root.first = _packed_layer(seed=20, su=torch.ones(32))
    metadata = qvq_grouped_p32_checkpoint_metadata(
        _shared_plan("first", "missing")
    )

    with pytest.raises(ValueError, match="missing module 'missing'"):
        install_qvq_checkpointed_p32_runtime(root, metadata)


def test_qvq_grouped_p32_checkpoint_cpu_fallback_keeps_canonical_payloads():
    su = torch.ones(32)
    root = torch.nn.Module()
    root.first = _packed_layer(seed=30, su=su)
    root.second = _packed_layer(seed=31, su=su)
    expected_state = copy.deepcopy(root.state_dict())
    config = SimpleNamespace(meta=None, format="qvq_v2b2_p32")
    metadata = set_qvq_grouped_p32_checkpoint_metadata(
        config, _shared_plan("first", "second")
    )

    compiled = install_qvq_grouped_p32_runtime_from_config(root, config)

    assert compiled is not None
    assert compiled.grouped_states == {}
    assert set(compiled.plain_fallbacks) == {"test.shared.input"}
    assert "requires CUDA-resident payloads" in compiled.plain_fallbacks[
        "test.shared.input"
    ]
    assert compiled.checkpoint_metadata_bytes == qvq_grouped_p32_checkpoint_metadata_bytes(
        metadata
    )
    assert isinstance(root.first, QVQLinear)
    assert isinstance(root.second, QVQLinear)
    assert root.state_dict().keys() == expected_state.keys()
    for key, expected in expected_state.items():
        torch.testing.assert_close(root.state_dict()[key], expected, rtol=0, atol=0)


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


def test_qvq_refactored_runtime_preserves_module_local_su_with_plain_fallback():
    generator = torch.Generator().manual_seed(350)
    su = torch.ones(32)
    root = torch.nn.Module()
    root.first = _packed_layer(seed=351, su=su)
    root.second = _packed_layer(seed=352, su=-su)
    baseline_first = copy.deepcopy(root.first)
    baseline_second = copy.deepcopy(root.second)
    x = torch.randn((4, 32), generator=generator)
    expected = (baseline_first(x), baseline_second(x))

    compiled = install_qvq_refactored_p32_runtime(
        root, _shared_plan("first", "second")
    )
    actual = (root.first(x), root.second(x))

    assert compiled.grouped_states == {}
    assert set(compiled.plain_fallbacks) == {"test.shared.input"}
    assert "bit-identical stored SU" in compiled.plain_fallbacks["test.shared.input"]
    assert isinstance(root.first, QVQLinear)
    assert isinstance(root.second, QVQLinear)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


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
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    state = states["test.shared.input"]
    assert state.transform_invocations == 1
    assert state.grouped_gemv_invocations == 1
    assert state.completed_cycles == 1
    assert state.bank_alt_ids.tolist() == [1, 3]
    assert state.bank_alt_boundaries == (2,)
    assert state.metadata_overhead_bytes == 0
    assert root.first.linear.trellis.numel() == 0
    assert root.second.linear.trellis.numel() == 0
    with pytest.raises(RuntimeError, match="not checkpoint-serializable"):
        root.state_dict()


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires NVIDIA CUDA compute capability >= 8.0",
)
def test_qvq_refactored_runtime_groups_compatible_payloads():
    generator = torch.Generator().manual_seed(650)
    su = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    root = torch.nn.Module()
    root.first = _packed_layer(seed=651, su=su).half().cuda()
    root.second = _packed_layer(seed=652, su=su).half().cuda()
    baseline_first = copy.deepcopy(root.first)
    baseline_second = copy.deepcopy(root.second)

    with torch.inference_mode():
        x = torch.randn((8, 32), generator=generator, dtype=torch.float16).cuda()
        expected = (baseline_first(x), baseline_second(x))
        compiled = install_qvq_refactored_p32_runtime(
            root, _shared_plan("first", "second")
        )
        actual = (root.first(x), root.second(x))

    assert compiled.plain_fallbacks == {}
    assert set(compiled.grouped_states) == {"test.shared.input"}
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires NVIDIA CUDA compute capability >= 8.0",
)
def test_qvq_checkpointed_p32_runtime_roundtrips_canonical_payload_exactly():
    generator = torch.Generator().manual_seed(675)
    su = torch.randint(0, 2, (32,), generator=generator).mul_(2).sub_(1).float()
    root = torch.nn.Module()
    root.first = _packed_layer(seed=676, su=su).half().cuda()
    root.second = _packed_layer(seed=677, su=su).half().cuda()
    root.first.bank_ids.fill_(0xAA)
    root.second.bank_ids.fill_(0x55)
    root.first.bank_alt_id.fill_(1)
    root.second.bank_alt_id.fill_(3)
    # Model the accepted fixed-trellis correction representation: local SV is
    # allowed to differ while trellis, selectors, and SU remain frozen.
    root.first.SV.mul_(0.9375)
    root.second.SV.mul_(1.0625)
    canonical_state = copy.deepcopy(root.state_dict())
    plan = _shared_plan("first", "second")
    config = SimpleNamespace(meta=None, format="qvq_v2b2_p32")
    metadata = set_qvq_grouped_p32_checkpoint_metadata(config, plan)

    with torch.inference_mode():
        x = torch.randn((8, 32), generator=generator, dtype=torch.float16).cuda()
        expected = (root.first(x), root.second(x))
        compiled = install_qvq_grouped_p32_runtime_from_config(root, config)
        actual = (root.first(x), root.second(x))

    assert compiled is not None
    assert compiled.plain_fallbacks == {}
    assert set(compiled.grouped_states) == {"test.shared.input"}
    assert isinstance(root.first, QVQLinear)
    assert isinstance(root.second, QVQLinear)
    assert root.first.trellis.numel() == 0
    assert root.second.trellis.numel() == 0
    state = compiled.grouped_states["test.shared.input"]
    assert state.grouped_gemv_invocations == 1
    assert state.metadata_overhead_bytes == 0
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    saved_state = root.state_dict()
    assert saved_state.keys() == canonical_state.keys()
    assert not any("_qvq_grouped_p32" in key for key in saved_state)
    for key, expected_tensor in canonical_state.items():
        torch.testing.assert_close(
            saved_state[key], expected_tensor, rtol=0, atol=0
        )

    checkpoint = io.BytesIO()
    torch.save(saved_state, checkpoint)
    checkpoint.seek(0)
    loaded_state = torch.load(checkpoint, weights_only=True)
    reloaded = torch.nn.Module()
    reloaded.first = _packed_layer(seed=678, su=su).half().cuda()
    reloaded.second = _packed_layer(seed=679, su=su).half().cuda()
    reloaded.load_state_dict(loaded_state, strict=True)
    reloaded.eval()
    reloaded_runtime = install_qvq_checkpointed_p32_runtime(
        reloaded, copy.deepcopy(metadata)
    )
    with torch.inference_mode():
        reloaded_actual = (reloaded.first(x), reloaded.second(x))

    assert reloaded_runtime.plain_fallbacks == {}
    assert reloaded_runtime.checkpoint_metadata_bytes == (
        qvq_grouped_p32_checkpoint_metadata_bytes(metadata)
    )
    torch.testing.assert_close(reloaded_actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(reloaded_actual[1], expected[1], rtol=0, atol=0)
    reserialized = reloaded.state_dict()
    assert reserialized.keys() == canonical_state.keys()
    for key, expected_tensor in canonical_state.items():
        torch.testing.assert_close(
            reserialized[key], expected_tensor, rtol=0, atol=0
        )


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires NVIDIA CUDA compute capability >= 8.0",
)
def test_qvq_grouped_p32_preserves_child_split_k_arithmetic():
    """Grouped Llama QKV must retain each plain child's FP32 reduction order."""

    generator = torch.Generator(device="cuda").manual_seed(700)
    k = 2048
    widths = (2048, 512, 512)
    alternatives = (1, 2, 3)
    k_tiles = k // 16
    trellis_parts = []
    selector_parts = []
    child_payloads = []
    for width in widths:
        n_tiles = width // 16
        tile_count = k_tiles * n_tiles
        trellis = torch.randint(
            -(2**31),
            2**31 - 1,
            (tile_count, 16),
            generator=generator,
            device="cuda",
            dtype=torch.int32,
        )
        selectors = torch.randint(
            0,
            256,
            (tile_count,),
            generator=generator,
            device="cuda",
            dtype=torch.uint8,
        )
        child_payloads.append((trellis, selectors))
        trellis_parts.append(trellis.view(k_tiles, n_tiles, 16))
        selector_parts.append(selectors.view(k_tiles, n_tiles))

    grouped_trellis = torch.cat(trellis_parts, dim=1).reshape(-1, 16)
    grouped_selectors = torch.cat(selector_parts, dim=1).reshape(-1)
    grouped_alternatives = torch.tensor(
        alternatives, device="cuda", dtype=torch.uint8
    )

    for dtype in (torch.float16, torch.bfloat16):
        for rows in (1, 2, 4, 8):
            x = torch.randn(
                (rows, k), generator=generator, device="cuda", dtype=dtype
            )
            expected = tuple(
                qvq_cuda_gemv(
                    x,
                    trellis,
                    2,
                    out_features=width,
                    output_fp32=True,
                    bank_ids=selectors,
                    v2b2_p32=True,
                    bank_alt_id=alternative,
                )
                for width, alternative, (trellis, selectors) in zip(
                    widths, alternatives, child_payloads, strict=True
                )
            )
            grouped = qvq_cuda_gemv(
                x,
                grouped_trellis,
                2,
                out_features=sum(widths),
                output_fp32=True,
                bank_ids=grouped_selectors,
                v2b2_p32=True,
                bank_alt_ids=grouped_alternatives,
                bank_alt_boundaries=(128, 160),
                _bank_alt_ids_validated=True,
            )
            for child, segment in zip(
                expected, grouped.split(widths, dim=-1), strict=True
            ):
                torch.testing.assert_close(segment, child, rtol=0, atol=0)
