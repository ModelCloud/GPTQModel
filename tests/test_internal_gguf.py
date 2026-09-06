import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf as gguf_qlinear
from gptqmodel.utils import internal_gguf


def test_internal_gguf_types_match_current_upstream_assignments():
    assert int(internal_gguf.GGMLQuantizationType.NVFP4) == 40
    assert int(internal_gguf.GGMLQuantizationType.Q1_0) == 41
    assert int(internal_gguf.GGMLQuantizationType.Q2_0) == 42
    assert internal_gguf.GGMLQuantizationType.Q1_0_g128 is internal_gguf.GGMLQuantizationType.Q1_0

    assert internal_gguf.GGML_QUANT_SIZES[internal_gguf.GGMLQuantizationType.NVFP4] == (64, 36)
    assert internal_gguf.GGML_QUANT_SIZES[internal_gguf.GGMLQuantizationType.Q1_0] == (128, 18)
    assert internal_gguf.GGML_QUANT_SIZES[internal_gguf.GGMLQuantizationType.Q2_0] == (64, 18)


def _encode_gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_minimal_gguf_tensor(tmp_path, *, tensor_type, shape, data):
    payload = bytearray()
    payload.extend(struct.pack("<I", internal_gguf.GGUF_MAGIC))
    payload.extend(struct.pack("<I", internal_gguf.GGUF_VERSION))
    payload.extend(struct.pack("<Q", 1))
    payload.extend(struct.pack("<Q", 0))
    payload.extend(_encode_gguf_string("weight"))
    payload.extend(struct.pack("<I", len(shape)))
    for dimension in shape:
        payload.extend(struct.pack("<Q", dimension))
    payload.extend(struct.pack("<I", int(tensor_type)))
    payload.extend(struct.pack("<Q", 0))
    payload.extend(b"\x00" * ((-len(payload)) % internal_gguf.GGUF_DEFAULT_ALIGNMENT))
    payload.extend(data)

    path = tmp_path / f"minimal-{tensor_type.name.lower()}.gguf"
    path.write_bytes(payload)
    return path


def test_internal_gguf_dequantizes_prism_q1_0_g128_blocks():
    scale = np.array([1.5], dtype=np.float16).view(np.uint8)
    sign_bits = (np.arange(128, dtype=np.uint8) % 3 == 0).astype(np.uint8)
    packed_bits = np.packbits(sign_bits, bitorder="little")
    row = np.concatenate([scale, packed_bits], axis=0).reshape(1, -1)

    actual = internal_gguf.dequantize(row, internal_gguf.GGMLQuantizationType.Q1_0_g128)
    expected = np.where(sign_bits == 1, np.float32(1.5), np.float32(-1.5)).reshape(1, 128)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_internal_gguf_quantizes_official_q1_0_blocks():
    values = np.linspace(-2.0, 2.0, 128, dtype=np.float32).reshape(1, -1)

    packed = internal_gguf.quantize(values, internal_gguf.GGMLQuantizationType.Q1_0)
    actual = internal_gguf.dequantize(packed, internal_gguf.GGMLQuantizationType.Q1_0)

    expected_scale = np.mean(np.abs(values)).astype(np.float16).astype(np.float32)
    expected = np.where(values >= 0, expected_scale, -expected_scale)
    assert packed.shape == (1, 18)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_internal_gguf_quantizes_and_dequantizes_official_q2_0_blocks():
    values = np.tile(np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32), 16).reshape(1, -1)
    expected_packed = np.concatenate(
        [
            np.array([1.0], dtype=np.float16).view(np.uint8),
            np.full(16, 0x64, dtype=np.uint8),
        ]
    ).reshape(1, -1)

    actual_packed = internal_gguf.quantize(values, internal_gguf.GGMLQuantizationType.Q2_0)
    actual_values = internal_gguf.dequantize(expected_packed, internal_gguf.GGMLQuantizationType.Q2_0)

    np.testing.assert_array_equal(actual_packed, expected_packed)
    np.testing.assert_allclose(actual_values, values, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("tensor_qtype", "block_size"),
    [
        (internal_gguf.GGMLQuantizationType.Q2_0, 64),
        (internal_gguf.GGMLQuantizationType.TQ1_0, 256),
        (internal_gguf.GGMLQuantizationType.TQ2_0, 256),
    ],
)
def test_internal_gguf_fallback_rounds_float32_half_thresholds(monkeypatch, tensor_qtype, block_size):
    monkeypatch.setattr(gguf_qlinear, "_GGUF_AVAILABLE", False)
    positive_below = np.nextafter(np.float32(0.5), np.float32(0.0))
    positive_above = np.nextafter(np.float32(0.5), np.float32(1.0))
    negative_below = np.nextafter(np.float32(-0.5), np.float32(-1.0))
    negative_above = np.nextafter(np.float32(-0.5), np.float32(0.0))

    values = np.zeros((1, block_size), dtype=np.float32)
    values[0, :7] = [
        1.0,
        positive_below,
        0.5,
        positive_above,
        negative_below,
        -0.5,
        negative_above,
    ]
    expected = np.zeros_like(values)
    expected[0, :7] = [1.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0]

    packed = internal_gguf.quantize(values, tensor_qtype)
    actual = internal_gguf.dequantize(packed, tensor_qtype)

    np.testing.assert_array_equal(actual, expected)


def test_internal_gguf_dequantizes_nvfp4_blocks():
    scales = np.full(4, 64, dtype=np.uint8)
    packed_values = np.full(32, 0x91, dtype=np.uint8)
    packed = np.concatenate([scales, packed_values]).reshape(1, -1)

    actual = internal_gguf.dequantize(packed, internal_gguf.GGMLQuantizationType.NVFP4)
    expected_group = np.concatenate([np.ones(8, dtype=np.float32), -np.ones(8, dtype=np.float32)])
    expected = np.tile(expected_group, 4).reshape(1, -1)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("tensor_qtype", "type_size", "block_size"),
    [
        (internal_gguf.GGMLQuantizationType.Q2_0, 18, 64),
        (internal_gguf.GGMLQuantizationType.TQ1_0, 54, 256),
        (internal_gguf.GGMLQuantizationType.TQ2_0, 66, 256),
        (internal_gguf.GGMLQuantizationType.MXFP4, 17, 32),
        (internal_gguf.GGMLQuantizationType.NVFP4, 36, 64),
    ],
)
def test_internal_gguf_new_dequantizers_preserve_leading_dimensions(tensor_qtype, type_size, block_size):
    one_dimensional = internal_gguf.dequantize(np.zeros(type_size, dtype=np.uint8), tensor_qtype)
    stacked = internal_gguf.dequantize(np.zeros((2, 3, type_size * 2), dtype=np.uint8), tensor_qtype)

    assert one_dimensional.shape == (block_size,)
    assert stacked.shape == (2, 3, block_size * 2)

    with pytest.raises(ValueError, match="row byte width"):
        internal_gguf.dequantize(np.zeros(type_size + 1, dtype=np.uint8), tensor_qtype)


def test_internal_gguf_dequantize_uses_torch_sign_only_path_when_requested(monkeypatch):
    scale = np.array([0.75], dtype=np.float16).view(np.uint8)
    sign_bits = (np.arange(128, dtype=np.uint8) % 5 < 2).astype(np.uint8)
    packed_bits = np.packbits(sign_bits, bitorder="little")
    row = np.concatenate([scale, packed_bits], axis=0).reshape(1, -1)
    expected = np.where(sign_bits == 1, np.float32(0.75), np.float32(-0.75)).reshape(1, 128)

    calls = {"count": 0}
    original = internal_gguf._dequantize_sign_only_torch_to_numpy

    def _wrapped(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setenv("GPTQMODEL_INTERNAL_GGUF_DEQUANT_DEVICE", "cpu")
    monkeypatch.setattr(internal_gguf, "_dequantize_sign_only_torch_to_numpy", _wrapped)

    actual = internal_gguf.dequantize(row, internal_gguf.GGMLQuantizationType.Q1_0_g128)

    assert calls["count"] == 1
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_internal_gguf_dequantize_to_torch_returns_exact_prism_tensor():
    scale = np.array([0.5], dtype=np.float16).view(np.uint8)
    sign_bits = (np.arange(128, dtype=np.uint8) % 7 < 3).astype(np.uint8)
    packed_bits = np.packbits(sign_bits, bitorder="little")
    row = np.concatenate([scale, packed_bits], axis=0).reshape(1, -1)
    expected = np.where(sign_bits == 1, np.float32(0.5), np.float32(-0.5)).reshape(1, 128)

    actual = internal_gguf.dequantize_to_torch(
        row,
        internal_gguf.GGMLQuantizationType.Q1_0_g128,
        device="cpu",
        dtype=torch.float32,
    )

    assert isinstance(actual, torch.Tensor)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0.0, atol=0.0)


def test_internal_gguf_reader_reads_minimal_f32_tensor(tmp_path):
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = _write_minimal_gguf_tensor(
        tmp_path,
        tensor_type=internal_gguf.GGMLQuantizationType.F32,
        shape=(2, 2),
        data=tensor.tobytes(),
    )

    reader = internal_gguf.GGUFReader(path)

    assert len(reader.tensors) == 1
    assert reader.tensors[0].name == "weight"
    assert reader.tensors[0].tensor_type == internal_gguf.GGMLQuantizationType.F32
    np.testing.assert_allclose(reader.tensors[0].data, tensor, rtol=0.0, atol=0.0)


def test_internal_gguf_reader_uses_current_nvfp4_storage_size(tmp_path):
    packed = np.arange(36, dtype=np.uint8)
    path = _write_minimal_gguf_tensor(
        tmp_path,
        tensor_type=internal_gguf.GGMLQuantizationType.NVFP4,
        shape=(64,),
        data=packed.tobytes(),
    )

    tensor = internal_gguf.GGUFReader(path).tensors[0]

    assert tensor.tensor_type == internal_gguf.GGMLQuantizationType.NVFP4
    assert tensor.n_elements == 64
    assert tensor.n_bytes == 36
    np.testing.assert_array_equal(tensor.data, packed)
    assert internal_gguf.dequantize(tensor.data, tensor.tensor_type).shape == (64,)


@pytest.mark.parametrize(
    ("tensor_qtype", "bits_alias"),
    [
        (internal_gguf.GGMLQuantizationType.Q1_0, "q1_0"),
        (internal_gguf.GGMLQuantizationType.Q2_0, "q2_0"),
    ],
)
def test_internal_gguf_inspect_quantized_checkpoint_detects_qwen3_spec(tensor_qtype, bits_alias):
    class _Field:
        def __init__(self, value):
            self._value = value

        def contents(self, index_or_slice=slice(None)):
            del index_or_slice
            return self._value

    reader = SimpleNamespace(
        tensors=[
            SimpleNamespace(
                name="blk.0.attn_q.weight",
                tensor_type=tensor_qtype,
            ),
            SimpleNamespace(
                name="blk.0.ffn_gate.weight",
                tensor_type=tensor_qtype,
            ),
            SimpleNamespace(
                name="output.weight",
                tensor_type=tensor_qtype,
            ),
        ],
        get_field=lambda key: _Field("qwen3") if key == "general.architecture" else None,
    )

    spec = internal_gguf.inspect_quantized_checkpoint(reader)

    assert spec is not None
    assert spec.model_type == internal_gguf.MODEL_ARCH_QWEN3
    assert spec.bits_alias == bits_alias
    assert spec.tensor_qtype == tensor_qtype
    assert spec.lm_head_quantized is True
