import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gptqmodel.utils import internal_gguf


def _encode_gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def test_internal_gguf_dequantizes_prism_q1_0_g128_blocks():
    scale = np.array([1.5], dtype=np.float16).view(np.uint8)
    sign_bits = (np.arange(128, dtype=np.uint8) % 3 == 0).astype(np.uint8)
    packed_bits = np.packbits(sign_bits, bitorder="little")
    row = np.concatenate([scale, packed_bits], axis=0).reshape(1, -1)

    actual = internal_gguf.dequantize(row, internal_gguf.GGMLQuantizationType.Q1_0_g128)
    expected = np.where(sign_bits == 1, np.float32(1.5), np.float32(-1.5)).reshape(1, 128)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_internal_gguf_dequantizes_prism_q2_0_and_pq2_0_blocks():
    scale = np.array([1.5], dtype=np.float16).view(np.uint8)
    codes = (np.arange(128, dtype=np.uint8) % 3).reshape(-1, 4)
    packed_codes = (
        codes[:, 0]
        | (codes[:, 1] << np.uint8(2))
        | (codes[:, 2] << np.uint8(4))
        | (codes[:, 3] << np.uint8(6))
    )
    row = np.concatenate([scale, packed_codes], axis=0).reshape(1, -1)
    expected = ((codes.reshape(-1).astype(np.int16) - 1) * np.float32(1.5)).reshape(1, 128)

    for qtype in (
        internal_gguf.GGMLQuantizationType.Q2_0,
        internal_gguf.GGMLQuantizationType.PQ2_0,
    ):
        actual = internal_gguf.dequantize(row, qtype)
        torch_actual = internal_gguf.dequantize_to_torch(row, qtype, device="cpu", dtype=torch.float32)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(torch_actual.numpy(), expected, rtol=0.0, atol=0.0)


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

    payload = bytearray()
    payload.extend(struct.pack("<I", internal_gguf.GGUF_MAGIC))
    payload.extend(struct.pack("<I", internal_gguf.GGUF_VERSION))
    payload.extend(struct.pack("<Q", 1))  # tensor_count
    payload.extend(struct.pack("<Q", 0))  # kv_count
    payload.extend(_encode_gguf_string("weight"))
    payload.extend(struct.pack("<I", 2))  # n_dims
    payload.extend(struct.pack("<Q", 2))
    payload.extend(struct.pack("<Q", 2))
    payload.extend(struct.pack("<I", int(internal_gguf.GGMLQuantizationType.F32)))
    payload.extend(struct.pack("<Q", 0))

    padding = (-len(payload)) % internal_gguf.GGUF_DEFAULT_ALIGNMENT
    payload.extend(b"\x00" * padding)
    payload.extend(tensor.tobytes())

    path = tmp_path / "minimal.gguf"
    path.write_bytes(payload)

    reader = internal_gguf.GGUFReader(path)

    assert len(reader.tensors) == 1
    assert reader.tensors[0].name == "weight"
    assert reader.tensors[0].tensor_type == internal_gguf.GGMLQuantizationType.F32
    np.testing.assert_allclose(reader.tensors[0].data, tensor, rtol=0.0, atol=0.0)


def test_internal_gguf_reader_rejects_mislabeled_prism_q2_0_g64_storage(tmp_path):
    def write_q2_file(name: str, data_bytes: int):
        payload = bytearray()
        payload.extend(struct.pack("<I", internal_gguf.GGUF_MAGIC))
        payload.extend(struct.pack("<I", internal_gguf.GGUF_VERSION))
        payload.extend(struct.pack("<Q", 1))
        payload.extend(struct.pack("<Q", 0))
        payload.extend(_encode_gguf_string("weight"))
        payload.extend(struct.pack("<I", 1))
        payload.extend(struct.pack("<Q", 128))
        payload.extend(struct.pack("<I", int(internal_gguf.GGMLQuantizationType.Q2_0)))
        payload.extend(struct.pack("<Q", 0))
        payload.extend(b"\x00" * ((-len(payload)) % internal_gguf.GGUF_DEFAULT_ALIGNMENT))
        payload.extend(b"\x00" * data_bytes)
        path = tmp_path / name
        path.write_bytes(payload)
        return path

    canonical = internal_gguf.GGUFReader(write_q2_file("canonical-q2.gguf", data_bytes=34))
    assert canonical.tensors[0].n_bytes == 34

    with pytest.raises(ValueError, match="incompatible Q2_0_g64 layout"):
        internal_gguf.GGUFReader(write_q2_file("mislabeled-q2-g64.gguf", data_bytes=36))


def test_internal_gguf_inspect_quantized_checkpoint_detects_qwen3_prism_spec():
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
                tensor_type=internal_gguf.GGMLQuantizationType.Q1_0_g128,
            ),
            SimpleNamespace(
                name="blk.0.ffn_gate.weight",
                tensor_type=internal_gguf.GGMLQuantizationType.Q1_0_g128,
            ),
            SimpleNamespace(
                name="output.weight",
                tensor_type=internal_gguf.GGMLQuantizationType.Q1_0_g128,
            ),
        ],
        get_field=lambda key: _Field("qwen3") if key == "general.architecture" else None,
    )

    spec = internal_gguf.inspect_quantized_checkpoint(reader)

    assert spec is not None
    assert spec.model_type == internal_gguf.MODEL_ARCH_QWEN3
    assert spec.bits_alias == "q1_0_g128"
    assert spec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0_g128
    assert spec.lm_head_quantized is True


def test_internal_gguf_inspect_quantized_checkpoint_detects_qwen3_prism_ternary_spec():
    class _Field:
        def contents(self, index_or_slice=slice(None)):
            del index_or_slice
            return "qwen3"

    reader = SimpleNamespace(
        tensors=[
            SimpleNamespace(
                name="blk.0.attn_q.weight",
                tensor_type=internal_gguf.GGMLQuantizationType.PQ2_0,
            ),
            SimpleNamespace(
                name="blk.0.ffn_gate.weight",
                tensor_type=internal_gguf.GGMLQuantizationType.PQ2_0,
            ),
        ],
        get_field=lambda key: _Field() if key == "general.architecture" else None,
    )

    spec = internal_gguf.inspect_quantized_checkpoint(reader)

    assert spec is not None
    assert spec.bits_alias == "q2_0"
    assert spec.tensor_qtype == internal_gguf.GGMLQuantizationType.PQ2_0
    assert spec.lm_head_quantized is False
