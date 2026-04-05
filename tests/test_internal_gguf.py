import struct

import numpy as np

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
