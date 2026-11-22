# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GGUF format: ggml-org/llama.cpp, MIT, https://github.com/ggml-org/llama.cpp

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from gptqmodel.nn_modules.qlinear import gguf as gguf_qlinear
from gptqmodel.utils import internal_gguf
from gptqmodel.utils.ggml import GGML_1_0, ggml_version_at_least, is_ggml_1_0_or_newer, parse_ggml_version


def test_internal_gguf_matches_ggml_1_0_type_table():
    # ggml 1.0 keeps the pre-1.0 assignments and exposes the OCP FP8 E4M3
    # scalar type at id 43 (ggml/include/ggml.h).
    assert int(internal_gguf.GGMLQuantizationType.Q2_0) == 42
    assert int(internal_gguf.GGMLQuantizationType.F8_E4M3) == 43
    assert internal_gguf.GGML_QUANT_SIZES[internal_gguf.GGMLQuantizationType.F8_E4M3] == (1, 1)


def test_gguf_resolves_ggml_1_0_f8_e4m3_tensor_type():
    assert gguf_qlinear._GGUF_TENSOR_QTYPE_BY_VALUE[43] == "F8_E4M3"
    assert gguf_qlinear._resolve_gguf_tensor_qtype(43) == "F8_E4M3"
    assert gguf_qlinear._resolve_gguf_tensor_qtype("F8_E4M3") == "F8_E4M3"


def test_internal_gguf_dequantizes_f8_e4m3_like_torch():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable")

    # 0.0, 1.0, 1.5, 2.0, -1.0, -2.5, 1/512 (subnormal)
    payload = np.array([[0x00, 0x38, 0x3C, 0x40, 0xB8, 0xC4, 0x01]], dtype=np.uint8)
    expected = torch.from_numpy(payload).view(torch.float8_e4m3fn).to(torch.float32).numpy()

    actual = internal_gguf.dequantize(payload, internal_gguf.GGMLQuantizationType.F8_E4M3)

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_gguf_qlinear_dequantizes_f8_e4m3_like_torch():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable")

    payload = np.array([[0x00, 0x38, 0x3C, 0x40, 0xB8, 0xC4]], dtype=np.uint8)
    expected = torch.from_numpy(payload).view(torch.float8_e4m3fn).to(torch.float32).numpy()

    actual = gguf_qlinear._dequantize_gguf_tensor_numpy(payload, "F8_E4M3")

    np.testing.assert_array_equal(actual, expected)


def _encode_gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_f8_e4m3_gguf(tmp_path, payload: np.ndarray) -> Path:
    rows, columns = payload.shape
    encoded = bytearray()
    encoded.extend(struct.pack("<I", internal_gguf.GGUF_MAGIC))
    encoded.extend(struct.pack("<I", internal_gguf.GGUF_VERSION))
    encoded.extend(struct.pack("<Q", 1))
    encoded.extend(struct.pack("<Q", 0))
    encoded.extend(_encode_gguf_string("weight"))
    encoded.extend(struct.pack("<I", 2))
    encoded.extend(struct.pack("<Q", columns))
    encoded.extend(struct.pack("<Q", rows))
    encoded.extend(struct.pack("<I", int(internal_gguf.GGMLQuantizationType.F8_E4M3)))
    encoded.extend(struct.pack("<Q", 0))
    encoded.extend(b"\x00" * ((-len(encoded)) % internal_gguf.GGUF_DEFAULT_ALIGNMENT))
    encoded.extend(payload.tobytes())

    path = tmp_path / "f8-e4m3.gguf"
    path.write_bytes(encoded)
    return path


def test_internal_gguf_reader_reads_f8_e4m3_tensors(tmp_path):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable")

    payload = np.array([[0x00, 0x38, 0x3C], [0x40, 0xB8, 0xC4]], dtype=np.uint8)
    reader = internal_gguf.GGUFReader(_write_f8_e4m3_gguf(tmp_path, payload))

    tensor = reader.tensors[0]
    assert tensor.tensor_type == internal_gguf.GGMLQuantizationType.F8_E4M3
    # GGUF stores dimensions fastest-varying first; the reader reverses them.
    assert tuple(int(dimension) for dimension in tensor.shape) == payload.shape[::-1]

    expected = torch.from_numpy(payload).view(torch.float8_e4m3fn).to(torch.float32).numpy()
    np.testing.assert_array_equal(internal_gguf.dequantize(tensor.data, tensor.tensor_type), expected)


def test_ggml_version_helpers_accept_1_0_and_reject_older_builds():
    assert parse_ggml_version("1.0.0") == GGML_1_0
    assert parse_ggml_version("v1.0.0") == GGML_1_0
    assert parse_ggml_version(b"1.0.0") == GGML_1_0
    assert parse_ggml_version("b4567") is None
    assert parse_ggml_version("") is None

    assert ggml_version_at_least("1.0.0")
    assert ggml_version_at_least("1.0.1")
    assert not ggml_version_at_least("0.29.0")
    assert not ggml_version_at_least("b4567")
    assert is_ggml_1_0_or_newer("1.0.0")
    assert not is_ggml_1_0_or_newer(None)


def test_ggml_bridge_reports_the_loaded_ggml_release():
    from gptqmodel.nn_modules.qlinear.gguf_cpp import _GGMLBridge

    bridge = object.__new__(_GGMLBridge)
    bridge._ggml_version_fn = lambda: b"1.0.0"
    assert bridge.ggml_version == "1.0.0"
    assert bridge.supports_ggml_1_0()

    bridge._ggml_version_fn = lambda: b"0.29.0"
    assert bridge.ggml_version == "0.29.0"
    assert not bridge.supports_ggml_1_0()

    bridge._ggml_version_fn = None
    assert bridge.ggml_version is None
    assert not bridge.supports_ggml_1_0()


def test_gguf_extra_requires_the_ggml_1_0_llama_cpp_release():
    tomllib = pytest.importorskip("tomllib")
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]

    gguf_extra = extras["gguf"]
    assert "llama-cpp-python>=0.3.37" in gguf_extra
    # Loading a `.gguf` file goes through Transformers' own GGUF integration,
    # which needs the real `gguf` package: GPT-QModel never substitutes its
    # reader for it.
    assert any(dependency.startswith("gguf>=") for dependency in gguf_extra)
    # The GGUF reader/dequantizers import NumPy, which is no longer a base
    # requirement on this branch.
    assert any(dependency.startswith("numpy") for dependency in gguf_extra)
