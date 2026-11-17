# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF loader/dequant selection is a config policy, never an environment switch.

``GPTQMODEL_INTERNAL_GGUF_QUANTIZED_LOADER``, ``..._TORCH_LOADER``,
``..._DEQUANT_DEVICE`` and ``..._DEQUANT_MAX_BYTES`` decided which parser ran,
which device the dense import dequantized on, and how much transient memory one
chunk could take. Two processes with the same public call options therefore
could load the same checkpoint along different code paths. The replacement
fields live on ``KernelConfig`` (and the shared ``RuntimeConfig`` adapter), and
these tests pin that the environment no longer participates.
"""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gptqmodel
from gptqmodel.models import loader
from gptqmodel.utils import hf as hf_utils
from gptqmodel.utils import internal_gguf
from gptqmodel.utils.kernel import KernelConfig, get_kernel_config, reset_kernel_config


RETIRED = (
    "GPTQMODEL_INTERNAL_GGUF_QUANTIZED_LOADER",
    "GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER",
    "GPTQMODEL_INTERNAL_GGUF_DEQUANT_DEVICE",
    "GPTQMODEL_INTERNAL_GGUF_DEQUANT_MAX_BYTES",
)


@pytest.fixture(autouse=True)
def _clean_policy():
    reset_kernel_config()
    try:
        yield
    finally:
        reset_kernel_config()


def _encode_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _write_qwen3_gguf(path, *, tensor_type=8, block_size=32, type_size=34):
    """Write one minimal Qwen3 GGUF with a single quantized layout."""

    payload = bytearray()
    payload.extend(struct.pack("<I", internal_gguf.GGUF_MAGIC))
    payload.extend(struct.pack("<I", internal_gguf.GGUF_VERSION))
    payload.extend(struct.pack("<Q", 2))  # tensor count
    payload.extend(struct.pack("<Q", 1))  # kv count
    payload.extend(_encode_string("general.architecture"))
    payload.extend(struct.pack("<I", int(internal_gguf.GGUFValueType.STRING)))
    payload.extend(_encode_string("qwen3"))
    for name in ("blk.0.attn_q.weight", "output.weight"):
        payload.extend(_encode_string(name))
        payload.extend(struct.pack("<I", 1))
        payload.extend(struct.pack("<Q", block_size))
        payload.extend(struct.pack("<I", tensor_type))
        payload.extend(struct.pack("<Q", 0))
    payload.extend(b"\x00" * ((-len(payload)) % internal_gguf.GGUF_DEFAULT_ALIGNMENT))
    block = bytearray(type_size * 1)
    block[0:2] = np.array([0.5], dtype=np.float16).view(np.uint8).tobytes()
    payload.extend(bytes(block) * 2)
    path.write_bytes(bytes(payload))
    return str(path)


def _dequant_row(scale=0.75):
    scale_bytes = np.array([scale], dtype=np.float16).view(np.uint8)
    sign_bits = (np.arange(128, dtype=np.uint8) % 5 < 2).astype(np.uint8)
    packed = np.packbits(sign_bits, bitorder="little")
    return np.concatenate([scale_bytes, packed]).reshape(1, -1), sign_bits


def test_native_quantized_loader_is_on_by_default_and_ignores_environment(monkeypatch):
    for name, value in zip(RETIRED, ("0", "0", "cpu", "1")):
        monkeypatch.setenv(name, value)

    # The retired variables are reported, never honoured.
    assert loader._native_quantized_loader_enabled() is True
    assert internal_gguf.native_quantized_loader_enabled() is True
    # No dequant device was requested, so the dense import keeps the NumPy path.
    assert internal_gguf._resolve_torch_dequant_device() is None
    assert internal_gguf.torch_dequant_loader_enabled() is False


def test_native_quantized_loader_policy_selects_the_parser(tmp_path, monkeypatch):
    checkpoint = _write_qwen3_gguf(tmp_path / "qwen3-q8_0.gguf")
    monkeypatch.setenv("GPTQMODEL_INTERNAL_GGUF_QUANTIZED_LOADER", "0")

    enabled = loader._resolve_native_quantized_gguf_checkpoint(
        str(tmp_path),
        {"gguf_file": "qwen3-q8_0.gguf"},
        policy=KernelConfig(),
    )
    disabled = loader._resolve_native_quantized_gguf_checkpoint(
        str(tmp_path),
        {"gguf_file": "qwen3-q8_0.gguf"},
        policy=KernelConfig(gguf_native_quantized_loader=False),
    )

    path, spec = enabled
    assert path == checkpoint
    assert spec is not None
    assert spec.model_type == internal_gguf.MODEL_ARCH_QWEN3
    assert spec.bits_alias == "q8_0"
    assert disabled == (None, None)


def test_dequant_device_policy_selects_torch_path_and_environment_does_not(monkeypatch):
    row, sign_bits = _dequant_row()
    expected = np.where(sign_bits == 1, np.float32(0.75), np.float32(-0.75)).reshape(1, 128)

    calls = []
    original = internal_gguf._dequantize_sign_only_torch_to_numpy

    def _wrapped(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(internal_gguf, "_dequantize_sign_only_torch_to_numpy", _wrapped)
    monkeypatch.setenv("GPTQMODEL_INTERNAL_GGUF_DEQUANT_DEVICE", "cpu")

    # Default policy: NumPy reference path, even with the retired variable set.
    default = internal_gguf.dequantize(row, internal_gguf.GGMLQuantizationType.Q1_0_g128)
    assert calls == []
    np.testing.assert_allclose(default, expected, rtol=0.0, atol=0.0)

    # Explicit policy: the Torch conversion on the requested device.
    configured = internal_gguf.dequantize(
        row,
        internal_gguf.GGMLQuantizationType.Q1_0_g128,
        policy=KernelConfig(gguf_dequant_device="cpu"),
    )
    assert len(calls) == 1
    assert calls[0]["device"] == torch.device("cpu")
    np.testing.assert_allclose(configured, expected, rtol=0.0, atol=0.0)


def test_dequant_chunk_bytes_policy_controls_the_chunk_size(monkeypatch):
    # One fp32 output row of 128 values is 512 bytes, so 1500 bytes allows 2 rows.
    kwargs = {"packed_row_bytes": 18, "block_size": 128, "type_size": 18}
    monkeypatch.setenv("GPTQMODEL_INTERNAL_GGUF_DEQUANT_MAX_BYTES", "1")

    assert internal_gguf._resolve_torch_dequant_chunk_rows(
        **kwargs, policy=KernelConfig(gguf_dequant_chunk_bytes=1500)
    ) == 2
    # The retired variable cannot shrink the budget below the configured one.
    assert internal_gguf._resolve_torch_dequant_chunk_rows(**kwargs) == 256 * 1024 * 1024 // 512
    with pytest.raises(ValueError, match="gguf_dequant_chunk_bytes"):
        KernelConfig(gguf_dequant_chunk_bytes=0)


def test_torch_dequant_loader_policy_drives_the_dense_import(monkeypatch):
    class _Processor:
        def __init__(self, *, config):
            self.config = config

        @staticmethod
        def process(*, weights, name, tensor_key_mapping, parsed_parameters):
            del tensor_key_mapping, parsed_parameters
            return SimpleNamespace(weights=torch.as_tensor(weights), name=name)

    class _GgufUtils:
        TensorProcessor = _Processor
        TENSOR_PROCESSORS = {}

        @staticmethod
        def get_gguf_hf_weights_map(model_to_load, processor):
            del model_to_load, processor
            return {"blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight"}

        @staticmethod
        def tqdm(iterable, **_kwargs):
            return iterable

    calls = []
    tensor = SimpleNamespace(
        name="blk.0.attn_q.weight",
        tensor_type=internal_gguf.GGMLQuantizationType.F32,
        data=np.zeros((1, 4), dtype=np.float32),
    )

    def _parse(*_args, return_tensors, **_kwargs):
        calls.append(return_tensors)
        if not return_tensors:
            return {"config": {"model_type": internal_gguf.MODEL_ARCH_QWEN3}}
        return {"variant": "original"}

    monkeypatch.setattr(internal_gguf, "GGUFReader", lambda _path: SimpleNamespace(tensors=[tensor]))
    monkeypatch.setattr(
        internal_gguf,
        "dequantize_to_torch",
        lambda data, _qtype, *, device=None, dtype=None: torch.zeros((1, 4), device=device, dtype=dtype),
    )
    monkeypatch.setenv("GPTQMODEL_INTERNAL_GGUF_TORCH_LOADER", "1")

    def _run(policy):
        return hf_utils._load_gguf_checkpoint_torch_direct(
            gguf_utils=_GgufUtils,
            original_load_gguf_checkpoint=_parse,
            gguf_checkpoint_path="bonsai.gguf",
            return_tensors=True,
            model_to_load=object(),
            policy=policy,
        )

    # The retired variable does not enable the direct loader on its own.
    assert _run(KernelConfig()) == {"variant": "original"}
    assert calls == [True]

    # Explicit policy: the direct loader converts once, on the configured device.
    direct = _run(KernelConfig(gguf_torch_dequant_loader=True, gguf_dequant_device="cpu"))
    assert calls == [True, False]
    loaded = direct["tensors"]["model.layers.0.self_attn.q_proj.weight"]
    assert isinstance(loaded, torch.Tensor)
    assert loaded.device.type == "cpu"


def test_torch_dequant_loader_resolution_and_explicit_disable():
    assert internal_gguf.torch_dequant_loader_enabled(KernelConfig()) is False
    assert internal_gguf.torch_dequant_loader_enabled(
        KernelConfig(gguf_dequant_device="cpu")
    ) is True
    # An explicit value wins over the device-derived default.
    assert internal_gguf.torch_dequant_loader_enabled(
        KernelConfig(gguf_torch_dequant_loader=False, gguf_dequant_device="cpu")
    ) is False


@pytest.mark.parametrize("device", ["", "foo", "xpu", "meta"])
def test_invalid_dequant_device_is_rejected(device):
    with pytest.raises((TypeError, ValueError)):
        KernelConfig(gguf_dequant_device=device)


def test_unsupported_dequant_device_ordinal_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert internal_gguf._resolve_torch_dequant_device(
        KernelConfig(gguf_dequant_device="cuda:1")
    ) == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match="outside the 2 device"):
        internal_gguf._resolve_torch_dequant_device(KernelConfig(gguf_dequant_device="cuda:2"))


def test_unavailable_accelerator_is_rejected_not_downgraded(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="no available CUDA device"):
        internal_gguf._resolve_torch_dequant_device(KernelConfig(gguf_dequant_device="cuda"))


def test_retired_gguf_env_vars_warn_and_have_no_effect(monkeypatch):
    calls = []

    class _Recorder:
        def warn(self, message):
            calls.append(message)

    monkeypatch.setattr(gptqmodel, "log", _Recorder())
    monkeypatch.setattr(gptqmodel, "_WARNED_RETIRED_KERNEL_ENV_VARS", set())
    for name in RETIRED:
        monkeypatch.setenv(name, "1")

    warned = set(gptqmodel.warn_retired_kernel_env_vars())

    assert warned == set(RETIRED)
    text = "\n".join(calls)
    assert "KernelConfig.gguf_native_quantized_loader" in text
    assert "KernelConfig.gguf_torch_dequant_loader" in text
    assert "KernelConfig.gguf_dequant_device" in text
    assert "KernelConfig.gguf_dequant_chunk_bytes" in text
    # The same call options resolve to the same policy with every variable set.
    assert get_kernel_config() == KernelConfig()


def test_public_load_uses_the_kernel_policy_to_choose_the_gguf_parser(tmp_path, monkeypatch):
    """Same call, same checkpoint: the option, not the environment, decides."""

    from transformers import GenerationConfig

    from gptqmodel.utils.hf import INTERNAL_HF_GGUF_FILE_KWARG

    checkpoint = _write_qwen3_gguf(tmp_path / "qwen3-q8_0.gguf")
    model_dir = str(tmp_path)
    monkeypatch.setenv("GPTQMODEL_INTERNAL_GGUF_QUANTIZED_LOADER", "0")

    class _Config:
        model_type = "qwen3"
        sub_configs = {}
        dtype = None
        architectures = ["FakeModelForCausalLM"]
        _experts_implementation = None

        def to_dict(self):
            return {"max_position_embeddings": 128}

    class _Model:
        def __init__(self, config):
            self.config = config
            self.generation_config = GenerationConfig()

        def eval(self):
            return self

    dense_calls = []
    redirected = []

    @loader.ModelLoader
    class DummyQModel:
        loader = None
        require_dtype = None
        loader_requires_dtype = False
        require_fast_init = False
        require_trust_remote_code = False
        require_pkgs = []
        supports_desc_act = [True, False]
        support_offload_to_disk = False
        config_class = None

        @staticmethod
        def before_model_load(*_args, **_kwargs):
            return None

        def __init__(self, model, **_kwargs):
            self.model = model
            self.config = model.config
            self.quantized = False
            self.quantize_config = None

    class _Inner:
        @staticmethod
        def from_pretrained(path, config=None, **kwargs):
            dense_calls.append({"path": path, "kwargs": kwargs})
            return _Model(config)

    DummyQModel.loader = _Inner

    def _normalize(_model_id_or_path, kwargs, *, api_name):
        assert api_name == "DummyQModel.from_pretrained"
        kwargs[INTERNAL_HF_GGUF_FILE_KWARG] = "qwen3-q8_0.gguf"
        return model_dir

    monkeypatch.setattr(loader, "check_versions", lambda *_a, **_k: None)
    monkeypatch.setattr(loader, "normalize_model_id_or_path_for_hf_gguf", _normalize)
    monkeypatch.setattr(loader, "get_model_local_path", lambda *_a, **_k: model_dir)
    monkeypatch.setattr(loader, "auto_select_device", lambda *_a, **_k: torch.device("cpu"))
    monkeypatch.setattr(loader, "auto_dtype", lambda *_a, **_k: torch.bfloat16)
    monkeypatch.setattr(loader, "load_hf_tokenizer", lambda *_a, **_k: object())
    monkeypatch.setattr(loader, "print_module_tree", lambda *_a, **_k: None)
    monkeypatch.setattr(loader.defuser, "replace_fused_blocks", lambda *_a, **_k: None)
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *_a, **_k: _Config())
    monkeypatch.setattr(
        DummyQModel,
        "from_quantized",
        classmethod(lambda cls, model_id_or_path, **kwargs: redirected.append(kwargs) or "redirected"),
    )

    # AUTO on a Q8_0 checkpoint redirects to the native quantized loader.
    assert DummyQModel.from_pretrained(checkpoint, quantize_config=None) == "redirected"
    assert len(redirected) == 1

    # An explicit opt-out resolves the dense import instead -- with the retired
    # environment variable still set to "0", which cannot force either path.
    result = DummyQModel.from_pretrained(
        checkpoint,
        quantize_config=None,
        kernel=KernelConfig(gguf_native_quantized_loader=False),
    )
    assert len(redirected) == 1
    assert result.quantized is False
    assert dense_calls[0]["kwargs"]["gguf_file"] == "qwen3-q8_0.gguf"
