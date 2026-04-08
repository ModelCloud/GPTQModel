# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tabulate import tabulate
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from gptqmodel.utils.model import get_state_dict_for_save, move_to, streaming_state_dict_to_shards
from gptqmodel.utils.offload import offload_to_disk, undo_offload_to_disk
from gptqmodel.utils.structure import LazySafetensorsTurtle, alias_all_from_turtle_if_meta, alias_from_turtle_for_submodule


class _LinearWithBuffers(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.register_buffer("scale_buffer", torch.linspace(0.0, 1.0, out_features))
        self.register_buffer("mask_buffer", torch.randint(0, 2, (out_features, in_features)).bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.mask_buffer.float()) * self.scale_buffer


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


class _HybridBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.inner = nn.Linear(width, width, bias=False)
        self.dt_bias = nn.Parameter(torch.randn(width))
        self.register_buffer("dt_scale", torch.linspace(0.0, 1.0, width))


class _HybridWrapper(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = _HybridBlock(width)


class _SharedDirectBlock(nn.Module):
    def __init__(self, width: int, shared_bias: nn.Parameter):
        super().__init__()
        self.inner = nn.Linear(width, width, bias=False)
        self.dt_bias = shared_bias


class _SharedDirectWrapper(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        shared_bias = nn.Parameter(torch.randn(width))
        self.left = _SharedDirectBlock(width, shared_bias)
        self.right = _SharedDirectBlock(width, shared_bias)


class _CustomParameter(nn.Parameter):
    pass


class _CustomParamBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.inner = nn.Linear(width, width, bias=False)
        self.dt_bias = _CustomParameter(torch.randn(width))


class _CustomParamWrapper(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = _CustomParamBlock(width)


def _tiny_llama_config() -> LlamaConfig:
    # Keep the rotary test cheap while still using the real HF module init path.
    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


class _RotaryWrapper(nn.Module):
    # Pair one checkpoint-backed tensor with a non-persistent rotary module.
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.block = nn.Module()
        self.block.linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.block.rotary = LlamaRotaryEmbedding(config, device=torch.device("cpu"))


def _write_checkpoint_index(path: Path, shard_name: str, state_dict: dict[str, torch.Tensor]) -> None:
    weight_map = {name: shard_name for name in state_dict}
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _build_lazy_turtle_from_module(tmp_path: Path, model: nn.Module) -> LazySafetensorsTurtle:
    """Persist cloned checkpoint values and reopen them through the lazy turtle source."""

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    # Checkpoint-backed lazy turtle reconstructs tensor values, not Python-side alias identity.
    state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    save_file(state_dict, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, state_dict)
    source = LazySafetensorsTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert source is not None
    return source


def test_offload_to_disk_writes_single_dat_file(tmp_path):
    model = _LinearWithBuffers(in_features=128, out_features=96)
    original_state = _clone_state_dict(model.linear)

    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=model.linear, model=model, disk_path=str(offload_root))

    module_dir = offload_root / "linear"
    assert module_dir.is_dir(), "Expected per-module directory to exist"

    files = sorted(module_dir.iterdir(), key=lambda p: p.name)
    rows = [(path.name, path.stat().st_size) for path in files]
    print(tabulate(rows, headers=["file", "bytes"], tablefmt="github"))

    safetensor_files = [path for path in files if path.suffix == ".safetensors"]
    assert len(safetensor_files) == 1, "offload_to_disk should produce exactly one safetensors file per module"
    assert safetensor_files[0].name == "module.safetensors"

    with open(module_dir / "index.json", encoding="utf-8") as fp:
        index = json.load(fp)

    expected_keys = set(model.linear.state_dict().keys())
    assert set(index.keys()) == expected_keys
    assert all(Path(entry.get("safetensors_file")).name == "module.safetensors" for entry in index.values())
    assert all(entry.get("data_offsets") is not None for entry in index.values())

    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
    expected_files, tensor_to_filename, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )

    assert len(expected_files) == 1
    shard_path = save_dir / expected_files[0]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        for name, tensor in original_state.items():
            saved = handler.get_tensor(f"linear.{name}")
            torch.testing.assert_close(saved, tensor)

    # Materialize the module back and ensure values match the snapshot captured before offload.
    undo_offload_to_disk(model.linear, delete_offload_folders=False)
    for name, tensor in model.linear.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name])


def test_alias_all_from_turtle_restores_direct_meta_tensors_with_offloaded_children(tmp_path):
    source_model = _HybridWrapper(width=64)
    shell_model = _HybridWrapper(width=64)
    shell_model.load_state_dict(source_model.state_dict())
    turtle_model = _build_lazy_turtle_from_module(tmp_path, source_model)

    original_state = _clone_state_dict(source_model)
    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=shell_model.block.inner, model=shell_model, disk_path=str(offload_root))

    shell_model.block.dt_bias = nn.Parameter(
        torch.empty_like(shell_model.block.dt_bias, device="meta"),
        requires_grad=shell_model.block.dt_bias.requires_grad,
    )
    shell_model.block.register_buffer(
        "dt_scale",
        torch.empty_like(shell_model.block.dt_scale, device="meta"),
        persistent=True,
    )

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=turtle_model)

    state_dict = get_state_dict_for_save(shell_model, offload_root=str(offload_root))
    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    expected_files, _, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )

    shard_path = save_dir / expected_files[0]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        for name, tensor in original_state.items():
            saved = handler.get_tensor(name)
            torch.testing.assert_close(saved, tensor)


def test_alias_all_from_turtle_preserves_shell_dtype_for_direct_meta_tensors(tmp_path):
    source_model = _HybridWrapper(width=64)
    shell_model = _HybridWrapper(width=64)
    shell_model.load_state_dict(source_model.state_dict())
    turtle_model = _build_lazy_turtle_from_module(tmp_path, source_model)

    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=shell_model.block.inner, model=shell_model, disk_path=str(offload_root))

    shell_model.block.dt_bias = nn.Parameter(
        torch.empty(shell_model.block.dt_bias.shape, dtype=torch.float16, device="meta"),
        requires_grad=shell_model.block.dt_bias.requires_grad,
    )
    shell_model.block.register_buffer(
        "dt_scale",
        torch.empty(shell_model.block.dt_scale.shape, dtype=torch.float16, device="meta"),
        persistent=True,
    )

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=turtle_model)

    assert shell_model.block.dt_bias.dtype == torch.float16
    assert shell_model.block.dt_scale.dtype == torch.float16
    torch.testing.assert_close(shell_model.block.dt_bias, source_model.block.dt_bias.to(torch.float16))
    torch.testing.assert_close(shell_model.block.dt_scale, source_model.block.dt_scale.to(torch.float16))


def test_alias_all_from_turtle_materializes_shared_value_tensors(tmp_path):
    source_model = _SharedDirectWrapper(width=64)
    shell_model = _SharedDirectWrapper(width=64)
    shell_model.load_state_dict(source_model.state_dict())
    turtle_model = _build_lazy_turtle_from_module(tmp_path, source_model)

    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=shell_model.left.inner, model=shell_model, disk_path=str(offload_root))
    offload_to_disk(module=shell_model.right.inner, model=shell_model, disk_path=str(offload_root))

    shell_model.left.dt_bias = nn.Parameter(torch.empty_like(shell_model.left.dt_bias, device="meta"))
    shell_model.right.dt_bias = nn.Parameter(torch.empty_like(shell_model.right.dt_bias, device="meta"))

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=turtle_model)

    torch.testing.assert_close(shell_model.left.dt_bias, source_model.left.dt_bias)
    torch.testing.assert_close(shell_model.right.dt_bias, source_model.right.dt_bias)


def test_alias_all_from_turtle_materializes_custom_parameter_checkpoint_values(tmp_path):
    source_model = _CustomParamWrapper(width=16)
    shell_model = _CustomParamWrapper(width=16)
    shell_model.load_state_dict(source_model.state_dict())
    turtle_model = _build_lazy_turtle_from_module(tmp_path, source_model)

    shell_model.block.dt_bias = nn.Parameter(torch.empty_like(shell_model.block.dt_bias, device="meta"))

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=turtle_model)

    torch.testing.assert_close(shell_model.block.dt_bias, source_model.block.dt_bias)


def test_lazy_safetensors_turtle_materializes_recursive_submodule(tmp_path):
    source_model = _HybridWrapper(width=16)
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()

    shard_name = "model.safetensors"
    save_file(source_model.state_dict(), str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, source_model.state_dict())

    shell_model = _HybridWrapper(width=16)
    shell_model.load_state_dict(source_model.state_dict())
    shell_model.block.inner.weight = nn.Parameter(
        torch.empty_like(shell_model.block.inner.weight, device="meta"),
        requires_grad=shell_model.block.inner.weight.requires_grad,
    )
    shell_model.block.dt_bias = nn.Parameter(
        torch.empty_like(shell_model.block.dt_bias, device="meta"),
        requires_grad=shell_model.block.dt_bias.requires_grad,
    )
    shell_model.block.register_buffer(
        "dt_scale",
        torch.empty_like(shell_model.block.dt_scale, device="meta"),
        persistent=True,
    )

    source = LazySafetensorsTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )

    assert source is not None

    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=source,
        target_submodule=shell_model.block,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(shell_model.block.inner.weight, source_model.block.inner.weight)
    torch.testing.assert_close(shell_model.block.dt_bias, source_model.block.dt_bias)
    torch.testing.assert_close(shell_model.block.dt_scale, source_model.block.dt_scale)


def test_lazy_safetensors_turtle_restores_nonpersistent_buffers_from_module_init(tmp_path):
    config = _tiny_llama_config()
    source_model = _RotaryWrapper(config)
    shell_model = _RotaryWrapper(config)
    shell_model.load_state_dict(source_model.state_dict())

    shell_model.block.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.block.linear.weight, device="meta"),
        requires_grad=shell_model.block.linear.weight.requires_grad,
    )
    shell_model.block.rotary.register_buffer(
        "inv_freq",
        torch.empty_like(shell_model.block.rotary.inv_freq, device="meta"),
        persistent=False,
    )
    shell_model.block.rotary.register_buffer(
        "original_inv_freq",
        torch.empty_like(shell_model.block.rotary.original_inv_freq, device="meta"),
        persistent=False,
    )

    source = _build_lazy_turtle_from_module(tmp_path, source_model)

    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=source,
        target_submodule=shell_model.block,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(shell_model.block.linear.weight, source_model.block.linear.weight)
    torch.testing.assert_close(shell_model.block.rotary.inv_freq, source_model.block.rotary.inv_freq)
    torch.testing.assert_close(shell_model.block.rotary.original_inv_freq, source_model.block.rotary.original_inv_freq)
    assert shell_model.block.rotary.inv_freq.device.type == "cpu"
    assert shell_model.block.rotary._non_persistent_buffers_set == {"inv_freq", "original_inv_freq"}


def test_alias_all_from_lazy_turtle_restores_direct_meta_tensors(tmp_path):
    source_model = _HybridWrapper(width=16)
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()

    shard_name = "model.safetensors"
    save_file(source_model.state_dict(), str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, source_model.state_dict())

    shell_model = _HybridWrapper(width=16)
    shell_model.load_state_dict(source_model.state_dict())
    shell_model.block.dt_bias = nn.Parameter(
        torch.empty_like(shell_model.block.dt_bias, device="meta"),
        requires_grad=shell_model.block.dt_bias.requires_grad,
    )
    shell_model.block.register_buffer(
        "dt_scale",
        torch.empty_like(shell_model.block.dt_scale, device="meta"),
        persistent=True,
    )

    source = LazySafetensorsTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )

    assert source is not None

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=source)

    torch.testing.assert_close(shell_model.block.dt_bias, source_model.block.dt_bias)
    torch.testing.assert_close(shell_model.block.dt_scale, source_model.block.dt_scale)


def test_streaming_state_dict_pads_safetensors_header_to_8_bytes(tmp_path):
    model = nn.Linear(3, 5, bias=False)
    state_dict = get_state_dict_for_save(model)

    # Force an unaligned raw header so the regression test would fail without padding.
    metadata = {"format": "pt"}
    for size in range(1, 33):
        candidate = dict(metadata, pad=("x" * size))
        raw_header = {
            "__metadata__": candidate,
            "weight": {
                "dtype": "F32",
                "shape": list(model.weight.shape),
                "data_offsets": [0, model.weight.numel() * model.weight.element_size()],
            },
        }
        raw_header_len = len(json.dumps(raw_header, separators=(",", ":")).encode("utf-8"))
        if raw_header_len % 8 != 0:
            metadata = candidate
            break
    else:
        raise AssertionError("Failed to construct an unaligned safetensors header for the regression test.")

    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    expected_files, _, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata=metadata,
        max_shard_size=None,
    )

    shard_path = save_dir / expected_files[0]
    with shard_path.open("rb") as handle:
        stored_header_len = struct.unpack("<Q", handle.read(8))[0]

    assert raw_header_len % 8 != 0
    assert stored_header_len % 8 == 0

    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        torch.testing.assert_close(handler.get_tensor("weight"), model.weight.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for offload rematerialization test")
def test_move_to_restores_offloaded_meta_leaves_before_gpu_transfer(tmp_path):
    model = _LinearWithBuffers(in_features=32, out_features=24)
    original_state = _clone_state_dict(model.linear)

    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=model.linear, model=model, disk_path=str(offload_root))

    restored = move_to(model.linear, device=torch.device("cuda", 0))

    for name, tensor in restored.state_dict().items():
        assert tensor.device.type == "cuda"
        torch.testing.assert_close(tensor.cpu(), original_state[name])
