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
from gptqmodel.utils.structure import (
    LazyTurtle,
    alias_all_from_turtle_if_meta,
    alias_from_turtle_for_submodule,
)


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


class _TransformerPrefixedHybridWrapper(nn.Module):
    """Wrap the real module tree under an extra root to mimic shell-only prefixes."""

    def __init__(self, width: int):
        super().__init__()
        self.transformer = _HybridWrapper(width)


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


class _AttrBufferTemplate(nn.Module):
    """Template module whose non-persistent buffers depend on constructor attributes."""

    def __init__(self, width: int, scale: float = 1.0, device=None):
        super().__init__()
        self.width = width
        self.scale = scale
        base = torch.arange(width, dtype=torch.float32, device=device)
        self.register_buffer("cache", base * scale, persistent=False)
        self.register_buffer("cache_plus_one", base + 1, persistent=False)


class _AttrBufferWrapper(nn.Module):
    """Hybrid wrapper that pairs checkpoint tensors with attribute-driven init-only buffers."""

    def __init__(self, width: int, scale: float = 1.0):
        super().__init__()
        self.block = nn.Module()
        self.block.linear = nn.Linear(width, width, bias=False)
        self.block.rotary = _AttrBufferTemplate(width=width, scale=scale, device=torch.device("cpu"))


class _ScalarMetaBufferTemplate(nn.Module):
    """Template module that keeps the constructor scalar separately from the registered buffer."""

    def __init__(self, scale: float, device=None):
        super().__init__()
        self.scalar_scale = scale
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32, device=device), persistent=False)


class _ScalarMetaBufferWrapper(nn.Module):
    """Wrapper used to cover constructors that take a scalar but expose a same-named buffer."""

    def __init__(self, scale: float):
        super().__init__()
        self.block = nn.Module()
        self.block.linear = nn.Linear(8, 8, bias=False)
        self.block.scale_holder = _ScalarMetaBufferTemplate(scale=scale, device=torch.device("cpu"))


class _SplitGateUpBlock(nn.Module):
    """Tiny stand-in for Defuser runtime MLPs that expose split projections from a fused checkpoint tensor."""

    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)


class _SplitGateUpWrapper(nn.Module):
    """Wrapper used to exercise lazy-turtle rematerialization from `gate_up_proj` checkpoints."""

    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.block = _SplitGateUpBlock(width, intermediate)


class _TinyExpert(nn.Module):
    """Small expert used to mirror Defuser's split expert runtime layout."""

    def __init__(self, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=True)
        self.up_proj = nn.Linear(width, width, bias=True)
        self.down_proj = nn.Linear(width, width, bias=True)


class _RectExpert(nn.Module):
    """Expert with distinct hidden/intermediate sizes to catch wrong split/transpose rules."""

    def __init__(self, width: int, intermediate: int, *, bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=bias)
        self.up_proj = nn.Linear(width, intermediate, bias=bias)
        self.down_proj = nn.Linear(intermediate, width, bias=bias)


class _FusedExpertsWrapper(nn.Module):
    """Wrapper used to exercise fused expert checkpoint slicing during lazy-turtle rematerialization."""

    def __init__(self, width: int, expert_count: int):
        super().__init__()
        self.block = nn.Module()
        self.block.experts = nn.Module()
        for expert_idx in range(expert_count):
            self.block.experts.add_module(str(expert_idx), _TinyExpert(width))


class _RectFusedExpertsWrapper(nn.Module):
    """Wrapper used to validate real rectangular expert layouts, including transposed storage."""

    def __init__(self, width: int, intermediate: int, expert_count: int, *, is_transposed: bool):
        super().__init__()
        self.block = nn.Module()
        self.block.experts = nn.Module()
        self.block.experts.is_transposed = is_transposed
        for expert_idx in range(expert_count):
            self.block.experts.add_module(str(expert_idx), _RectExpert(width, intermediate))


def _write_checkpoint_index(path: Path, shard_name: str, state_dict: dict[str, torch.Tensor]) -> None:
    weight_map = dict.fromkeys(state_dict, shard_name)
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _build_lazy_turtle_from_module(tmp_path: Path, model: nn.Module) -> LazyTurtle:
    """Persist cloned checkpoint values and reopen them through the lazy turtle source."""

    state_dict = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    return _build_lazy_turtle_from_checkpoint_tensors(tmp_path, state_dict)


def _build_lazy_turtle_from_checkpoint_tensors(tmp_path: Path, checkpoint_tensors: dict[str, torch.Tensor]) -> LazyTurtle:
    """Persist an arbitrary safetensors checkpoint and reopen it through LazyTurtle."""

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)
    source = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert source is not None
    return source


def _build_rect_fused_expert_checkpoint_tensors(
    source_model: _RectFusedExpertsWrapper,
    *,
    include_down_proj: bool = True,
) -> dict[str, torch.Tensor]:
    """Build fused expert checkpoint tensors that mimic the HF/Defuser expert storage layouts."""

    experts = [expert for _, expert in source_model.block.experts.named_children()]
    is_transposed = bool(getattr(source_model.block.experts, "is_transposed", False))
    gate_split_dim = 1 if is_transposed else 0

    checkpoint_tensors = {
        "block.experts.gate_up_proj": torch.stack(
            [
                torch.cat(
                    [
                        expert.gate_proj.weight.detach().clone().transpose(0, 1) if is_transposed else expert.gate_proj.weight.detach().clone(),
                        expert.up_proj.weight.detach().clone().transpose(0, 1) if is_transposed else expert.up_proj.weight.detach().clone(),
                    ],
                    dim=gate_split_dim,
                )
                for expert in experts
            ],
            dim=0,
        ),
        "block.experts.gate_up_proj_bias": torch.stack(
            [
                torch.cat(
                    [
                        expert.gate_proj.bias.detach().clone(),
                        expert.up_proj.bias.detach().clone(),
                    ],
                    dim=0,
                )
                for expert in experts
            ],
            dim=0,
        ),
    }

    if include_down_proj:
        checkpoint_tensors["block.experts.down_proj"] = torch.stack(
            [
                expert.down_proj.weight.detach().clone().transpose(0, 1) if is_transposed else expert.down_proj.weight.detach().clone()
                for expert in experts
            ],
            dim=0,
        )
        checkpoint_tensors["block.experts.down_proj_bias"] = torch.stack(
            [expert.down_proj.bias.detach().clone() for expert in experts],
            dim=0,
        )

    return checkpoint_tensors


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


def test_lazy_turtle_materializes_recursive_submodule(tmp_path):
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

    source = LazyTurtle.maybe_create(
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


def test_lazy_turtle_materializes_submodule_when_shell_has_extra_root_prefix(tmp_path):
    """Checkpoint-backed materialization should ignore wrapper prefixes absent from shard names."""

    source_model = _HybridWrapper(width=16)
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()

    shard_name = "model.safetensors"
    save_file(source_model.state_dict(), str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, source_model.state_dict())

    shell_model = _TransformerPrefixedHybridWrapper(width=16)
    shell_model.transformer.load_state_dict(source_model.state_dict())
    shell_model.transformer.block.inner.weight = nn.Parameter(
        torch.empty_like(shell_model.transformer.block.inner.weight, device="meta"),
        requires_grad=shell_model.transformer.block.inner.weight.requires_grad,
    )
    shell_model.transformer.block.dt_bias = nn.Parameter(
        torch.empty_like(shell_model.transformer.block.dt_bias, device="meta"),
        requires_grad=shell_model.transformer.block.dt_bias.requires_grad,
    )
    shell_model.transformer.block.register_buffer(
        "dt_scale",
        torch.empty_like(shell_model.transformer.block.dt_scale, device="meta"),
        persistent=True,
    )

    source = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )

    assert source is not None

    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=source,
        target_submodule=shell_model.transformer.block,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(shell_model.transformer.block.inner.weight, source_model.block.inner.weight)
    torch.testing.assert_close(shell_model.transformer.block.dt_bias, source_model.block.dt_bias)
    torch.testing.assert_close(shell_model.transformer.block.dt_scale, source_model.block.dt_scale)


def test_alias_all_from_lazy_turtle_handles_shell_root_prefix_mismatch(tmp_path):
    """Direct meta tensors should resolve through the same prefix-stripping checkpoint aliases."""

    source_model = _HybridWrapper(width=16)
    turtle_model = _build_lazy_turtle_from_module(tmp_path, source_model)

    shell_model = _TransformerPrefixedHybridWrapper(width=16)
    shell_model.transformer.load_state_dict(source_model.state_dict())
    shell_model.transformer.block.dt_bias = nn.Parameter(
        torch.empty_like(shell_model.transformer.block.dt_bias, device="meta"),
        requires_grad=shell_model.transformer.block.dt_bias.requires_grad,
    )
    shell_model.transformer.block.register_buffer(
        "dt_scale",
        torch.empty_like(shell_model.transformer.block.dt_scale, device="meta"),
        persistent=True,
    )

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=turtle_model)

    torch.testing.assert_close(shell_model.transformer.block.dt_bias, source_model.block.dt_bias)
    torch.testing.assert_close(shell_model.transformer.block.dt_scale, source_model.block.dt_scale)


def test_lazy_turtle_restores_nonpersistent_buffers_from_module_init(tmp_path):
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


def test_lazy_turtle_restores_nonpersistent_buffers_from_attribute_ctor(tmp_path):
    """Init-only buffers should rebuild from constructor attributes when no config argument exists."""

    source_model = _AttrBufferWrapper(width=16, scale=0.5)
    shell_model = _AttrBufferWrapper(width=16, scale=0.5)
    shell_model.load_state_dict(source_model.state_dict())

    shell_model.block.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.block.linear.weight, device="meta"),
        requires_grad=shell_model.block.linear.weight.requires_grad,
    )
    shell_model.block.rotary.register_buffer(
        "cache",
        torch.empty_like(shell_model.block.rotary.cache, device="meta"),
        persistent=False,
    )
    shell_model.block.rotary.register_buffer(
        "cache_plus_one",
        torch.empty_like(shell_model.block.rotary.cache_plus_one, device="meta"),
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
    torch.testing.assert_close(shell_model.block.rotary.cache, source_model.block.rotary.cache)
    torch.testing.assert_close(shell_model.block.rotary.cache_plus_one, source_model.block.rotary.cache_plus_one)
    assert shell_model.block.rotary.cache.device.type == "cpu"
    assert shell_model.block.rotary._non_persistent_buffers_set == {"cache", "cache_plus_one"}


def test_lazy_turtle_restores_nonpersistent_buffers_from_scalar_shadow_attr(tmp_path):
    """Scalar constructor args should not be reconstructed from same-named meta buffers."""

    source_model = _ScalarMetaBufferWrapper(scale=3.5)
    shell_model = _ScalarMetaBufferWrapper(scale=3.5)
    shell_model.load_state_dict(source_model.state_dict())

    shell_model.block.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.block.linear.weight, device="meta"),
        requires_grad=shell_model.block.linear.weight.requires_grad,
    )
    shell_model.block.scale_holder.register_buffer(
        "scale",
        torch.empty_like(shell_model.block.scale_holder.scale, device="meta"),
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
    torch.testing.assert_close(shell_model.block.scale_holder.scale, source_model.block.scale_holder.scale)
    assert shell_model.block.scale_holder.scale.device.type == "cpu"
    assert shell_model.block.scale_holder._non_persistent_buffers_set == {"scale"}


def test_lazy_turtle_materializes_split_gate_up_from_fused_checkpoint_tensor(tmp_path):
    """Defused runtime `gate_proj`/`up_proj` leaves should restore from fused checkpoint `gate_up_proj` weights."""

    source_model = _SplitGateUpWrapper(width=8, intermediate=6)
    shell_model = _SplitGateUpWrapper(width=8, intermediate=6)
    shell_model.load_state_dict(source_model.state_dict())

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    checkpoint_tensors = {
        "block.gate_up_proj.weight": torch.cat(
            [
                source_model.block.gate_proj.weight.detach().clone(),
                source_model.block.up_proj.weight.detach().clone(),
            ],
            dim=0,
        ),
        "block.down_proj.weight": source_model.block.down_proj.weight.detach().clone(),
    }
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)

    shell_model.block.gate_proj.weight = nn.Parameter(
        torch.empty_like(shell_model.block.gate_proj.weight, device="meta"),
        requires_grad=shell_model.block.gate_proj.weight.requires_grad,
    )
    shell_model.block.up_proj.weight = nn.Parameter(
        torch.empty_like(shell_model.block.up_proj.weight, device="meta"),
        requires_grad=shell_model.block.up_proj.weight.requires_grad,
    )
    shell_model.block.down_proj.weight = nn.Parameter(
        torch.empty_like(shell_model.block.down_proj.weight, device="meta"),
        requires_grad=shell_model.block.down_proj.weight.requires_grad,
    )

    source = LazyTurtle.maybe_create(
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

    torch.testing.assert_close(shell_model.block.gate_proj.weight, source_model.block.gate_proj.weight)
    torch.testing.assert_close(shell_model.block.up_proj.weight, source_model.block.up_proj.weight)
    torch.testing.assert_close(shell_model.block.down_proj.weight, source_model.block.down_proj.weight)


def test_lazy_turtle_materializes_split_experts_from_fused_checkpoint_tensors(tmp_path):
    """Fused expert checkpoint tensors should rematerialize defused `experts.<idx>.*` leaves."""

    source_model = _FusedExpertsWrapper(width=4, expert_count=2)
    shell_model = _FusedExpertsWrapper(width=4, expert_count=2)
    shell_model.load_state_dict(source_model.state_dict())

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    checkpoint_tensors = {
        "block.experts.gate_up_proj": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.weight.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.weight.detach().clone(),
                    ],
                    dim=1,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.gate_up_proj_bias": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.bias.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.bias.detach().clone(),
                    ],
                    dim=0,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.down_proj": torch.stack(
            [
                source_model.block.experts.get_submodule(str(expert_idx)).down_proj.weight.detach().clone()
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.down_proj_bias": torch.stack(
            [
                source_model.block.experts.get_submodule(str(expert_idx)).down_proj.bias.detach().clone()
                for expert_idx in range(2)
            ],
            dim=0,
        ),
    }
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)

    for expert_idx in range(2):
        expert = shell_model.block.experts.get_submodule(str(expert_idx))
        expert.gate_proj.weight = nn.Parameter(
            torch.empty_like(expert.gate_proj.weight, device="meta"),
            requires_grad=expert.gate_proj.weight.requires_grad,
        )
        expert.gate_proj.bias = nn.Parameter(
            torch.empty_like(expert.gate_proj.bias, device="meta"),
            requires_grad=expert.gate_proj.bias.requires_grad,
        )
        expert.up_proj.weight = nn.Parameter(
            torch.empty_like(expert.up_proj.weight, device="meta"),
            requires_grad=expert.up_proj.weight.requires_grad,
        )
        expert.up_proj.bias = nn.Parameter(
            torch.empty_like(expert.up_proj.bias, device="meta"),
            requires_grad=expert.up_proj.bias.requires_grad,
        )
        expert.down_proj.weight = nn.Parameter(
            torch.empty_like(expert.down_proj.weight, device="meta"),
            requires_grad=expert.down_proj.weight.requires_grad,
        )
        expert.down_proj.bias = nn.Parameter(
            torch.empty_like(expert.down_proj.bias, device="meta"),
            requires_grad=expert.down_proj.bias.requires_grad,
        )

    source = LazyTurtle.maybe_create(
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

    for expert_idx in range(2):
        expected = source_model.block.experts.get_submodule(str(expert_idx))
        actual = shell_model.block.experts.get_submodule(str(expert_idx))
        torch.testing.assert_close(actual.gate_proj.weight, expected.gate_proj.weight)
        torch.testing.assert_close(actual.gate_proj.bias, expected.gate_proj.bias)
        torch.testing.assert_close(actual.up_proj.weight, expected.up_proj.weight)
        torch.testing.assert_close(actual.up_proj.bias, expected.up_proj.bias)
        torch.testing.assert_close(actual.down_proj.weight, expected.down_proj.weight)
        torch.testing.assert_close(actual.down_proj.bias, expected.down_proj.bias)


def test_lazy_turtle_materializes_rectangular_qwen_style_experts_from_fused_checkpoint_tensors(tmp_path):
    """Non-transposed expert checkpoints should split gate/up along the output dimension."""

    source_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=False)
    shell_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=False)
    shell_model.load_state_dict(source_model.state_dict())

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    checkpoint_tensors = {
        "block.experts.gate_up_proj": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.weight.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.weight.detach().clone(),
                    ],
                    dim=0,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.gate_up_proj_bias": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.bias.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.bias.detach().clone(),
                    ],
                    dim=0,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.down_proj": torch.stack(
            [
                source_model.block.experts.get_submodule(str(expert_idx)).down_proj.weight.detach().clone()
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.down_proj_bias": torch.stack(
            [
                source_model.block.experts.get_submodule(str(expert_idx)).down_proj.bias.detach().clone()
                for expert_idx in range(2)
            ],
            dim=0,
        ),
    }
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)

    for expert_idx in range(2):
        expert = shell_model.block.experts.get_submodule(str(expert_idx))
        expert.gate_proj.weight = nn.Parameter(
            torch.empty_like(expert.gate_proj.weight, device="meta"),
            requires_grad=expert.gate_proj.weight.requires_grad,
        )
        expert.gate_proj.bias = nn.Parameter(
            torch.empty_like(expert.gate_proj.bias, device="meta"),
            requires_grad=expert.gate_proj.bias.requires_grad,
        )
        expert.up_proj.weight = nn.Parameter(
            torch.empty_like(expert.up_proj.weight, device="meta"),
            requires_grad=expert.up_proj.weight.requires_grad,
        )
        expert.up_proj.bias = nn.Parameter(
            torch.empty_like(expert.up_proj.bias, device="meta"),
            requires_grad=expert.up_proj.bias.requires_grad,
        )
        expert.down_proj.weight = nn.Parameter(
            torch.empty_like(expert.down_proj.weight, device="meta"),
            requires_grad=expert.down_proj.weight.requires_grad,
        )
        expert.down_proj.bias = nn.Parameter(
            torch.empty_like(expert.down_proj.bias, device="meta"),
            requires_grad=expert.down_proj.bias.requires_grad,
        )

    source = LazyTurtle.maybe_create(
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

    for expert_idx in range(2):
        expected = source_model.block.experts.get_submodule(str(expert_idx))
        actual = shell_model.block.experts.get_submodule(str(expert_idx))
        torch.testing.assert_close(actual.gate_proj.weight, expected.gate_proj.weight)
        torch.testing.assert_close(actual.gate_proj.bias, expected.gate_proj.bias)
        torch.testing.assert_close(actual.up_proj.weight, expected.up_proj.weight)
        torch.testing.assert_close(actual.up_proj.bias, expected.up_proj.bias)
        torch.testing.assert_close(actual.down_proj.weight, expected.down_proj.weight)
        torch.testing.assert_close(actual.down_proj.bias, expected.down_proj.bias)


def test_lazy_turtle_materializes_leaf_qwen_style_expert_gate_proj_from_fused_checkpoint_tensor(tmp_path):
    """Leaf expert gate_proj modules should resolve fused expert sources from module_path alone."""

    source_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=False)
    shell_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=False)
    shell_model.load_state_dict(source_model.state_dict())

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    checkpoint_tensors = {
        "block.experts.gate_up_proj": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.weight.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.weight.detach().clone(),
                    ],
                    dim=0,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.gate_up_proj_bias": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.bias.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.bias.detach().clone(),
                    ],
                    dim=0,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
    }
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)

    expert = shell_model.block.experts.get_submodule("1").gate_proj
    expert.weight = nn.Parameter(torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad)
    expert.bias = nn.Parameter(torch.empty_like(expert.bias, device="meta"), requires_grad=expert.bias.requires_grad)

    source = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert source is not None

    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=source,
        target_submodule=expert,
        device=torch.device("cpu"),
    )

    expected = source_model.block.experts.get_submodule("1").gate_proj
    torch.testing.assert_close(expert.weight, expected.weight)
    torch.testing.assert_close(expert.bias, expected.bias)


def test_lazy_turtle_materializes_rectangular_transposed_experts_from_fused_checkpoint_tensors(tmp_path):
    """Transposed expert checkpoints should transpose weights before matching defused leaves."""

    source_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=True)
    shell_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=True)
    shell_model.load_state_dict(source_model.state_dict())

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    checkpoint_tensors = {
        "block.experts.gate_up_proj": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.weight.detach().clone().transpose(0, 1),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.weight.detach().clone().transpose(0, 1),
                    ],
                    dim=1,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.gate_up_proj_bias": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.bias.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.bias.detach().clone(),
                    ],
                    dim=0,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.down_proj": torch.stack(
            [
                source_model.block.experts.get_submodule(str(expert_idx)).down_proj.weight.detach().clone().transpose(0, 1)
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.down_proj_bias": torch.stack(
            [
                source_model.block.experts.get_submodule(str(expert_idx)).down_proj.bias.detach().clone()
                for expert_idx in range(2)
            ],
            dim=0,
        ),
    }
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)

    for expert_idx in range(2):
        expert = shell_model.block.experts.get_submodule(str(expert_idx))
        expert.gate_proj.weight = nn.Parameter(
            torch.empty_like(expert.gate_proj.weight, device="meta"),
            requires_grad=expert.gate_proj.weight.requires_grad,
        )
        expert.gate_proj.bias = nn.Parameter(
            torch.empty_like(expert.gate_proj.bias, device="meta"),
            requires_grad=expert.gate_proj.bias.requires_grad,
        )
        expert.up_proj.weight = nn.Parameter(
            torch.empty_like(expert.up_proj.weight, device="meta"),
            requires_grad=expert.up_proj.weight.requires_grad,
        )
        expert.up_proj.bias = nn.Parameter(
            torch.empty_like(expert.up_proj.bias, device="meta"),
            requires_grad=expert.up_proj.bias.requires_grad,
        )
        expert.down_proj.weight = nn.Parameter(
            torch.empty_like(expert.down_proj.weight, device="meta"),
            requires_grad=expert.down_proj.weight.requires_grad,
        )
        expert.down_proj.bias = nn.Parameter(
            torch.empty_like(expert.down_proj.bias, device="meta"),
            requires_grad=expert.down_proj.bias.requires_grad,
        )

    source = LazyTurtle.maybe_create(
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

    for expert_idx in range(2):
        expected = source_model.block.experts.get_submodule(str(expert_idx))
        actual = shell_model.block.experts.get_submodule(str(expert_idx))
        torch.testing.assert_close(actual.gate_proj.weight, expected.gate_proj.weight)
        torch.testing.assert_close(actual.gate_proj.bias, expected.gate_proj.bias)
        torch.testing.assert_close(actual.up_proj.weight, expected.up_proj.weight)
        torch.testing.assert_close(actual.up_proj.bias, expected.up_proj.bias)
        torch.testing.assert_close(actual.down_proj.weight, expected.down_proj.weight)
        torch.testing.assert_close(actual.down_proj.bias, expected.down_proj.bias)


def test_lazy_turtle_materializes_leaf_transposed_expert_gate_proj_from_fused_checkpoint_tensor(tmp_path):
    """Leaf expert gate_proj modules should also honor transposed fused expert layouts."""

    source_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=True)
    shell_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=True)
    shell_model.load_state_dict(source_model.state_dict())

    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    checkpoint_tensors = {
        "block.experts.gate_up_proj": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.weight.detach().clone().transpose(0, 1),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.weight.detach().clone().transpose(0, 1),
                    ],
                    dim=1,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
        "block.experts.gate_up_proj_bias": torch.stack(
            [
                torch.cat(
                    [
                        source_model.block.experts.get_submodule(str(expert_idx)).gate_proj.bias.detach().clone(),
                        source_model.block.experts.get_submodule(str(expert_idx)).up_proj.bias.detach().clone(),
                    ],
                    dim=0,
                )
                for expert_idx in range(2)
            ],
            dim=0,
        ),
    }
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)

    expert = shell_model.block.experts.get_submodule("1").gate_proj
    expert.weight = nn.Parameter(torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad)
    expert.bias = nn.Parameter(torch.empty_like(expert.bias, device="meta"), requires_grad=expert.bias.requires_grad)

    source = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert source is not None

    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=source,
        target_submodule=expert,
        device=torch.device("cpu"),
    )

    expected = source_model.block.experts.get_submodule("1").gate_proj
    torch.testing.assert_close(expert.weight, expected.weight)
    torch.testing.assert_close(expert.bias, expected.bias)


def test_alias_all_from_turtle_materializes_leaf_qwen_style_expert_gate_proj_from_fused_checkpoint_tensor(tmp_path):
    """Direct-meta sync should resolve fused non-transposed expert tensors for leaf Linear modules."""

    source_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=False)
    shell_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=False)
    shell_model.load_state_dict(source_model.state_dict())

    expert = shell_model.block.experts.get_submodule("1").gate_proj
    expert.weight = nn.Parameter(torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad)
    expert.bias = nn.Parameter(torch.empty_like(expert.bias, device="meta"), requires_grad=expert.bias.requires_grad)

    source = _build_lazy_turtle_from_checkpoint_tensors(
        tmp_path,
        _build_rect_fused_expert_checkpoint_tensors(source_model, include_down_proj=False),
    )

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=source)

    expected = source_model.block.experts.get_submodule("1").gate_proj
    torch.testing.assert_close(expert.weight, expected.weight)
    torch.testing.assert_close(expert.bias, expected.bias)


def test_alias_all_from_turtle_materializes_leaf_transposed_expert_gate_proj_from_fused_checkpoint_tensor(tmp_path):
    """Direct-meta sync should resolve fused transposed expert tensors for leaf Linear modules."""

    source_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=True)
    shell_model = _RectFusedExpertsWrapper(width=8, intermediate=6, expert_count=2, is_transposed=True)
    shell_model.load_state_dict(source_model.state_dict())

    expert = shell_model.block.experts.get_submodule("1").gate_proj
    expert.weight = nn.Parameter(torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad)
    expert.bias = nn.Parameter(torch.empty_like(expert.bias, device="meta"), requires_grad=expert.bias.requires_grad)

    source = _build_lazy_turtle_from_checkpoint_tensors(
        tmp_path,
        _build_rect_fused_expert_checkpoint_tensors(source_model, include_down_proj=False),
    )

    alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=source)

    expected = source_model.block.experts.get_submodule("1").gate_proj
    torch.testing.assert_close(expert.weight, expected.weight)
    torch.testing.assert_close(expert.bias, expected.bias)


def test_lazy_turtle_raises_when_submodule_materialization_cannot_match_target_shape(tmp_path):
    """Shape-derived transform failures should fail materialization immediately."""

    source_model = _HybridWrapper(width=16)
    shell_model = _HybridWrapper(width=16)
    shell_model.load_state_dict(source_model.state_dict())

    shell_model.block.dt_bias = nn.Parameter(torch.empty(8, device="meta"), requires_grad=source_model.block.dt_bias.requires_grad)
    source = _build_lazy_turtle_from_module(tmp_path, source_model)

    with pytest.raises(
        RuntimeError,
        match=r"submodule materialization param `dt_bias`.*could not be reshaped into the target layout.*target_shape=\(8,\)",
    ):
        alias_from_turtle_for_submodule(
            target_model=shell_model,
            turtle_model=source,
            target_submodule=shell_model.block,
            device=torch.device("cpu"),
        )


def test_alias_all_from_turtle_raises_when_direct_meta_shape_mismatch_slips_past_transform(monkeypatch, tmp_path):
    """Post-transform shape mismatches in direct-meta sync should fail immediately."""

    source_model = _HybridWrapper(width=16)
    shell_model = _HybridWrapper(width=16)
    shell_model.load_state_dict(source_model.state_dict())

    shell_model.block.dt_bias = nn.Parameter(
        torch.empty_like(shell_model.block.dt_bias, device="meta"),
        requires_grad=source_model.block.dt_bias.requires_grad,
    )
    source = _build_lazy_turtle_from_module(tmp_path, source_model)

    original_transform = LazyTurtle._transform_checkpoint_tensor

    def _return_wrong_shape(tensor: torch.Tensor, **kwargs) -> torch.Tensor | None:
        transformed = original_transform(tensor, **kwargs)
        if transformed is None:
            return None
        return transformed[:-1].contiguous()

    # `_transform_checkpoint_tensor()` now guards most shape mismatches up front.
    # Monkeypatch it here so the regression test still exercises the downstream
    # hard-failure branch that protects against malformed custom transforms.
    monkeypatch.setattr(LazyTurtle, "_transform_checkpoint_tensor", staticmethod(_return_wrong_shape))

    with pytest.raises(
        RuntimeError,
        match=r"direct-meta sync param `dt_bias`.*shape does not match the transformed checkpoint tensor.*source_shape=\(15,\).*target_shape=\(16,\)",
    ):
        alias_all_from_turtle_if_meta(shell_model=shell_model, turtle_model=source)


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

    source = LazyTurtle.maybe_create(
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
