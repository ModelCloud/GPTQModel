# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import json
import os
import stat
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import disk_offload
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

import gptqmodel.utils.model as model_utils
import gptqmodel.utils.structure as structure_module
from gptqmodel.utils.logger import render_table
from gptqmodel.utils.model import TensorSource, get_state_dict_for_save, move_to, streaming_state_dict_to_shards
from gptqmodel.utils.offload import (
    offload_to_disk,
    offload_to_safetensors_reference,
    undo_offload_to_disk,
)


LazyTurtle = structure_module.LazyTurtle
alias_all_from_turtle_if_meta = structure_module.alias_all_from_turtle_if_meta
alias_from_turtle_for_submodule = structure_module.alias_from_turtle_for_submodule


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


class _FlatLanguageCheckpointModel(nn.Module):
    """Checkpoint layout that stores the language model directly under `model.*`."""

    def __init__(self, vocab_size: int, width: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, width)
        self.layers = nn.ModuleList([nn.Linear(width, width, bias=False)])
        self.norm = nn.LayerNorm(width)


class _FlatLanguageCheckpointWrapper(nn.Module):
    def __init__(self, vocab_size: int, width: int):
        super().__init__()
        self.model = _FlatLanguageCheckpointModel(vocab_size=vocab_size, width=width)


class _NestedLanguageShellWrapper(nn.Module):
    """Shell layout that nests the language model under `model.language_model.*`."""

    def __init__(self, vocab_size: int, width: int):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _FlatLanguageCheckpointModel(vocab_size=vocab_size, width=width)


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
        self.block.rotary = LlamaRotaryEmbedding(config)


def _tiny_deepseek_v4_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        n_routed_experts=3,
        num_experts_per_tok=2,
        hc_mult=2,
        o_groups=2,
        o_lora_rank=8,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3,
        mlp_layer_types=["moe"] * 3,
        dspark_target_layer_ids=[0, 1, 2],
        dspark_markov_rank=4,
        dspark_block_size=5,
        dspark_noise_token_id=31,
        partial_rotary_factor=0.5,
        dtype="bfloat16",
    )


class _DeepseekV4RotaryWrapper(nn.Module):
    """Pair checkpoint state with V4's constructor-only compressor RoPE buffers."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.block = nn.Module()
        self.block.linear = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.block.rotary = DeepseekV4RotaryEmbedding(config)


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


class _DirectParameterAndInitBuffer(nn.Module):
    """Keep a checkpoint parameter beside a constructor-only buffer."""

    def __init__(self, width: int, device=None):
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(torch.ones(width, device=device))
        self.register_buffer("cache", torch.arange(width, device=device), persistent=False)


class _DirectParameterAndInitBufferWrapper(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.block = _DirectParameterAndInitBuffer(width)


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


def _build_lazy_turtle_from_checkpoint_tensors(
    tmp_path: Path, checkpoint_tensors: dict[str, torch.Tensor]
) -> LazyTurtle:
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
                        expert.gate_proj.weight.detach().clone().transpose(0, 1)
                        if is_transposed
                        else expert.gate_proj.weight.detach().clone(),
                        expert.up_proj.weight.detach().clone().transpose(0, 1)
                        if is_transposed
                        else expert.up_proj.weight.detach().clone(),
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
                expert.down_proj.weight.detach().clone().transpose(0, 1)
                if is_transposed
                else expert.down_proj.weight.detach().clone()
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
    print(render_table(rows, headers=["file", "bytes"], tablefmt="github"))

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


def test_reoffload_skips_buffer_only_meta_nf12_module(tmp_path):
    from gptqmodel.nn_modules.qlinear.nf12_ad import NF12ADLinear
    from gptqmodel.utils.offload import is_meta_module

    model = nn.Module()
    model.block = NF12ADLinear(bits=12, in_features=64, out_features=16)
    offload_root = tmp_path / "offload"
    offload_to_disk(model.block, model, disk_path=str(offload_root), force=True)
    index_path = offload_root / "block" / "index.json"
    original_index = index_path.read_bytes()
    assert model.block.nf12_packed.is_meta

    offload_to_disk(model.block, model, disk_path=str(offload_root), force=True)
    assert is_meta_module(model.block)
    assert index_path.read_bytes() == original_index
    assert not (offload_root / "block.tmp").exists()


def _offloaded_save_model(
    tmp_path: Path, offload_format: str, dtype: torch.dtype = torch.float32
) -> tuple[nn.Module, Path, dict[str, torch.Tensor]]:
    model = nn.Module()
    model.block = nn.Module()
    tensor = torch.arange(24, dtype=dtype).reshape(4, 6) / 8
    model.block.weight = nn.Parameter(tensor.clone())
    model.block.register_buffer("scale", tensor.clone())
    original_state = _clone_state_dict(model)
    offload_root = tmp_path / "offload_root"

    if offload_format == "safetensors":
        offload_to_disk(module=model.block, model=model, disk_path=str(offload_root), force=True)
    else:
        module_dir = offload_root / "block"
        disk_offload(
            model.block,
            offload_dir=str(module_dir),
            offload_buffers=True,
            execution_device=torch.device("cpu"),
        )
        if offload_format == "dat-filename":
            index_path = module_dir / "index.json"
            index = json.loads(index_path.read_text())
            for name, entry in index.items():
                entry["filename"] = f"{name}.dat"
            index_path.write_text(json.dumps(index))

    return model, offload_root, original_state


@pytest.mark.parametrize("offload_format", ["safetensors", "dat", "dat-filename"])
@pytest.mark.parametrize("leaf", ["weight", "scale"])
@pytest.mark.parametrize(
    "stored_dtype, shell_dtype",
    [(torch.float32, torch.float16), (torch.float16, torch.bfloat16)],
)
def test_offload_save_rejects_dtype_mismatch(
    tmp_path: Path,
    offload_format: str,
    leaf: str,
    stored_dtype: torch.dtype,
    shell_dtype: torch.dtype,
) -> None:
    model, offload_root, _ = _offloaded_save_model(tmp_path, offload_format, dtype=stored_dtype)
    tensor = getattr(model.block, leaf).to(shell_dtype)
    if leaf == "weight":
        tensor = nn.Parameter(tensor)
    setattr(model.block, leaf, tensor)

    with pytest.raises(ValueError, match="Offload metadata mismatch") as exc:
        get_state_dict_for_save(model, offload_root=str(offload_root))

    message = str(exc.value)
    assert f"block.{leaf}" in message
    assert str(offload_root / "block" / "index.json") in message
    assert f"expected dtype {shell_dtype}" in message
    assert f"found dtype {stored_dtype}" in message


@pytest.mark.parametrize("offload_format", ["safetensors", "dat", "dat-filename"])
@pytest.mark.parametrize("leaf", ["weight", "scale"])
def test_offload_save_rejects_shape_mismatch(tmp_path: Path, offload_format: str, leaf: str) -> None:
    model, offload_root, _ = _offloaded_save_model(tmp_path, offload_format)
    tensor = getattr(model.block, leaf).reshape(6, 4)
    if leaf == "weight":
        tensor = nn.Parameter(tensor)
    setattr(model.block, leaf, tensor)

    with pytest.raises(ValueError, match="Offload metadata mismatch") as exc:
        get_state_dict_for_save(model, offload_root=str(offload_root))

    message = str(exc.value)
    assert f"block.{leaf}" in message
    assert str(offload_root / "block" / "index.json") in message
    assert "shape (6, 4)" in message
    assert "shape (4, 6)" in message


@pytest.mark.parametrize("offload_format", ["safetensors", "dat", "dat-filename"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_offload_save_preserves_matching_metadata(tmp_path: Path, offload_format: str, dtype: torch.dtype) -> None:
    model, offload_root, original_state = _offloaded_save_model(tmp_path, offload_format, dtype=dtype)
    model.output = nn.Module()
    model.output.weight = model.block.weight

    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
    assert set(state_dict) == set(original_state)
    assert model.output.weight is model.block.weight
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

    with safe_open(str(save_dir / expected_files[0]), framework="pt") as handler:
        for name, tensor in original_state.items():
            torch.testing.assert_close(handler.get_tensor(name), tensor)



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


def test_lazy_turtle_hf_conversion_map_reversed_resolves_nested_language_model_prefix(tmp_path):
    source_model = _FlatLanguageCheckpointWrapper(vocab_size=32, width=16)
    shell_model = _NestedLanguageShellWrapper(vocab_size=32, width=16)

    shell_model.model.language_model.embed_tokens.weight = nn.Parameter(
        torch.empty_like(shell_model.model.language_model.embed_tokens.weight, device="meta"),
        requires_grad=shell_model.model.language_model.embed_tokens.weight.requires_grad,
    )
    shell_model.model.language_model.layers[0].weight = nn.Parameter(
        torch.empty_like(shell_model.model.language_model.layers[0].weight, device="meta"),
        requires_grad=shell_model.model.language_model.layers[0].weight.requires_grad,
    )

    checkpoint_tensors = {name: tensor.detach().clone() for name, tensor in source_model.state_dict().items()}
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    save_file(checkpoint_tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, checkpoint_tensors)

    source = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
        hf_conversion_map_reversed=(
            SimpleNamespace(source_patterns=["model.language_model"], target_patterns=["model"]),
            SimpleNamespace(source_patterns=["language_model"], target_patterns=["model"]),
        ),
    )
    assert source is not None

    embed_tensors = source.checkpoint_tensors_for_submodule(
        target_model=shell_model,
        target_submodule=shell_model.model.language_model.embed_tokens,
        recurse=True,
    )
    layer_tensors = source.checkpoint_tensors_for_submodule(
        target_model=shell_model,
        target_submodule=shell_model.model.language_model.layers[0],
        recurse=True,
    )

    assert tuple(embed_tensors) == ("weight",)
    assert tuple(layer_tensors) == ("weight",)

    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=source,
        target_submodule=shell_model.model.language_model.embed_tokens,
        device=torch.device("cpu"),
    )
    alias_from_turtle_for_submodule(
        target_model=shell_model,
        turtle_model=source,
        target_submodule=shell_model.model.language_model.layers[0],
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        shell_model.model.language_model.embed_tokens.weight,
        source_model.model.embed_tokens.weight,
    )
    torch.testing.assert_close(
        shell_model.model.language_model.layers[0].weight,
        source_model.model.layers[0].weight,
    )


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


def test_lazy_turtle_direct_meta_restores_nonpersistent_buffers_from_module_init(tmp_path):
    """Layer-forward direct state must rebuild constructor-only rotary buffers."""

    config = _tiny_llama_config()
    source_model = _RotaryWrapper(config)
    shell_model = _RotaryWrapper(config)
    shell_model.load_state_dict(source_model.state_dict())

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
    source.materialize_direct_meta_tensors(
        target_model=shell_model,
        target_submodule=shell_model.block.rotary,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        shell_model.block.rotary.inv_freq,
        source_model.block.rotary.inv_freq,
    )
    torch.testing.assert_close(
        shell_model.block.rotary.original_inv_freq,
        source_model.block.rotary.original_inv_freq,
    )
    assert shell_model.block.rotary.inv_freq.device.type == "cpu"
    assert shell_model.block.rotary._non_persistent_buffers_set == {
        "inv_freq",
        "original_inv_freq",
    }


def test_lazy_turtle_direct_meta_defaults_device_for_nonpersistent_buffers(tmp_path):
    """``device=None`` must resolve to the shell buffer's device, not crash.

    The public helper defaults to ``device=None``; passing it straight to
    ``safe_move_to`` raises ``TypeError`` from ``torch.device(None)``.
    """

    config = _tiny_llama_config()
    source_model = _RotaryWrapper(config)
    shell_model = _RotaryWrapper(config)
    shell_model.load_state_dict(source_model.state_dict())

    for name in ("inv_freq", "original_inv_freq"):
        shell_model.block.rotary.register_buffer(
            name,
            torch.empty_like(getattr(shell_model.block.rotary, name), device="meta"),
            persistent=False,
        )

    source = _build_lazy_turtle_from_module(tmp_path, source_model)
    source.materialize_direct_meta_tensors(
        target_model=shell_model,
        target_submodule=shell_model.block.rotary,
        device=None,
    )

    for name in ("inv_freq", "original_inv_freq"):
        restored = getattr(shell_model.block.rotary, name)
        assert not restored.is_meta
        torch.testing.assert_close(restored, getattr(source_model.block.rotary, name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for a real device mismatch")
@pytest.mark.parametrize("buffer_device", ["meta", "cpu"])
def test_lazy_turtle_direct_meta_defaults_to_materialized_parameter_device(tmp_path, buffer_device):
    source_model = _DirectParameterAndInitBufferWrapper(width=4)
    shell_model = _DirectParameterAndInitBufferWrapper(width=4).to("cuda:0")
    shell_model.block.register_buffer(
        "cache", torch.arange(4, dtype=torch.int64, device=buffer_device), persistent=False
    )
    source = _build_lazy_turtle_from_module(tmp_path, source_model)

    source.materialize_direct_meta_tensors(
        target_model=shell_model,
        target_submodule=shell_model.block,
    )

    assert shell_model.block.weight.device.type == "cuda"
    assert shell_model.block.cache.device == shell_model.block.weight.device
    torch.testing.assert_close(shell_model.block.cache.cpu(), source_model.block.cache)


def test_lazy_turtle_direct_meta_restores_deepseek_v4_rotary_buffers(tmp_path):
    """The layer-2 compressor fast path must rebuild all four V4 RoPE buffers."""

    config = _tiny_deepseek_v4_config()
    source_model = _DeepseekV4RotaryWrapper(config)
    shell_model = _DeepseekV4RotaryWrapper(config)
    shell_model.load_state_dict(source_model.state_dict())

    expected_names = {
        "main_inv_freq",
        "main_original_inv_freq",
        "compress_inv_freq",
        "compress_original_inv_freq",
    }
    assert shell_model.block.rotary._non_persistent_buffers_set == expected_names
    assert not shell_model.block.rotary.state_dict()

    for name, buffer in dict(shell_model.block.rotary.named_buffers(recurse=False)).items():
        shell_model.block.rotary.register_buffer(
            name,
            torch.empty_like(buffer, device="meta"),
            persistent=False,
        )

    source = _build_lazy_turtle_from_module(tmp_path, source_model)
    source.materialize_direct_meta_tensors(
        target_model=shell_model,
        target_submodule=shell_model.block.rotary,
        device=torch.device("cpu"),
    )

    restored = dict(shell_model.block.rotary.named_buffers(recurse=False))
    reference = dict(source_model.block.rotary.named_buffers(recurse=False))
    assert set(restored) == expected_names
    assert shell_model.block.rotary._non_persistent_buffers_set == expected_names
    assert not shell_model.block.rotary.state_dict()
    for name in expected_names:
        torch.testing.assert_close(restored[name], reference[name])
        assert restored[name].device.type == "cpu"


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


def test_lazy_turtle_materializes_split_gate_up_when_aliases_resolve_to_fused_checkpoint_tensor(tmp_path):
    """Alias-based gate/up resolution must still preserve the fused split metadata."""

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
        hf_conversion_map_reversed=(
            SimpleNamespace(source_patterns=["block.gate_proj.weight"], target_patterns=["block.gate_up_proj.weight"]),
            SimpleNamespace(source_patterns=["block.up_proj.weight"], target_patterns=["block.gate_up_proj.weight"]),
        ),
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
    expert.weight = nn.Parameter(
        torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad
    )
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
                        source_model.block.experts.get_submodule(str(expert_idx))
                        .gate_proj.weight.detach()
                        .clone()
                        .transpose(0, 1),
                        source_model.block.experts.get_submodule(str(expert_idx))
                        .up_proj.weight.detach()
                        .clone()
                        .transpose(0, 1),
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
                source_model.block.experts.get_submodule(str(expert_idx))
                .down_proj.weight.detach()
                .clone()
                .transpose(0, 1)
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
                        source_model.block.experts.get_submodule(str(expert_idx))
                        .gate_proj.weight.detach()
                        .clone()
                        .transpose(0, 1),
                        source_model.block.experts.get_submodule(str(expert_idx))
                        .up_proj.weight.detach()
                        .clone()
                        .transpose(0, 1),
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
    expert.weight = nn.Parameter(
        torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad
    )
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
    expert.weight = nn.Parameter(
        torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad
    )
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
    expert.weight = nn.Parameter(
        torch.empty_like(expert.weight, device="meta"), requires_grad=expert.weight.requires_grad
    )
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

    shell_model.block.dt_bias = nn.Parameter(
        torch.empty(8, device="meta"), requires_grad=source_model.block.dt_bias.requires_grad
    )
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


def test_sync_all_meta_reports_logbar_progress(monkeypatch):
    class _FakeProgress:
        def __init__(self):
            self.current_iter_step = 0
            self.titles = []
            self.subtitles = []
            self.draw_calls = []
            self.closed = False

        def manual(self):
            return self

        def set(self, **_kwargs):
            return self

        def title(self, value):
            self.titles.append(value)
            return self

        def subtitle(self, value):
            self.subtitles.append(value)
            return self

        def draw(self, force: bool = False):
            self.draw_calls.append((self.current_iter_step, force))
            return self

        def close(self):
            self.closed = True

    class _FakeLogger:
        def __init__(self):
            self.progress = _FakeProgress()
            self.iterable = None

        def pb(self, iterable, *, output_interval=None):
            del output_interval
            self.iterable = list(iterable)
            return self.progress

        def info(self, *_args, **_kwargs):
            return None

    shell_model = _HybridWrapper(width=8)
    turtle = LazyTurtle.__new__(LazyTurtle)
    turtle._lock = threading.RLock()

    fake_logger = _FakeLogger()
    visited_modules = []

    def _fake_materialize(self, *, module_path, **_kwargs):
        del self, _kwargs
        visited_modules.append(module_path)
        return 1

    monkeypatch.setattr(structure_module, "log", fake_logger)
    monkeypatch.setattr(LazyTurtle, "_materialize_direct_meta_tensors", _fake_materialize)

    materialized = turtle.sync_all_meta(shell_model=shell_model, tie_after=False)

    assert materialized == 3
    assert visited_modules == ["", "block", "block.inner"]
    assert fake_logger.iterable == [0, 1, 2]
    assert fake_logger.progress.titles == ["Syncing lazy checkpoint writes (3 modules)"]
    assert fake_logger.progress.subtitles[0] == "Preparing module writes"
    assert "Writing <root> (1/3)" in fake_logger.progress.subtitles
    assert "Writing block (2/3)" in fake_logger.progress.subtitles
    assert "Writing block.inner (3/3)" in fake_logger.progress.subtitles
    assert fake_logger.progress.subtitles[-1] == "Wrote block.inner (+1, total 3)"
    assert fake_logger.progress.draw_calls[0] == (0, True)
    assert fake_logger.progress.draw_calls[-1] == (3, False)
    assert fake_logger.progress.closed is True


def test_sync_all_meta_throttles_progress_for_large_moe_shell(monkeypatch):
    class _FakeProgress:
        def __init__(self):
            self.current_iter_step = 0
            self.draw_calls = []

        def manual(self):
            return self

        def set(self, **_kwargs):
            return self

        def title(self, _value):
            return self

        def subtitle(self, _value):
            return self

        def draw(self, force: bool = False):
            self.draw_calls.append((self.current_iter_step, force))
            return self

        def close(self):
            return None

    class _FakeLogger:
        def __init__(self):
            self.progress = _FakeProgress()

        def pb(self, _iterable, *, output_interval=None):
            del output_interval
            return self.progress

        def info(self, *_args, **_kwargs):
            return None

    shell_model = nn.Module()
    shell_model.experts = nn.ModuleList(nn.Identity() for _ in range(1000))
    turtle = LazyTurtle.__new__(LazyTurtle)
    turtle._lock = threading.RLock()
    visited_modules = []

    def _fake_materialize(self, *, module_path, **_kwargs):
        del self, _kwargs
        visited_modules.append(module_path)
        return 0

    fake_logger = _FakeLogger()
    monkeypatch.setattr(structure_module, "log", fake_logger)
    monkeypatch.setattr(
        LazyTurtle, "_materialize_direct_meta_tensors", _fake_materialize
    )

    materialized = turtle.sync_all_meta(
        shell_model=shell_model, tie_after=False
    )

    assert materialized == 0
    assert len(visited_modules) == 1002
    # Initial forced draw plus at most ~201 before/after checkpoints.
    assert len(fake_logger.progress.draw_calls) <= 405
    assert fake_logger.progress.draw_calls[0] == (0, True)
    assert fake_logger.progress.draw_calls[-1][0] == 1002


def test_copy_checkpoint_tensors_into_submodule_reports_load_progress(tmp_path, monkeypatch):
    class _FakeProgress:
        def __init__(self):
            self.current_iter_step = 0
            self.titles = []
            self.subtitles = []
            self.draw_calls = []
            self.closed = False

        def manual(self):
            return self

        def set(self, **_kwargs):
            return self

        def title(self, value):
            self.titles.append(value)
            return self

        def subtitle(self, value):
            self.subtitles.append(value)
            return self

        def draw(self, force: bool = False):
            self.draw_calls.append((self.current_iter_step, force))
            return self

        def close(self):
            self.closed = True

    class _FakeLogger:
        def __init__(self):
            self.progress = _FakeProgress()
            self.iterable = None

        def pb(self, iterable, *, output_interval=None):
            del output_interval
            self.iterable = list(iterable)
            return self.progress

    source_model = _HybridWrapper(width=8)
    shell_model = _HybridWrapper(width=8)
    source = _build_lazy_turtle_from_module(tmp_path, source_model)
    fake_logger = _FakeLogger()

    monkeypatch.setattr(structure_module, "log", fake_logger)

    source._copy_checkpoint_tensors_into_submodule(
        target_model=shell_model,
        target_submodule=shell_model.block,
        module_path="block",
        device=torch.device("cpu"),
        recurse=True,
        non_blocking=False,
    )

    assert fake_logger.iterable == [0, 1, 2]
    assert fake_logger.progress.titles == ["Loading checkpoint tensors (3)"]
    assert fake_logger.progress.subtitles[0] == "block: 0/3"
    assert "dt_bias: 1/3" in fake_logger.progress.subtitles
    assert "inner.weight: 2/3" in fake_logger.progress.subtitles
    assert "dt_scale: 3/3" in fake_logger.progress.subtitles
    assert fake_logger.progress.draw_calls[0] == (0, True)
    assert fake_logger.progress.draw_calls[-1] == (3, False)
    assert fake_logger.progress.closed is True


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


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"])
def test_streaming_state_dict_to_shards_writes_float8_tensors(tmp_path, dtype_name):
    if not hasattr(torch, dtype_name):
        pytest.skip(f"{dtype_name} dtype not available")
    dtype = getattr(torch, dtype_name)
    tensor = torch.randn(8, 16, dtype=torch.float32).to(dtype)
    state_dict = {
        "fp8_weight": TensorSource(
            name="fp8_weight",
            torch_dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            source=tensor,
        ),
    }
    expected_files, _, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(tmp_path),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )
    shard_path = tmp_path / expected_files[0]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        saved = handler.get_tensor("fp8_weight")

    assert saved.dtype is dtype
    assert torch.equal(saved.view(torch.uint8), tensor.view(torch.uint8))


@pytest.mark.skipif(not hasattr(torch, "float8_e8m0fnu"), reason="float8_e8m0fnu dtype not available")
def test_streaming_state_dict_to_shards_writes_float8_e8m0_tensors(tmp_path):
    tensor = torch.tensor([126, 127, 128], dtype=torch.uint8).view(torch.float8_e8m0fnu)
    state_dict = {
        "fp8_scale": TensorSource(
            name="fp8_scale",
            torch_dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            source=tensor,
        ),
    }
    expected_files, _, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(tmp_path),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )
    shard_path = tmp_path / expected_files[0]
    with safe_open(str(shard_path), framework="pt", device="cpu") as handler:
        saved = handler.get_tensor("fp8_scale")

    assert saved.dtype is torch.float8_e8m0fnu
    assert torch.equal(saved.view(torch.uint8), tensor.view(torch.uint8))


def test_streaming_state_dict_to_shards_reuses_only_authenticated_shards(tmp_path, monkeypatch):
    tensor = torch.arange(32, dtype=torch.int16).reshape(4, 8)
    state_dict = {
        "weight": TensorSource(
            name="weight",
            torch_dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            source=tensor,
        ),
    }
    arguments = {
        "state_dict": state_dict,
        "save_dir": str(tmp_path),
        "model_base_name": "model",
        "single_file_name": "model.safetensors",
        "metadata": {"format": "pt"},
        "max_shard_size": None,
    }
    expected_files, _, _ = streaming_state_dict_to_shards(**arguments)
    shard_path = tmp_path / expected_files[0]
    original = shard_path.read_bytes()
    original_writer = model_utils._write_shard_file

    def unexpected_write(*args, **kwargs):
        raise AssertionError("authenticated shard should have been reused")

    monkeypatch.setattr(model_utils, "_write_shard_file", unexpected_write)
    assert streaming_state_dict_to_shards(**arguments)[0] == expected_files

    with shard_path.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(bytes([original[-1] ^ 0xFF]))

    writes = 0

    def repair(path, entries, metadata):
        nonlocal writes
        writes += 1
        return original_writer(path, entries, metadata)

    monkeypatch.setattr(model_utils, "_write_shard_file", repair)
    streaming_state_dict_to_shards(**arguments)
    assert writes == 1
    assert shard_path.read_bytes() == original


def test_streaming_state_dict_to_shards_rewrites_when_source_changes(tmp_path):
    tensor = torch.arange(32, dtype=torch.int16).reshape(4, 8)
    arguments = {
        "state_dict": {
            "weight": TensorSource(
                name="weight",
                torch_dtype=tensor.dtype,
                shape=tuple(tensor.shape),
                source=tensor,
            ),
        },
        "save_dir": str(tmp_path),
        "model_base_name": "model",
        "single_file_name": "model.safetensors",
        "metadata": {"format": "pt"},
        "max_shard_size": None,
    }
    expected_files, _, _ = streaming_state_dict_to_shards(**arguments)
    shard_path = tmp_path / expected_files[0]
    original = shard_path.read_bytes()

    changed = tensor.clone()
    changed[0, 0] = 999
    arguments["state_dict"] = {
        "weight": TensorSource(
            name="weight",
            torch_dtype=changed.dtype,
            shape=tuple(changed.shape),
            source=changed,
        ),
    }
    streaming_state_dict_to_shards(**arguments)

    assert shard_path.read_bytes() != original


def test_streaming_state_dict_to_shards_reuses_legacy_unhashed_offload(tmp_path, monkeypatch):
    model = _LinearWithBuffers(in_features=128, out_features=96)
    offload_root = tmp_path / "offload_root"
    offload_to_disk(module=model.linear, model=model, disk_path=str(offload_root))
    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
    offload_refs = [
        entry.source
        for entry in state_dict.values()
        if isinstance(entry.source, model_utils.OffloadTensorRef)
    ]
    assert offload_refs
    assert all(source.sha256 is None for source in offload_refs)

    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    arguments = {
        "state_dict": state_dict,
        "save_dir": str(save_dir),
        "model_base_name": "model",
        "single_file_name": "model.safetensors",
        "metadata": {"format": "pt"},
        "max_shard_size": None,
    }
    expected_files, _, _ = streaming_state_dict_to_shards(**arguments)

    def unexpected_write(*args, **kwargs):
        raise AssertionError("legacy offload source should be compared by content")

    monkeypatch.setattr(model_utils, "_write_shard_file", unexpected_write)
    assert streaming_state_dict_to_shards(**arguments)[0] == expected_files


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


def test_safetensors_reference_offload_avoids_a_second_payload(tmp_path):
    model = _LinearWithBuffers(in_features=128, out_features=96)
    original_state = _clone_state_dict(model.linear)
    source_path = tmp_path / "checkpoint.safetensors"
    save_file(original_state, source_path)
    offload_root = tmp_path / "offload_root"

    index = offload_to_safetensors_reference(
        module=model.linear,
        model=model,
        source_path=source_path,
        disk_path=str(offload_root),
    )

    module_dir = offload_root / "linear"
    assert sorted(path.name for path in module_dir.iterdir()) == ["index.json"]
    assert all(tensor.device.type == "meta" for tensor in model.linear.state_dict().values())
    assert set(index) == set(original_state)
    assert all(entry["safetensors_file"] == str(source_path) for entry in index.values())

    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
    expected_files, _tensor_to_filename, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )
    with safe_open(str(save_dir / expected_files[0]), framework="pt", device="cpu") as handler:
        for name, tensor in original_state.items():
            torch.testing.assert_close(handler.get_tensor(f"linear.{name}"), tensor)

    # The index binds the source payload, not merely its header and geometry.
    save_file(
        {name: torch.zeros_like(tensor) for name, tensor in original_state.items()},
        source_path,
    )
    rejected_dir = tmp_path / "rejected"
    rejected_dir.mkdir()
    with pytest.raises(IOError, match="content digest differs"):
        streaming_state_dict_to_shards(
            state_dict,
            save_dir=str(rejected_dir),
            model_base_name="model",
            single_file_name="model.safetensors",
            metadata={},
            max_shard_size=None,
        )


@pytest.mark.parametrize("offload_kind", ["reference", "bundle"])
def test_nested_offload_index_keeps_distinct_same_named_weights(tmp_path, offload_kind):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.full((2, 2), 1.0))
            self.child = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.child.weight.fill_(2.0)

    model = nn.Module()
    model._tied_weights_keys = None
    model.block = Block()
    expected = _clone_state_dict(model.block)
    offload_root = tmp_path / "offload"
    if offload_kind == "reference":
        source = tmp_path / "source.safetensors"
        save_file(expected, source)
        offload_to_safetensors_reference(
            model.block, model, source, disk_path=str(offload_root)
        )
    else:
        offload_to_disk(model.block, model, disk_path=str(offload_root), force=True)

    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
    assert set(state_dict) == {"block.weight", "block.child.weight"}
    assert state_dict["block.child.weight"].source.weight_name == "child.weight"

    saved = tmp_path / "saved"
    saved.mkdir()
    streaming_state_dict_to_shards(
        state_dict, str(saved), "model", "model.safetensors", {}, None
    )
    with safe_open(str(saved / "model.safetensors"), framework="pt", device="cpu") as handle:
        for name, tensor in expected.items():
            torch.testing.assert_close(handle.get_tensor(f"block.{name}"), tensor)


@pytest.mark.parametrize("index_case", ["missing", "full_name", "conflict"])
def test_nested_offload_index_uses_scoped_key_and_rejects_conflicts(tmp_path, index_case):
    model = nn.Module()
    model._tied_weights_keys = None
    model.block = nn.Module()
    model.block.weight = nn.Parameter(torch.ones(2, 2))
    model.block.child = nn.Linear(2, 2, bias=False)
    source = tmp_path / "source.safetensors"
    save_file(_clone_state_dict(model.block), source)
    offload_root = tmp_path / "offload"
    offload_to_safetensors_reference(model.block, model, source, disk_path=str(offload_root))
    index_path = offload_root / "block" / "index.json"
    index = json.loads(index_path.read_text())
    child = index.pop("child.weight")
    if index_case == "full_name":
        index["block.child.weight"] = child
    elif index_case == "conflict":
        index["child.weight"] = child
        index["block.child.weight"] = index["weight"]
    index_path.write_text(json.dumps(index))

    if index_case == "missing":
        with pytest.raises(FileNotFoundError, match="block.child.weight"):
            get_state_dict_for_save(model, offload_root=str(offload_root))
    elif index_case == "conflict":
        with pytest.raises(ValueError, match="Ambiguous offload index entries"):
            get_state_dict_for_save(model, offload_root=str(offload_root))
    else:
        state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
        assert set(state_dict) == {"block.weight", "block.child.weight"}
        assert state_dict["block.child.weight"].source.weight_name == "child.weight"


def test_reference_source_inside_export_dir_is_rejected_before_any_shard_write(tmp_path):
    model = nn.Module()
    model.linear = nn.Linear(4, 4)
    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    # A 4x4 fp32 weight exceeds the shard limit, while the bias fits. The
    # first output would otherwise replace this source before reading bias.
    source_path = save_dir / "model-00001-of-00002.safetensors"
    original = _clone_state_dict(model.linear)
    save_file(original, source_path)
    source_bytes = source_path.read_bytes()
    offload_root = tmp_path / "offload"
    offload_to_safetensors_reference(
        model.linear, model, source_path, disk_path=str(offload_root)
    )
    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))

    with pytest.raises(ValueError, match="inside the export directory"):
        streaming_state_dict_to_shards(
            state_dict, str(save_dir), "model", "model.safetensors", {}, 40
        )

    assert source_path.read_bytes() == source_bytes
    assert sorted(path.name for path in save_dir.iterdir()) == [source_path.name]

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    names, _, _ = streaming_state_dict_to_shards(
        state_dict, str(other_dir), "model", "model.safetensors", {}, 40
    )
    assert len(names) == 2
    exported = {}
    for name in names:
        with safe_open(str(other_dir / name), framework="pt", device="cpu") as handle:
            exported.update({key: handle.get_tensor(key) for key in handle.keys()})
    for name, tensor in original.items():
        torch.testing.assert_close(exported[f"linear.{name}"], tensor)


@pytest.mark.skipif(os.name != "posix", reason="requires file symlinks")
@pytest.mark.parametrize("link_inside", [True, False])
def test_reference_export_preflight_checks_symlink_path_and_target(tmp_path, link_inside):
    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    actual_dir, link_dir = (outside, save_dir) if link_inside else (save_dir, outside)
    actual = actual_dir / "model.safetensors"
    actual.write_bytes(b"original source")
    link = link_dir / "source.safetensors"
    link.symlink_to(actual)
    entry = TensorSource(
        name="weight", torch_dtype=torch.float32, shape=(1,),
        source=model_utils.OffloadTensorRef(
            path=str(link), torch_dtype=torch.float32, shape=(1,), format="dat"
        ),
    )

    with pytest.raises(ValueError, match="inside the export directory"):
        streaming_state_dict_to_shards(
            {"weight": entry}, str(save_dir), "model", "model.safetensors", {}, None
        )

    assert actual.read_bytes() == b"original source"
    assert link.is_symlink()


@pytest.mark.parametrize("dtype_name", ["float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"])
def test_safetensors_reference_offload_supports_float8(tmp_path, dtype_name):
    if not hasattr(torch, dtype_name):
        pytest.skip(f"{dtype_name} dtype not available")
    dtype = getattr(torch, dtype_name)
    model = _LinearWithBuffers(in_features=8, out_features=4)
    weight = model.linear.weight
    model.linear.weight = nn.Parameter(
        torch.arange(weight.numel(), dtype=torch.float32)
        .to(dtype)
        .reshape(weight.shape),
        requires_grad=False,
    )
    original_state = _clone_state_dict(model.linear)
    source_path = tmp_path / "checkpoint.safetensors"
    save_file(original_state, source_path)

    index = offload_to_safetensors_reference(
        module=model.linear,
        model=model,
        source_path=source_path,
        disk_path=str(tmp_path / "offload_root"),
    )

    assert index["weight"]["dtype"] == dtype_name
    save_dir = tmp_path / "saved"
    save_dir.mkdir()
    state_dict = get_state_dict_for_save(model, offload_root=str(tmp_path / "offload_root"))
    expected_files, _tensor_to_filename, _ = streaming_state_dict_to_shards(
        state_dict,
        save_dir=str(save_dir),
        model_base_name="model",
        single_file_name="model.safetensors",
        metadata={},
        max_shard_size=None,
    )
    with safe_open(str(save_dir / expected_files[0]), framework="pt", device="cpu") as handler:
        exported = handler.get_tensor("linear.weight")
    assert exported.dtype == dtype
    assert torch.equal(exported.view(torch.uint8), original_state["weight"].view(torch.uint8))


def test_safetensors_reference_offload_rejects_non_persistent_buffers(tmp_path):
    model = _LinearWithBuffers(in_features=8, out_features=4)
    model.linear.register_buffer("cache", torch.ones(2), persistent=False)
    source_path = tmp_path / "checkpoint.safetensors"
    save_file(_clone_state_dict(model.linear), source_path)

    with pytest.raises(ValueError, match="non-persistent buffers"):
        offload_to_safetensors_reference(
            module=model.linear,
            model=model,
            source_path=source_path,
            disk_path=str(tmp_path / "offload_root"),
        )

    assert all(not tensor.is_meta for tensor in model.linear.state_dict().values())


@pytest.mark.parametrize("nested", [False, True])
def test_safetensors_reference_offload_allows_absent_non_persistent_buffers(tmp_path, nested):
    model = nn.Module()
    model._tied_weights_keys = None
    model.block = nn.Module()
    model.block.linear = nn.Linear(4, 4)
    holder = model.block.linear if nested else model.block
    holder.register_buffer("optional_cache", None, persistent=False)
    original = _clone_state_dict(model.block)
    source = tmp_path / "source.safetensors"
    save_file(original, source)
    offload_root = tmp_path / "offload"

    index = offload_to_safetensors_reference(
        model.block, model, source, disk_path=str(offload_root)
    )

    assert set(index) == set(original)
    assert holder.optional_cache is None
    assert all(tensor.is_meta for tensor in model.block.state_dict().values())
    state_dict = get_state_dict_for_save(model, offload_root=str(offload_root))
    saved = tmp_path / "saved"
    saved.mkdir()
    streaming_state_dict_to_shards(
        state_dict, str(saved), "model", "model.safetensors", {}, None
    )
    with safe_open(saved / "model.safetensors", framework="pt", device="cpu") as handle:
        for name, tensor in original.items():
            torch.testing.assert_close(handle.get_tensor(f"block.{name}"), tensor)


def test_safetensors_reference_offload_keeps_source_inside_target_directory(tmp_path):
    model = _LinearWithBuffers(in_features=8, out_features=4)
    original_state = _clone_state_dict(model.linear)
    offload_root = tmp_path / "offload_root"
    module_dir = offload_root / "linear"
    module_dir.mkdir(parents=True)
    source_path = module_dir / "source.safetensors"
    save_file(original_state, source_path)
    source_bytes = source_path.read_bytes()

    with pytest.raises(ValueError, match="source is inside the target offload directory"):
        offload_to_safetensors_reference(
            module=model.linear,
            model=model,
            source_path=source_path,
            disk_path=str(offload_root),
        )

    assert source_path.read_bytes() == source_bytes
    assert sorted(path.name for path in module_dir.iterdir()) == [source_path.name]
    for name, tensor in model.linear.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name])


def test_safetensors_reference_index_honors_process_umask(tmp_path, monkeypatch):
    model = _LinearWithBuffers(in_features=8, out_features=4)
    source_path = tmp_path / "checkpoint.safetensors"
    save_file(_clone_state_dict(model.linear), source_path)
    ordinary_file = tmp_path / "ordinary-file"
    ordinary_file.write_bytes(b"created under the process umask")

    def reject_forced_mode(*_args):
        raise AssertionError("reference index must honor the process umask")

    monkeypatch.setattr(os, "fchmod", reject_forced_mode, raising=False)
    offload_root = tmp_path / "offload_root"
    offload_to_safetensors_reference(
        module=model.linear,
        model=model,
        source_path=source_path,
        disk_path=str(offload_root),
    )

    index_mode = stat.S_IMODE((offload_root / "linear" / "index.json").stat().st_mode)
    assert index_mode == stat.S_IMODE(ordinary_file.stat().st_mode)
