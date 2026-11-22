# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Separate fused gate/up tensors must not require a gated-pair checkpoint."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from gptqmodel.utils.structure import LazyTurtle


def shell_and_checkpoint(path, layout, dtype):
    shell = nn.Module()
    shell.block = nn.Module()
    shell.block.experts = nn.ModuleList()
    shell._keep_in_fp32_modules_strict = ["expert_bias"]
    shell.block.register_buffer("expert_bias", torch.empty(3, dtype=dtype, device="meta"))
    state = {"block.expert_bias": torch.tensor([0.123456789, -1.000123, 17.00321])}
    expected = {"block.expert_bias": state["block.expert_bias"].clone()}
    for index in range(3):
        expert = nn.Module()
        for projection, shape in (("gate_proj", (6, 4)), ("up_proj", (6, 4)), ("down_proj", (4, 6))):
            expert.add_module(projection, nn.Linear(shape[1], shape[0], bias=True, device="meta", dtype=dtype))
            for suffix, size in (("weight", shape), ("bias", shape[:1])):
                value = torch.arange(torch.Size(size).numel(), dtype=torch.float32).reshape(size) / 31
                value = value + (10 if projection == "gate_proj" else 30 if projection == "up_proj" else -10) + index
                expected[f"block.experts.{index}.{projection}.{suffix}"] = value.to(dtype)
        shell.block.experts.append(expert)
    if layout == "indexed":
        state.update({name: value.float() for name, value in expected.items() if ".experts." in name})
    else:
        for suffix in ("weight", "bias"):
            packed_suffix = "" if suffix == "weight" else "_bias"
            for projection in ("gate_proj", "up_proj", "down_proj"):
                if layout == "gated" and projection != "down_proj":
                    continue
                state[f"block.experts.{projection}{packed_suffix}"] = torch.stack([
                    expected[f"block.experts.{i}.{projection}.{suffix}"].float() for i in range(3)
                ])
            if layout in {"gated", "both"}:
                # The gated-pair candidate must beat the incompatible separate
                # candidate even when both keys are present in a checkpoint.
                gate_up = torch.stack([torch.cat([
                    expected[f"block.experts.{i}.{p}.{suffix}"].float() for p in ("gate_proj", "up_proj")
                ]) for i in range(3)])
                state[f"block.experts.gate_up_proj{packed_suffix}"] = gate_up
                if layout == "both":
                    state[f"block.experts.up_proj{packed_suffix}"].fill_(-500)
                    state[f"block.experts.gate_proj{packed_suffix}"].fill_(500)
    save_file(state, path / "model.safetensors")
    turtle = LazyTurtle(model_local_path=str(path), config=SimpleNamespace(),
                        model_init_kwargs={"dtype": dtype}, target_model=shell,
                        hf_conversion_map_reversed={})
    return shell, turtle, expected


def restore(turtle, shell, mode):
    if mode == "recursive":
        turtle.materialize_submodule(target_model=shell, target_submodule=shell.block,
                                     module_path="block", device="cpu", show_progress=False)
    else:
        for name, module in shell.block.named_modules():
            turtle.materialize_submodule(target_model=shell, target_submodule=module,
                                         module_path="block" + ("." + name if name else ""),
                                         device="cpu", recurse=False, show_progress=False)


@pytest.mark.parametrize("layout", ["separate", "gated", "both", "indexed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["recursive", "leaves"])
def test_separate_gated_precedence_and_direct_indexed_exact_materialization(tmp_path, layout, dtype, mode):
    shell, turtle, expected = shell_and_checkpoint(tmp_path, layout, dtype)
    for _ in range(2):
        restore(turtle, shell, mode)
        assert shell.block.expert_bias.dtype == torch.float32
        for name, value in expected.items():
            torch.testing.assert_close(shell.state_dict()[name], value, rtol=0, atol=0, msg=name)
        x = torch.tensor([[.25, -.5, .125, .75]], dtype=dtype)
        for i, expert in enumerate(shell.block.experts):
            oracle = torch.nn.functional.linear(x, expected[f"block.experts.{i}.up_proj.weight"],
                                                expected[f"block.experts.{i}.up_proj.bias"])
            torch.testing.assert_close(expert.up_proj(x), oracle, rtol=0, atol=0)
        shell.block.to_empty(device="meta")
        shell.block.to(dtype=dtype)


def test_separate_fused_expert_missing_shard_repairs_on_same_turtle(tmp_path):
    shell, turtle, expected = shell_and_checkpoint(tmp_path, "separate", torch.bfloat16)
    shard = tmp_path / "model.safetensors"
    payload = shard.read_bytes()
    shard.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError), match="model.safetensors"):
        restore(turtle, shell, "recursive")
    shard.write_bytes(payload)
    restore(turtle, shell, "recursive")
    for name, value in expected.items():
        torch.testing.assert_close(shell.state_dict()[name], value, atol=0, rtol=0)


@pytest.mark.parametrize("family", ["nemotron_h", "qwen3_moe", "inkling_shared"])
def test_real_native_separate_and_gated_experts_retain_values_and_forward(tmp_path, family):
    from transformers import InklingTextConfig, NemotronHConfig, Qwen3MoeConfig
    from transformers.models.inkling.modeling_inkling import InklingSharedExperts
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    if family == "nemotron_h":
        native = NemotronHExperts(NemotronHConfig(hidden_size=4, moe_intermediate_size=6, n_routed_experts=3))
        projections = ("up_proj", "down_proj")
    elif family == "qwen3_moe":
        native = Qwen3MoeExperts(Qwen3MoeConfig(hidden_size=4, moe_intermediate_size=6, num_experts=3))
        projections = ("gate_proj", "up_proj", "down_proj")
    else:
        native = InklingSharedExperts(InklingTextConfig(hidden_size=4, moe_intermediate_size=6, n_shared_experts=3))
        projections = ("gate_proj", "up_proj", "down_proj")
    with torch.no_grad():
        for i, param in enumerate(native.parameters()):
            param.copy_(torch.sin(torch.arange(param.numel()).reshape(param.shape) + i) / 5)
    values = {"block.experts." + k: v.clone() for k, v in native.state_dict().items()}
    save_file(values, tmp_path / "model.safetensors")
    shell = nn.Module()
    shell.block = nn.Module()
    shell.block.experts = nn.ModuleList()
    for _ in range(3):
        expert = nn.Module()
        for name in projections:
            expert.add_module(name, nn.Linear(6 if name == "down_proj" else 4,
                                               4 if name == "down_proj" else 6, bias=False, device="meta"))
        shell.block.experts.append(expert)
    turtle = LazyTurtle(model_local_path=str(tmp_path), config=SimpleNamespace(),
                        target_model=shell, hf_conversion_map_reversed={})
    restore(turtle, shell, "recursive")
    for i, expert in enumerate(shell.block.experts):
        for name in projections:
            original = getattr(native, name, None)
            if original is None:
                original = native.gate_up_proj[:, :6] if name == "gate_proj" else native.gate_up_proj[:, 6:]
            torch.testing.assert_close(expert.get_submodule(name).weight, original[i], atol=0, rtol=0)
    x = torch.tensor([[.2, -.5, .1, .9], [-.3, .1, .8, -.5]])
    indices = torch.tensor([[0, 1], [2, 1]])
    weights = torch.tensor([[.3, .7], [.6, .4]])
    oracle = torch.zeros_like(x)
    if family == "inkling_shared":
        gammas = torch.tensor([[.2, .7, .1], [.4, .1, .5]])
        actual = native(x, gammas)
        for i, expert in enumerate(shell.block.experts):
            oracle += expert.down_proj(native.act_fn(expert.gate_proj(x)) * expert.up_proj(x) * gammas[:, i:i+1])
    else:
        actual = native(x, indices, weights)
        for row in range(2):
            for rank in range(2):
                expert = shell.block.experts[int(indices[row, rank])]
                up = native.act_fn(expert.up_proj(x[row:row+1]))
                if family == "qwen3_moe":
                    up = native.act_fn(expert.gate_proj(x[row:row+1])) * expert.up_proj(x[row:row+1])
                oracle[row:row+1] += expert.down_proj(up) * weights[row, rank]
    torch.testing.assert_close(actual, oracle, atol=2e-7, rtol=2e-6)
