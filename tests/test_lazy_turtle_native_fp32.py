# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Root native dtype policies must survive lazy dense checkpoint reads."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from gptqmodel.utils.structure import LazyTurtle
from tests.test_native_fp32_checkpoint_loading import PolicyModel, expected_dtype, write_checkpoint


def materialize(turtle, shell, mode):
    if mode == "recursive":
        turtle.materialize_submodule(target_model=shell, target_submodule=shell.block,
                                     module_path="block", device=torch.device("cpu"), show_progress=False)
    elif mode == "leaves":
        for name, module in shell.block.named_modules():
            turtle.materialize_submodule(target_model=shell, target_submodule=module,
                                         module_path="block" + ("." + name if name else ""),
                                         device=torch.device("cpu"), recurse=False, show_progress=False)
    elif mode == "direct":
        for module in shell.block.modules():
            turtle.materialize_direct_meta_tensors(target_model=shell, target_submodule=module, device=torch.device("cpu"))
    else:
        turtle.sync_all_meta(shell_model=shell, tie_after=False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mode", ["recursive", "leaves", "direct", "sync"])
@pytest.mark.parametrize("prefix", ["", "upstream_"])
def test_lazy_root_native_policy_exact_values_and_packed_storage_control(tmp_path, dtype, mode, prefix):
    _, values = write_checkpoint(tmp_path, sharded=True, prefix=prefix)
    shell = PolicyModel().to(dtype=dtype)
    shell.block.to_empty(device="meta")
    packed = {name: tensor.clone() for name, tensor in shell.state_dict().items() if name.startswith(("packed.", "exl3."))}
    kwargs = {"hf_conversion_map_reversed": {key: prefix + key for key in values}} if prefix else {}
    turtle = LazyTurtle(model_local_path=str(tmp_path), config=SimpleNamespace(),
                        model_init_kwargs={"dtype": dtype}, target_model=shell, **kwargs)
    for _ in range(2):
        materialize(turtle, shell, mode)
        for name, value in values.items():
            if name.startswith("block."):
                torch.testing.assert_close(shell.state_dict()[name], value.to(expected_dtype(name, value, dtype)),
                                           atol=0, rtol=0, msg=name)
        for name, value in packed.items():
            torch.testing.assert_close(shell.state_dict()[name], value, atol=0, rtol=0, msg=name)
        assert shell.block.moe_statics.scratch.dtype == dtype
        # Re-eviction must not retain narrowed state on another materialization.
        shell.block.to_empty(device="meta")
        shell.block.to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lazy_native_policy_missing_shard_repaired_retry_and_save(tmp_path, dtype):
    write_checkpoint(tmp_path, sharded=True)
    shell = PolicyModel().to(dtype=dtype)
    shell.block.to_empty(device="meta")
    turtle = LazyTurtle(model_local_path=str(tmp_path), config=SimpleNamespace(),
                        model_init_kwargs={"dtype": dtype}, target_model=shell)
    shard = tmp_path / "model-0.safetensors"
    saved = shard.read_bytes()
    shard.unlink()
    with pytest.raises((RuntimeError, FileNotFoundError), match="model-0.safetensors"):
        materialize(turtle, shell, "recursive")
    shard.write_bytes(saved)
    materialize(turtle, shell, "recursive")
    state = {name: tensor.contiguous() for name, tensor in shell.state_dict().items()}
    destination = tmp_path / "roundtrip.safetensors"
    save_file(state, destination)
    loaded = load_file(destination)
    assert loaded.keys() == state.keys()
    for name, value in state.items():
        torch.testing.assert_close(loaded[name], value, atol=0, rtol=0)
    assert loaded["block.moe_statics.correction"].dtype == torch.float32
    assert not list(tmp_path.glob("*.bin"))


def test_late_target_model_and_auto_dtype_use_native_policy_not_checkpoint_dtype(tmp_path):
    _, values = write_checkpoint(tmp_path)
    shell = PolicyModel().half()
    shell.block.to_empty(device="meta")
    turtle = LazyTurtle(model_local_path=str(tmp_path), config=SimpleNamespace(), model_init_kwargs={"dtype": "auto"})
    materialize(turtle, shell, "recursive")
    for name, value in values.items():
        if name.startswith("block."):
            torch.testing.assert_close(shell.state_dict()[name], value.to(expected_dtype(name, value, torch.float16)),
                                       atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_storage_slots_and_invalid_policy_repaired_retry(tmp_path, dtype):
    _, values = write_checkpoint(tmp_path)
    shell = PolicyModel().to(dtype=dtype)
    shell._keep_in_fp32_modules_strict = ["["]
    turtle = LazyTurtle(model_local_path=str(tmp_path), config=SimpleNamespace(),
                        model_init_kwargs={"torch_dtype": dtype})
    with pytest.raises(ValueError, match="invalid native FP32"):
        materialize(turtle, shell, "recursive")
    shell._keep_in_fp32_modules_strict = PolicyModel._keep_in_fp32_modules_strict
    materialize(turtle, shell, "recursive")
    for name, value in values.items():
        if name.startswith("block."):
            torch.testing.assert_close(shell.state_dict()[name], value.to(expected_dtype(name, value, dtype)),
                                       atol=0, rtol=0, msg=name)
