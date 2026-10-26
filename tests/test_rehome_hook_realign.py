# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from accelerate import cpu_offload_with_hook, disk_offload
from accelerate.hooks import (
    AlignDevicesHook,
    CpuOffload,
    ModelHook,
    SequentialHook,
    add_hook_to_module,
)

from gptqmodel.utils.looper_helpers import (
    clone_module_for_devices,
    rehome_module_to_device,
)
from gptqmodel.utils.model import simple_dispatch_model


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("target", [torch.device("cpu"), torch.device("cpu:0")])
def test_rehome_repairs_attached_stale_hook(
    nested: bool,
    target: torch.device,
) -> None:
    module = torch.nn.Linear(4, 4)
    inputs = torch.ones(2, 4)
    expected = module(inputs)
    hook = AlignDevicesHook(execution_device=torch.device("cpu"))
    attached = SequentialHook(ModelHook(), SequentialHook(hook)) if nested else hook
    add_hook_to_module(module, attached)
    hook.execution_device = torch.device("meta")
    with pytest.raises(RuntimeError):
        module(inputs)
    rehome_module_to_device(module, target, move_parameters=True, root_module=module)
    assert module._hf_hook is attached
    assert hook.execution_device == torch.device("cpu")
    torch.testing.assert_close(module(inputs), expected)


@pytest.mark.parametrize("execution_device", [None, "cuda:1", 1])
def test_hook_target_forms(execution_device: object) -> None:
    module = torch.nn.Linear(4, 4)
    hook = AlignDevicesHook(execution_device=execution_device)
    module._hf_hook = hook
    rehome_module_to_device(
        module, torch.device("cpu"), move_parameters=True, root_module=module
    )
    expected = None if execution_device is None else torch.device("cpu")
    assert hook.execution_device == expected


@pytest.mark.parametrize(
    "exclusion",
    [
        "buffer-only",
        "excluded-buffer",
        "failed-move",
        "replica",
        "offload",
        "cpu-offload",
    ],
)
def test_partial_or_managed_placement_preserves_owned_hooks(exclusion: str) -> None:
    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    hooks = [AlignDevicesHook(execution_device="cuda:1") for _ in module]
    for sub, hook in zip(module, hooks):
        sub._hf_hook = hook
    kwargs = {"move_parameters": True}
    if exclusion == "buffer-only":
        kwargs["move_parameters"] = False
    elif exclusion == "excluded-buffer":
        module[1].register_buffer(
            "cache", torch.empty(4, device="meta"), persistent=False
        )
        kwargs["include_non_persistent_buffers"] = False
    elif exclusion == "failed-move":
        module[1].weight = torch.nn.Parameter(torch.empty(4, 4, device="meta"))
    elif exclusion == "replica":
        module[1]._is_replica = True
    elif exclusion == "cpu-offload":
        module[1]._hf_hook = SequentialHook(
            hooks[1], CpuOffload(execution_device=torch.device("cpu"))
        )
    else:
        hooks[1].offload = True
    rehome_module_to_device(module, torch.device("cpu"), root_module=module, **kwargs)
    if exclusion in {"offload", "cpu-offload"}:
        assert hooks[0].execution_device == torch.device("cpu")
        assert hooks[1].execution_device == "cuda:1"
    else:
        assert all(hook.execution_device == "cuda:1" for hook in hooks)


def test_resident_parent_realigns_with_offloaded_child() -> None:
    class Parent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(4))
            self.child = torch.nn.Linear(4, 4)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.child(inputs * self.scale)

    module = Parent()
    inputs = torch.ones(2, 4)
    expected = module(inputs)
    child_hook = CpuOffload(execution_device=torch.device("cpu:0"))
    add_hook_to_module(module.child, child_hook)
    parent_hook = AlignDevicesHook(execution_device=torch.device("cpu"))
    add_hook_to_module(module, parent_hook)
    parent_hook.execution_device = torch.device("meta")

    with pytest.raises(RuntimeError):
        module(inputs)
    rehome_module_to_device(
        module, torch.device("cpu"), move_parameters=True, root_module=module
    )

    assert parent_hook.execution_device == torch.device("cpu")
    assert child_hook.execution_device == torch.device("cpu:0")
    torch.testing.assert_close(module(inputs), expected)


def test_real_offload_mapping_is_untouched(tmp_path: Path) -> None:
    module = torch.nn.Linear(4, 4)
    inputs = torch.ones(2, 4)
    expected = module(inputs)
    disk_offload(
        module, str(tmp_path / "offload"), execution_device=torch.device("cpu")
    )
    hook = module._hf_hook.hooks[-1]
    mapping = hook.weights_map
    rehome_module_to_device(
        module, torch.device("cpu"), move_parameters=True, root_module=module
    )
    assert hook.weights_map is mapping
    assert module.weight.is_meta
    for _ in range(2):
        torch.testing.assert_close(module(inputs), expected)
        assert module.weight.is_meta


def test_unrelated_hook_is_untouched() -> None:
    module = torch.nn.Linear(4, 4)
    hook = ModelHook()
    hook.execution_device = "meta"
    module._hf_hook = hook
    rehome_module_to_device(
        module, torch.device("cpu"), move_parameters=True, root_module=module
    )
    assert hook.execution_device == "meta"


def test_rehome_without_hook() -> None:
    module = torch.nn.Linear(4, 4)
    rehome_module_to_device(
        module, torch.device("cpu"), move_parameters=True, root_module=module
    )
    assert not hasattr(module, "_hf_hook")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("indexless", [False, True])
def test_rehome_attached_hook_across_cuda_devices(indexless: bool) -> None:
    source, target = torch.device("cuda:0"), torch.device("cuda:1")
    module = torch.nn.Linear(4, 4, device=source)
    inputs = torch.ones(2, 4, device=source)
    expected = module(inputs).cpu()
    hook = AlignDevicesHook(execution_device=source)
    add_hook_to_module(module, hook)
    with torch.cuda.device(target):
        requested = torch.device("cuda") if indexless else target
        rehome_module_to_device(
            module, requested, move_parameters=True, root_module=module
        )
    assert module.weight.device == target
    assert hook.execution_device == target
    actual = module(inputs)
    assert actual.device == target
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")
def test_rehome_attached_hook_to_indexless_mps() -> None:
    module = torch.nn.Linear(4, 4)
    inputs = torch.ones(2, 4)
    expected = module(inputs)
    hook = AlignDevicesHook(execution_device=torch.device("cpu"))
    add_hook_to_module(module, hook)
    rehome_module_to_device(
        module, torch.device("mps"), move_parameters=True, root_module=module
    )
    assert module.weight.device == torch.device("mps:0")
    assert hook.execution_device == module.weight.device
    actual = module(inputs)
    assert actual.device == module.weight.device
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("whole_parent", [False, True])
@pytest.mark.parametrize("known_root", [False, True])
def test_cpu_offload_parent_preserves_child_execution_device(
    whole_parent: bool,
    known_root: bool,
) -> None:
    module = torch.nn.Sequential(torch.nn.Linear(4, 4))
    inputs = torch.ones(2, 4)
    expected = module(inputs)
    execution_device = torch.device("cuda:0")
    child_hook = AlignDevicesHook(execution_device=execution_device)
    add_hook_to_module(module[0], child_hook)
    _, offload_hook = cpu_offload_with_hook(module, execution_device=execution_device)
    with torch.no_grad():
        torch.testing.assert_close(module(inputs).cpu(), expected)
    offload_hook.offload()
    assert not module[0].weight.is_inference()

    rehome_module_to_device(
        module if whole_parent else module[0],
        torch.device("cpu"),
        move_parameters=True,
        root_module=module if known_root else None,
    )

    assert not module[0].weight.is_inference()
    assert child_hook.execution_device == execution_device
    actual = module(inputs)
    assert actual.device == execution_device
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("wrong_root", [False, True])
def test_unknown_ownership_preserves_hooks(wrong_root: bool) -> None:
    module = torch.nn.Linear(4, 4)
    hook = AlignDevicesHook(execution_device="meta")
    module._hf_hook = hook
    rehome_module_to_device(
        module,
        torch.device("cpu"),
        move_parameters=True,
        root_module=torch.nn.Identity() if wrong_root else None,
    )
    assert hook.execution_device == "meta"


@pytest.mark.parametrize("managed_first", [False, True])
@pytest.mark.parametrize("cpu_offload", [False, True])
def test_shared_child_preserves_offload_ownership(
    managed_first: bool,
    cpu_offload: bool,
) -> None:
    child = torch.nn.Linear(4, 4)
    resident = torch.nn.Sequential(child)
    managed = torch.nn.Sequential(child)
    root = torch.nn.ModuleList(
        [managed, resident] if managed_first else [resident, managed]
    )
    hook = AlignDevicesHook(execution_device="cuda:0")
    child._hf_hook = hook
    offload = (
        CpuOffload(execution_device="cuda:0")
        if cpu_offload
        else AlignDevicesHook(execution_device="cuda:0", offload=True)
    )
    managed._hf_hook = SequentialHook(ModelHook(), SequentialHook(offload))
    rehome_module_to_device(
        resident,
        torch.device("cpu"),
        move_parameters=True,
        root_module=root,
    )
    assert hook.execution_device == "cuda:0"


def test_offloaded_sibling_does_not_block_resident_realign() -> None:
    root = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    hook = AlignDevicesHook(execution_device="meta")
    root[0]._hf_hook = hook
    root[1]._hf_hook = CpuOffload(execution_device="cuda:0")
    rehome_module_to_device(
        root[0],
        torch.device("cpu"),
        move_parameters=True,
        root_module=root,
    )
    assert hook.execution_device == torch.device("cpu")
    assert root[1]._hf_hook.execution_device == "cuda:0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dispatch_pipeline_preserves_offloaded_child_hook() -> None:
    model = torch.nn.Sequential(
        OrderedDict(
            [
                ("block", torch.nn.Sequential(torch.nn.Linear(4, 4))),
                ("next_block", torch.nn.Sequential(torch.nn.Linear(4, 4))),
            ]
        )
    )
    inputs = torch.ones(2, 4)
    expected = model(inputs)
    simple_dispatch_model(
        model,
        {"block": "cpu", "next_block": "cpu", "block.0": "cuda:0"},
    )
    with torch.no_grad():
        torch.testing.assert_close(model(inputs).cpu(), expected)
    assert model.block[0].weight.device == torch.device("cpu")
    rehome_module_to_device(
        model.block[0],
        torch.device("cpu"),
        move_parameters=True,
        root_module=model,
    )
    with torch.no_grad():
        torch.testing.assert_close(model(inputs).cpu(), expected)


def test_clone_preparation_realigns_owned_source() -> None:
    root = torch.nn.Sequential(torch.nn.Linear(4, 4))
    child = root[0]
    inputs = torch.ones(2, 4)
    expected = child(inputs)
    hook = AlignDevicesHook(execution_device=torch.device("cpu"))
    add_hook_to_module(child, hook)
    hook.execution_device = torch.device("meta")
    clones = clone_module_for_devices(child, [torch.device("cpu")], root_module=root)
    assert clones[torch.device("cpu")] is child
    torch.testing.assert_close(child(inputs), expected)


def test_detached_deepcopies_realign_their_own_hooks() -> None:
    module = torch.nn.Linear(4, 4)
    inputs = torch.ones(2, 4)
    expected = module(inputs)
    hook = AlignDevicesHook(execution_device=torch.device("cpu"))
    add_hook_to_module(module, hook)
    hook.execution_device = torch.device("meta")
    clones = clone_module_for_devices(
        module,
        [torch.device("cpu"), torch.device("cpu:0")],
    )
    assert hook.execution_device == torch.device("meta")
    for clone in clones.values():
        assert clone is not module
        torch.testing.assert_close(clone(inputs), expected)
