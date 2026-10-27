"""Marlin's packed weights must match its Accelerate CPU offload snapshot."""

import copy

import accelerate
import pytest
import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from torch import nn

import gptqmodel.nn_modules.qlinear.marlin as marlin_module
from gptqmodel.utils.model import gptqmodel_post_init


def _stub_cuda_runtime(monkeypatch):
    # Keep Marlin's post-init flow while making its CUDA-only operations
    # deterministic on the CPU runner.
    monkeypatch.setattr(marlin_module, "marlin_import_exception", None)
    monkeypatch.setattr(
        marlin_module, "marlin_validate_runtime_device", lambda *args, **kwargs: (8, 0)
    )
    monkeypatch.setattr(marlin_module, "marlin_runtime_available", lambda dtype: True)
    monkeypatch.setattr(
        marlin_module,
        "marlin_make_workspace_new",
        lambda device: torch.zeros(1, dtype=torch.int32, device=device),
    )
    monkeypatch.setattr(
        marlin_module,
        "gptq_marlin_repack",
        lambda packed, perm, size_k, size_n, num_bits, dtype=None: packed.reshape(
            size_k // 16, size_n * 16 // (32 // num_bits)
        ).contiguous(),
    )
    monkeypatch.setattr(
        marlin_module, "marlin_permute_scales", lambda scales, **kwargs: scales
    )
    monkeypatch.setattr(marlin_module, "marlin_permute_bias", lambda bias: bias)


def _layer(desc_act):
    layer = marlin_module.MarlinLinear(
        bits=4,
        group_size=128,
        desc_act=desc_act,
        sym=True,
        in_features=256,
        out_features=128,
        bias=True,
        dtype=torch.float16,
    )
    generator = torch.Generator().manual_seed(1499)
    layer.qweight.copy_(
        torch.randint(
            -(2**31),
            2**31 - 1,
            layer.qweight.shape,
            dtype=torch.int32,
            generator=generator,
        )
    )
    layer.scales.copy_(torch.randn(layer.scales.shape, generator=generator))
    layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator))
    layer.qzeros.zero_()
    layer.g_idx.copy_(
        torch.arange(layer.in_features, dtype=torch.int32) // layer.group_size
    )
    return layer


@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("sequential_hook", [False, True])
def test_marlin_cpu_offload_refreshes_packed_weights(
    monkeypatch, desc_act, sequential_hook
):
    _stub_cuda_runtime(monkeypatch)
    model = nn.Sequential(_layer(desc_act))
    resident = copy.deepcopy(model[0])
    resident.post_init()
    expected = resident.state_dict()

    accelerate.cpu_offload(model, execution_device=torch.device("cpu"))
    layer = model[0]
    assert layer.qweight.is_meta
    if sequential_hook:
        add_hook_to_module(layer, ModelHook(), append=True)
    hook = layer._hf_hook.hooks[0] if sequential_hook else layer._hf_hook
    original_keys = set(hook.weights_map.dataset)

    gptqmodel_post_init(model, use_act_order=desc_act)
    assert layer.qweight.is_meta
    assert layer.g_idx.shape == expected["g_idx"].shape

    # Two reloads catch both the first-forward shape failure and a stale map
    # after the hook has returned the layer to meta.
    for _ in range(2):
        layer._hf_hook.pre_forward(layer)
        try:
            for name, want in expected.items():
                torch.testing.assert_close(
                    layer.state_dict()[name], want, rtol=0, atol=0
                )
        finally:
            layer._hf_hook.post_forward(layer, None)
        assert all(parameter.is_meta for parameter in layer.parameters(recurse=False))

    assert all(
        tensor.device.type == "cpu" for tensor in hook.weights_map.dataset.values()
    )
    assert set(hook.weights_map.dataset) == original_keys
    assert sum(
        tensor.numel() * tensor.element_size()
        for tensor in hook.weights_map.dataset.values()
    ) == sum(tensor.numel() * tensor.element_size() for tensor in expected.values())
    assert set(layer.state_dict()) == {"qweight", "scales", "qzeros", "g_idx", "bias"}


def test_marlin_disk_offload_is_rejected_before_repacking(monkeypatch, tmp_path):
    _stub_cuda_runtime(monkeypatch)
    model = nn.Sequential(_layer(False), nn.LogSigmoid())
    accelerate.dispatch_model(
        model, device_map={"0": "disk", "1": "cpu"}, offload_dir=str(tmp_path)
    )
    assert model[0].qweight.is_meta
    with pytest.raises(RuntimeError, match="writable in-memory weights map"):
        gptqmodel_post_init(model, use_act_order=False)
    assert model[0].qweight.is_meta
    assert model[0].g_idx.shape == (256,)
