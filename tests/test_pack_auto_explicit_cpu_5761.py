# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AUTO packing must honour an explicit CPU target when a GPU is visible.

``pack_impl="auto"`` selected ``pack_gpu`` from device availability alone, so a
caller that explicitly selected CPU reached the GPU packer, which rejects that
target. Both entry points then ran ``pack_original`` -- several times slower
than the bounded CPU block packer AUTO uses on a CPU-only host -- and
``pack_module`` still reported ``module.pack_gpu`` for the work the original
packer did (#5761).
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import FORMAT, GPTQConfig
from gptqmodel.utils.model import pack_module, resolve_pack_device
from gptqmodel.utils.packing import _normalize_pack_impl, hf_pack_layer


BITS = 4
GROUP_SIZE = 16
IN_FEATURES = 64
OUT_FEATURES = 64


def _linear():
    torch.manual_seed(5761)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(
            (torch.randint(0, 16, linear.weight.shape, dtype=torch.int32) - 8) * 0.5
        )
    return linear


def _module():
    return TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=False,
        format=FORMAT.GPTQ_V2,
        compile=False,
    )


def _inputs():
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.full((OUT_FEATURES, groups), 0.5)
    zeros = torch.full((OUT_FEATURES, groups), 8.0)
    g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    return scales, zeros, g_idx


def _quantize_config(pack_impl, device):
    return GPTQConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        device=device,
        pack_impl=pack_impl,
        format=FORMAT.GPTQ_V2,
        offload_to_disk=False,
    )


def _trace_packers(monkeypatch, calls, *, gpu_error=None):
    """Record which packers run; optionally make ``pack_gpu`` reject the layer."""
    for name in ("pack_gpu", "pack_block", "pack_original"):
        original = getattr(TorchLinear, name)

        def traced(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            if _name == "pack_gpu" and gpu_error is not None:
                raise gpu_error
            return _original(*args, **kwargs)

        monkeypatch.setattr(TorchLinear, name, traced)


def _pack_module(*, pack_impl, config_device, device):
    module = _module()
    scales, zeros, g_idx = _inputs()
    label = pack_module(
        "proj",
        {"proj": module},
        scales,
        zeros,
        g_idx,
        {"proj": _linear()},
        TorchLinear,
        lock=None,
        quantize_config=_quantize_config(pack_impl, config_device),
        device=device,
    )
    return label, module


def _hf_pack_layer(*, pack_impl, device):
    module = _module()
    scales, zeros, g_idx = _inputs()
    hf_pack_layer(
        module,
        _linear(),
        scales,
        zeros,
        g_idx,
        pack_impl=pack_impl,
        device=device,
        checkpoint_format=FORMAT.GPTQ_V2,
    )
    return module


def _assert_same_payload(actual, expected):
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        left = getattr(actual, name)
        right = getattr(expected, name)
        assert left.dtype == right.dtype, name
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)


def test_explicit_cpu_target_resolves_to_block_with_visible_gpu(monkeypatch):
    # The accelerator predicate is the only thing simulated; the resolver and
    # the packers stay real.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_pack_device(_quantize_config("auto", "cpu"), "cpu") == torch.device("cpu")

    module, linear = _module(), _linear()
    assert _normalize_pack_impl("auto", module, linear, target="cpu") == "block"
    assert _normalize_pack_impl("auto", module, linear, target=torch.device("cpu")) == "block"
    assert _normalize_pack_impl("auto", module, linear, target=DEVICE.CPU) == "block"


def test_explicit_cpu_target_resolves_to_block_without_accelerator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _normalize_pack_impl("auto", _module(), _linear(), target="cpu") == "block"


def test_auto_with_visible_gpu_keeps_gpu_path(monkeypatch):
    # Regression: an unspecified target (or an accelerator target) must still
    # select the GPU packer when one is visible.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _normalize_pack_impl("auto", _module(), _linear()) == "gpu"
    assert _normalize_pack_impl("auto", _module(), _linear(), target=None) == "gpu"
    assert _normalize_pack_impl("auto", _module(), _linear(), target="cuda:1") == "gpu"
    assert _normalize_pack_impl("auto", _module(), _linear(), target=DEVICE.CUDA) == "gpu"


def test_explicit_gpu_request_on_cpu_target_uses_cpu_packer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    module, linear = _module(), _linear()

    assert _normalize_pack_impl("gpu", module, linear, target="cpu") == "block"
    assert _normalize_pack_impl("gpu", module, linear) == "gpu"
    assert _normalize_pack_impl("gpu", module, linear, target="cuda:1") == "gpu"

    calls = []
    _trace_packers(monkeypatch, calls)
    label, _ = _pack_module(pack_impl="gpu", config_device="cpu", device="cpu")
    assert calls == ["pack_block"]
    assert label == "module.pack_block"


def test_pack_module_reports_block_for_explicit_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    _trace_packers(monkeypatch, calls)

    label, module = _pack_module(pack_impl="auto", config_device="cpu", device="cpu")

    # The GPU packer is never attempted for an explicit CPU target, and the
    # reported label names the packer that actually finished.
    assert calls == ["pack_block"]
    assert label == "module.pack_block"
    assert module.qweight.shape[0] == IN_FEATURES // 32 * BITS


def test_hf_pack_layer_uses_block_for_explicit_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    _trace_packers(monkeypatch, calls)

    _hf_pack_layer(pack_impl="auto", device="cpu")

    assert calls == ["pack_block"]


@pytest.mark.parametrize("entry", ["model", "hf"])
def test_explicit_cpu_payload_matches_block_and_original(monkeypatch, entry):
    # The dispatch change must not change a single packed word.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    if entry == "model":
        _, auto = _pack_module(pack_impl="auto", config_device="cpu", device="cpu")
        _, block = _pack_module(pack_impl="block", config_device="cpu", device="cpu")
        _, original = _pack_module(pack_impl="original", config_device="cpu", device="cpu")
    else:
        auto = _hf_pack_layer(pack_impl="auto", device="cpu")
        block = _hf_pack_layer(pack_impl="block", device="cpu")
        original = _hf_pack_layer(pack_impl="original", device="cpu")

    _assert_same_payload(auto, block)
    _assert_same_payload(auto, original)


def test_auto_gpu_rejection_falls_back_to_block_and_relabels(monkeypatch):
    # AUTO selected the GPU packer from availability, but this layer's geometry
    # is unsupported there. The block packer runs next and owns the label.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    _trace_packers(
        monkeypatch,
        calls,
        gpu_error=ValueError("pack_gpu expected a CUDA device, got `cpu`."),
    )

    label, _ = _pack_module(pack_impl="auto", config_device=None, device=None)

    assert calls == ["pack_gpu", "pack_block"]
    assert label == "module.pack_block"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device required")
def test_explicit_cpu_with_real_gpu_packs_on_cpu(monkeypatch):
    # Native evidence: a real visible accelerator must not be used when the
    # caller pinned the CPU target.
    torch.cuda.set_device(0)
    calls = []
    _trace_packers(monkeypatch, calls)

    label, _ = _pack_module(pack_impl="auto", config_device="cpu", device="cpu")

    assert calls == ["pack_block"]
    assert label == "module.pack_block"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device required")
def test_auto_with_real_gpu_packs_on_gpu(monkeypatch):
    # Regression: AUTO with a visible accelerator still selects pack_gpu.
    torch.cuda.set_device(0)
    calls = []
    _trace_packers(monkeypatch, calls)

    label, _ = _pack_module(pack_impl="auto", config_device="cuda", device="cuda:0")

    assert calls == ["pack_gpu"]
    assert label == "module.pack_gpu"
