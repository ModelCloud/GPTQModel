# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EXL3 in-session inference after ``quantize()`` must work off CUDA (#4789).

``quantize()`` leaves the packed layers on CPU (with the default
``offload_to_disk=True`` and with it disabled). ``ExllamaV3Linear`` is the
native CUDA-only kernel, so the first fp16 forward after quantization raised
``NotImplementedError`` from ``_ensure_inner``. BF16 already reconstructed
through the portable kernel off CUDA, which is why only fp16 failed.

The native class now falls back to the same portable reconstruction whenever
its trellis is not on CUDA. Keeping the class means ``save()`` still writes the
native EXL3 module schema, and the in-session output equals the reloaded
``EXL3_TORCH`` checkpoint exactly.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper.checkpoint_modules import is_packed_module, packed_module_config
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.quantization.config import EXL3Config


DTYPES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]


def _tensors(in_features: int, out_features: int, *, bits: int = 4, seed: int = 4789):
    generator = torch.Generator().manual_seed(seed)
    return {
        "trellis": torch.randint(
            -32768,
            32767,
            (in_features // 16, out_features // 16, bits * 16),
            dtype=torch.int16,
            generator=generator,
        ),
        "suh": (torch.rand(in_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "svh": (torch.rand(out_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "bias": torch.randn(out_features, generator=generator, dtype=torch.float16) * 0.01,
    }


def _in_session_module(in_features: int = 128, out_features: int = 128, *, name: str = "proj"):
    """The module ``quantize()`` leaves behind: the native class, on CPU."""

    return ExllamaV3Linear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name=name,
        tensors=_tensors(in_features, out_features),
    ).eval()


def _reloaded_module(in_features: int = 128, out_features: int = 128, *, name: str = "proj"):
    """What loading the saved checkpoint for a CPU device builds (EXL3_TORCH)."""

    return ExllamaV3TorchLinear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name=name,
        tensors=_tensors(in_features, out_features),
    ).eval()


def test_in_session_module_starts_on_cpu_with_the_native_class():
    module = _in_session_module()
    assert type(module) is ExllamaV3Linear
    assert module.SUPPORTS_DEVICES == [DEVICE.CUDA, DEVICE.ROCM]
    assert module.trellis.device.type == "cpu"
    assert module._needs_torch_fallback()


@pytest.mark.parametrize("dtype", DTYPES)
def test_in_session_forward_matches_the_reloaded_checkpoint(dtype):
    module = _in_session_module()
    x = torch.randn(4, 128, dtype=dtype)

    with torch.inference_mode():
        output = module(x)
        expected = _reloaded_module()(x)

    assert output.dtype is dtype
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    # The portable reconstruction is what ran; the native handle was never built.
    assert module._inner is None
    assert module._torch_inner is not None


@pytest.mark.parametrize("dtype", DTYPES)
def test_offloaded_decoder_stack_forwards_in_session(dtype):
    """A two-projection decoder stack left on CPU must run in both dtypes."""

    class _Stack(nn.Module):
        def __init__(self, module_cls):
            super().__init__()
            self.up = module_cls.from_tensors(
                in_features=128, out_features=256, name="up", tensors=_tensors(128, 256)
            )
            self.down = module_cls.from_tensors(
                in_features=256, out_features=128, name="down", tensors=_tensors(256, 128)
            )

        def forward(self, hidden):
            return self.down(torch.nn.functional.gelu(self.up(hidden)))

    in_session = _Stack(ExllamaV3Linear).eval()
    reloaded = _Stack(ExllamaV3TorchLinear).eval()
    assert all(type(module) is ExllamaV3Linear for module in in_session.modules() if hasattr(module, "trellis"))

    hidden = torch.randn(2, 5, 128, dtype=dtype)
    with torch.inference_mode():
        output = in_session(hidden)
        expected = reloaded(hidden)

    assert output.shape == (2, 5, 128)
    assert output.dtype is dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_in_session_forward_keeps_the_saved_exl3_schema():
    """The in-session fallback must not change what ``save()`` writes."""

    qcfg = EXL3Config(bits=4)
    assert qcfg.offload_to_disk is True
    module = _in_session_module()
    before = packed_module_config(module, qcfg, module_name="model.layers.0.mlp.up_proj")

    with torch.inference_mode():
        module(torch.randn(2, 128, dtype=torch.float16))

    after = packed_module_config(module, qcfg, module_name="model.layers.0.mlp.up_proj")
    assert after == before
    assert after["kind"] == "exl3"
    assert after["buffers"]
    # Still the native class, so the checkpoint keeps the native EXL3 layout.
    assert type(module) is ExllamaV3Linear
    assert is_packed_module(module)


@pytest.mark.parametrize("device", [torch.device("cpu"), DEVICE.CPU, "cpu"])
def test_native_device_error_names_the_module_device(device):
    with pytest.raises(NotImplementedError) as excinfo:
        ExllamaV3Linear.validate_device(device)

    message = str(excinfo.value)
    assert "`cpu`" in message
    assert "move the model to a CUDA device" in message
    assert "EXL3_TORCH" in message
    # The host has CUDA; only this module is on the wrong device.
    assert "host" not in message.lower()


def test_meta_trellis_still_reports_the_unmaterialized_state():
    """The fallback must not mask an unmaterialized checkpoint."""

    module = ExllamaV3Linear(in_features=128, out_features=128, name="meta", tensors={})
    module.trellis = SimpleNamespace(device=torch.device("meta"))
    assert not module._needs_torch_fallback()

    with pytest.raises(RuntimeError, match="has not been materialized"):
        with torch.inference_mode():
            module(torch.randn(2, 128, dtype=torch.float16))


def test_cuda_trellis_keeps_the_native_kernel_path():
    """CUDA models (and the BF16 fast path) must keep `_ensure_inner`."""

    module = ExllamaV3Linear(in_features=128, out_features=128, name="cuda", tensors={})
    module.trellis = SimpleNamespace(device=torch.device("cuda", 0))
    assert not module._needs_torch_fallback()
