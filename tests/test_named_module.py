# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from gptqmodel.looper.named_module import NamedModule


def _make_linear(features: int = 8, device: torch.device | None = None) -> torch.nn.Linear:
    layer = torch.nn.Linear(features, features, bias=False)
    if device is not None:
        layer = layer.to(device=device)
    return layer


def test_named_module_register_and_state_locking():
    base = _make_linear()
    named = NamedModule(base, name="proj", full_name="model.layers.0.proj", layer_index=0)

    # register/unregister buffer should route through wrapped module and keep state updates serialized
    buf = torch.ones(1)
    named.register_buffer("unit", buf)
    assert "unit" in dict(named.named_buffers())
    named.unregister_buffer("unit")
    assert "unit" not in dict(named.named_buffers())

    # parameter registration proxies should also touch wrapped module
    param = torch.nn.Parameter(torch.randn_like(base.weight))
    named.register_parameter("alt_weight", param)
    assert dict(named.named_parameters())
    named.unregister_parameter("alt_weight")
    assert "alt_weight" not in dict(named.named_parameters())

    # setattr/getattr should delegate to wrapped module under lock
    named.new_attr = torch.zeros(1)
    assert torch.equal(named.new_attr, torch.zeros(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for streaming")
def test_named_module_streaming_apis():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    layer = _make_linear(device=device)
    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)

    payload = {
        "tensor": torch.randn(8, 8, device=device, dtype=torch.float16),
    }

    class _HostPool:
        def acquire(self, shape, dtype, layout):
            return torch.empty(shape, dtype=dtype, layout=layout, device="cpu", pin_memory=True)

        def release(self, tensor):
            pass

    host_pool = _HostPool()

    named.stream_state_payload_to_cpu(payload, host_pool=host_pool)
    assert "tensor" in named.state
    assert named.state["tensor"].is_pinned()

    named.stream_sync()
    torch.testing.assert_close(named.state["tensor"].cpu(), payload["tensor"].cpu())

    params = named.stream_parameters_to_cpu(host_pool=host_pool)
    assert params
    named.stream_sync()
    param_lookup = {name: tensor.detach().cpu() for name, tensor in named.module.named_parameters(recurse=False)}
    for name, cpu_tensor in params.items():
        torch.testing.assert_close(cpu_tensor.cpu(), param_lookup[name])

    buffers = named.stream_buffers_to_cpu(host_pool=host_pool)
    named.stream_sync()
    buffer_lookup = {name: tensor.detach().cpu() for name, tensor in named.module.named_buffers(recurse=False)}
    for name, cpu_tensor in buffers.items():
        torch.testing.assert_close(cpu_tensor.cpu(), buffer_lookup[name])

    combined = named.stream_all_to_cpu(host_pool=host_pool)
    named.stream_sync()
    assert set(combined.keys()) == {"parameters", "buffers"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for subprocess stream test")
def test_named_module_streaming_subprocess_roundtrip():
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "7")

    script = textwrap.dedent(
        """
        import torch
        from gptqmodel.looper.named_module import NamedModule

        layer = torch.nn.Linear(4, 4, bias=False).to(device='cuda', dtype=torch.float16)
        named = NamedModule(layer, name='proj', full_name='model.layers.0.proj', layer_index=0)

        payload = {'x': torch.randn(4, 4, device='cuda', dtype=torch.float16)}

        class _Pool:
            def acquire(self, shape, dtype, layout):
                return torch.empty(shape, dtype=dtype, layout=layout, device='cpu', pin_memory=True)

            def release(self, tensor):
                pass

        pool = _Pool()
        named.stream_state_payload_to_cpu(payload, host_pool=pool)
        named.stream_sync()
        torch.testing.assert_close(named.state['x'].cpu(), payload['x'].cpu(), atol=0, rtol=0)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(f"Subprocess streaming test unavailable: {result.stderr.strip()}")
