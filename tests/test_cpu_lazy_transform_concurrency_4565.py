# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU int4pack first-forward repacks must be single-flight (#4565).

The CPU kernels rewrite checkpoint buffers into their runtime layout on the
first eval forward. Concurrent first forwards must not both run that transform:
one would read half-converted tensors or repack already-packed words.
"""

import importlib
import inspect
import threading
import time

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import (
    TorchAtenLinear,
    _has_local_int4pack_cpu_ops,
)
from gptqmodel.nn_modules.qlinear.torch_fused import TORCH_HAS_FUSED_OPS, TorchFusedLinear


def _make_layer(cls):
    layer = cls(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=512,
        out_features=64,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    layer.post_init()
    layer.qweight.zero_()
    layer.qzeros.zero_()
    layer.scales.fill_(0.01)
    layer.g_idx = torch.arange(512, dtype=torch.int32) // 128
    layer.eval()
    return layer


@pytest.mark.parametrize("cls", [TorchAtenLinear, TorchFusedLinear], ids=["aten", "fused"])
def test_concurrent_first_forward_transforms_once(monkeypatch, cls):
    if cls is TorchAtenLinear and not _has_local_int4pack_cpu_ops():
        pytest.skip("ATen CPU int4pack ops are unavailable")
    if cls is TorchFusedLinear and not TORCH_HAS_FUSED_OPS:
        pytest.skip("CPU fused int4pack ops are unavailable")

    reference = _make_layer(cls)
    module = _make_layer(cls)
    x = torch.randn(2, 512, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = reference(x)

    original_transform = cls.transform
    calls = []

    def counted_transform(self, *args, **kwargs):
        calls.append(threading.get_ident())
        # Widen the check-then-act window without changing the transform.
        time.sleep(0.05)
        return original_transform(self, *args, **kwargs)

    monkeypatch.setattr(cls, "transform", counted_transform)

    errors = []
    barrier = threading.Barrier(8)

    def run():
        barrier.wait()
        try:
            with torch.inference_mode():
                module(x)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors[0]
    assert len(calls) == 1

    with torch.inference_mode():
        actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("gptqmodel.nn_modules.qlinear.torch_aten_kernel", "TorchAtenLinear"),
        ("gptqmodel.nn_modules.qlinear.torch_fused", "TorchFusedLinear"),
        ("gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq", "TorchAtenAwqLinear"),
        ("gptqmodel.nn_modules.qlinear.torch_fused_awq", "TorchFusedAwqLinear"),
        ("gptqmodel.nn_modules.qlinear.torch_cpu_avx512", "TorchCpuAvx512Linear"),
        ("gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq", "TorchCpuAvx512AwqLinear"),
    ],
)
def test_cpu_first_forward_uses_the_shared_transform_lock(module_name, class_name):
    """All six CPU lazy-repack forwards must take the shared cold-path lock."""
    cls = getattr(importlib.import_module(module_name), class_name)
    assert "_cpu_transform_lock" in inspect.getsource(cls.forward)
