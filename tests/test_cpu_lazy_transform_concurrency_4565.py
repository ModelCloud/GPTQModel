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
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_fused import TORCH_HAS_FUSED_OPS, TorchFusedLinear
from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear


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


_INT4PACK_CLASSES = [
    pytest.param(TorchAtenLinear, id="aten"),
    pytest.param(TorchFusedLinear, id="fused"),
    pytest.param(TorchAtenAwqLinear, id="aten-awq"),
    pytest.param(TorchFusedAwqLinear, id="fused-awq"),
]


@pytest.mark.parametrize("cls", _INT4PACK_CLASSES)
def test_state_dict_during_first_forward_is_consistent(monkeypatch, cls):
    """A save racing the cold transform must not crash or emit runtime tensors.

    ``state_dict()`` iterates ``_buffers`` while the first forward registers the
    fused snapshot, rebinds ``qweight``/``qzeros`` and pops the ``qweight``
    clone. Without the shared lock it raised
    ``dictionary changed size during iteration`` (GIL on) and could return the
    runtime int4pack words under the checkpoint key (GIL off) - issue #4788.
    """

    if cls in (TorchAtenLinear, TorchAtenAwqLinear) and not _has_local_int4pack_cpu_ops():
        pytest.skip("ATen CPU int4pack ops are unavailable")
    if cls in (TorchFusedLinear, TorchFusedAwqLinear) and not TORCH_HAS_FUSED_OPS:
        pytest.skip("CPU fused int4pack ops are unavailable")

    module = _make_layer(cls)
    # Pre-transform state dict is the checkpoint layout the save must keep.
    reference = {key: value.detach().clone() for key, value in module.state_dict().items()}

    original_transform = cls.transform

    def slow_transform(self, *args, **kwargs):
        # Widen the window the transform holds the buffers in.
        time.sleep(0.05)
        return original_transform(self, *args, **kwargs)

    monkeypatch.setattr(cls, "transform", slow_transform)

    x = torch.randn(2, 512, dtype=torch.bfloat16)
    errors = []
    mismatches = []
    stop = threading.Event()
    barrier = threading.Barrier(3)

    def hammer():
        barrier.wait()
        while not stop.is_set():
            try:
                state = module.state_dict()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)
                continue
            for key, expected in reference.items():
                actual = state.get(key)
                if actual is None:
                    mismatches.append(f"{key} missing")
                elif actual.dtype != expected.dtype or actual.shape != expected.shape:
                    mismatches.append(
                        f"{key} {actual.dtype}{tuple(actual.shape)} != "
                        f"{expected.dtype}{tuple(expected.shape)}"
                    )
                elif not torch.equal(actual, expected):
                    mismatches.append(f"{key} values differ")

    threads = [threading.Thread(target=hammer) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    try:
        with torch.inference_mode():
            module(x)
        time.sleep(0.1)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=30)

    assert not errors, [f"{type(exc).__name__}: {exc}" for exc in errors[:4]]
    assert not mismatches, mismatches[:8]
    assert module.linear_mode == "inference"


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
def test_state_dict_waits_for_the_shared_transform_lock(module_name, class_name):
    """`state_dict()` must be atomic with the cold transform, not just forwards.

    The transform registers the fused snapshot buffers, rebinds `qweight` and
    pops the clone; a reader that does not take the same lock can observe any
    of those intermediate states (#4788). Hold the lock and prove the reader
    blocks instead of returning a half-built state dict.
    """

    cls = getattr(importlib.import_module(module_name), class_name)
    available, error = cls.validate_once()
    if not available:
        pytest.skip(f"{class_name} is unavailable on this host: {error}")

    module = _make_layer(cls)
    started = threading.Event()
    finished = threading.Event()

    def read():
        started.set()
        module.state_dict()
        finished.set()

    with module._cpu_transform_lock:
        reader = threading.Thread(target=read)
        reader.start()
        assert started.wait(5)
        # Blocked on the lock the transform holds.
        assert not finished.wait(0.2)
    assert finished.wait(5)
    reader.join(timeout=5)
