# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4793.

``ExllamaV3Linear._ensure_inner`` used to drop the old handle, build a new one
and then re-read ``self._inner`` before returning it, with no lock. A burst of
concurrent cold-start forwards therefore each rebuilt the handle (every rebuild
unloads the previous one) and could be handed the ``None`` another thread had
just published:

    AttributeError: 'NoneType' object has no attribute 'forward'

The native ``LinearEXL3`` is stubbed here: the bug is the Python handle
publication, not the kernel.
"""

import threading
import time

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear


THREADS = 8
IN_FEATURES = 32
OUT_FEATURES = 32


class _FakeNative:
    """Minimal stand-in for ``LinearEXL3`` that records its lifecycle."""

    made: list["_FakeNative"] = []
    build_delay = 0.01

    def __init__(self, **kwargs):
        # Widen the window the real constructor needs (trellis upload).
        time.sleep(type(self).build_delay)
        self.in_features = kwargs["in_features"]
        self.out_features = kwargs["out_features"]
        self.unloaded = False
        type(self).made.append(self)

    def unload(self):
        self.unloaded = True

    def forward(self, x, cache, out_dtype=None):
        if self.unloaded:
            raise RuntimeError("the native handle was unloaded by another thread")
        dtype = out_dtype or x.dtype
        return torch.zeros((*x.shape[:-1], self.out_features), dtype=dtype)


def _stub_native(monkeypatch):
    import gptqmodel.exllamav3.modules.quant.exl3 as native_module

    _FakeNative.made = []
    monkeypatch.setattr(native_module, "LinearEXL3", _FakeNative)
    monkeypatch.setattr(
        ExllamaV3Linear, "validate_device", classmethod(lambda cls, device: None)
    )


def _module() -> ExllamaV3Linear:
    packed = {
        "trellis": torch.zeros((2, 2, 16), dtype=torch.int16),
        "suh": torch.ones(IN_FEATURES, dtype=torch.float16),
        "svh": torch.ones(OUT_FEATURES, dtype=torch.float16),
        "su": None,
        "sv": None,
        "bias": None,
        "mcg": None,
        "mul1": None,
    }
    return ExllamaV3Linear.from_tensors(
        in_features=IN_FEATURES, out_features=OUT_FEATURES, name="proj", tensors=packed
    )


def test_cold_start_burst_publishes_one_handle(monkeypatch):
    """One cold start builds one handle; every thread gets that handle."""

    _stub_native(monkeypatch)
    module = _module()
    x = torch.zeros(2, IN_FEATURES, dtype=torch.float16)

    results = []
    errors = []
    barrier = threading.Barrier(THREADS)

    def run():
        barrier.wait()
        try:
            with torch.inference_mode():
                module(x)
            results.append(module._inner)
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=run) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors[:4]
    # Every rebuild unloads the previous handle, so more than one build means a
    # thread was handed a dropped handle (the reported AttributeError).
    assert len(_FakeNative.made) == 1, len(_FakeNative.made)
    assert module._inner is _FakeNative.made[0]
    assert all(not handle.unloaded for handle in _FakeNative.made)
    assert set(map(id, results)) == {id(_FakeNative.made[0])}


@pytest.mark.parametrize("rounds", [5])
def test_repeated_clear_then_burst_never_returns_none(monkeypatch, rounds):
    """`clear_weight_cache()` then a burst is the reported repro loop."""

    _stub_native(monkeypatch)
    module = _module()
    x = torch.zeros(2, IN_FEATURES, dtype=torch.float16)

    errors = []
    for _ in range(rounds):
        module.clear_weight_cache()
        before = len(_FakeNative.made)
        barrier = threading.Barrier(THREADS)

        def run():
            barrier.wait()
            try:
                with torch.inference_mode():
                    out = module(x)
                if out.shape != (2, OUT_FEATURES):
                    errors.append(f"unexpected shape {tuple(out.shape)}")
            except BaseException as exc:  # noqa: BLE001 - asserted below
                errors.append(f"{type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=run) for _ in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors, errors[:4]
        assert len(_FakeNative.made) - before == 1, len(_FakeNative.made) - before
