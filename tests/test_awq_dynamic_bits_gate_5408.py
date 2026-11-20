# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The model-wide AWQ GEMM pack gate must not reject per-module widths.

``AWQConfig(dynamic={"...": {"bits": 8}})`` on a 4-bit GEMM run is supported:
``AWQProcessor._resolve_qlinear_kernel`` packs the non-4-bit module with the
portable ``AwqTorchLinear``. The pre-quantize ``select_quant_linear(...,
pack=True)`` call, however, validated the model-wide ``AwqGEMMLinear``
candidate against every dynamic width and raised before calibration (#5408).
"""

import pytest

from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils import importer
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


class _GemmKernel:
    """Stand-in for ``AwqGEMMLinear``: 4-bit only, but packable."""

    SUPPORTS_BITS = [4]
    SUPPORTS_SHARDS = True
    SUPPORTS_SHARDED_LOAD = True

    calls = []

    @classmethod
    def pack(cls, *args, **kwargs):  # pragma: no cover - selection only
        raise AssertionError("the gate test never packs")

    @classmethod
    def cached_validate_once(cls):
        return True, None

    @classmethod
    def validate(cls, **kwargs):
        cls.calls.append(kwargs.get("dynamic"))
        return True, None


class _PortableKernel:
    """Stand-in for ``AwqTorchLinear``: 2-8 bits."""

    SUPPORTS_BITS = [2, 3, 4, 5, 6, 7, 8]
    SUPPORTS_SHARDS = True
    SUPPORTS_SHARDED_LOAD = True

    calls = []

    @classmethod
    def pack(cls, *args, **kwargs):  # pragma: no cover - selection only
        raise AssertionError("the gate test never packs")

    @classmethod
    def cached_validate_once(cls):
        return True, None

    @classmethod
    def validate(cls, **kwargs):
        cls.calls.append(kwargs.get("dynamic"))
        return True, None


def _record_gate(monkeypatch, kernel):
    kernel.calls = []
    importer.clear_validation_cache()
    monkeypatch.setattr(importer, "get_kernel_for_backend", lambda *args: kernel)

    class Recorded:
        def __getitem__(self, key):
            if key == "dynamic":
                assert kernel.calls, "the gate never validated the kernel"
                return kernel.calls[-1]
            if key == "bits":
                return 4
            raise KeyError(key)

    return Recorded()


def _select(**overrides):
    kwargs = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "device": None,
        "backend": BACKEND.AWQ_GEMM,
        "format": FORMAT.GEMM,
        "quant_method": METHOD.AWQ,
        "pack": True,
        "dynamic": {"model.layers.1.mlp.down_proj": {"bits": 8}},
    }
    kwargs.update(overrides)
    return select_quant_linear(**kwargs)


def _dynamic_bits_supported(kernel, bits, dynamic):
    return importer._dynamic_bits_supported(kernel, bits, dynamic)


def test_dynamic_bits_supported_mirrors_the_kernel_contract():
    dynamic = {"layer": {"bits": 8}}
    assert not _dynamic_bits_supported(_GemmKernel, 4, dynamic)
    assert _dynamic_bits_supported(_PortableKernel, 4, dynamic)
    assert _dynamic_bits_supported(_GemmKernel, 4, None)
    # A rule that only changes another key keeps the global width.
    assert _dynamic_bits_supported(_GemmKernel, 4, {"layer": {"group_size": 64}})


def test_awq_gemm_pack_gate_ignores_per_module_widths(monkeypatch):
    recorded = _record_gate(monkeypatch, _GemmKernel)

    assert _select() is _GemmKernel
    # The model-wide candidate is validated against the global width only; the
    # per-module AwqTorchLinear fallback owns the 8-bit layer.
    assert recorded["dynamic"] is None
    assert recorded["bits"] == 4


def test_awq_gemm_load_gate_keeps_every_dynamic_width(monkeypatch):
    recorded = _record_gate(monkeypatch, _GemmKernel)

    assert _select(pack=False) is _GemmKernel
    assert recorded["dynamic"] == {"model.layers.1.mlp.down_proj": {"bits": 8}}


def test_awq_gemm_pack_gate_keeps_supported_dynamic_widths(monkeypatch):
    recorded = _record_gate(monkeypatch, _PortableKernel)

    assert _select() is _PortableKernel
    assert recorded["dynamic"] == {"model.layers.1.mlp.down_proj": {"bits": 8}}


def test_awq_gemm_pack_gate_keeps_same_width_rules(monkeypatch):
    recorded = _record_gate(monkeypatch, _GemmKernel)

    dynamic = {"model.layers.1.mlp.down_proj": {"group_size": 64}}
    assert _select(dynamic=dynamic) is _GemmKernel
    assert recorded["dynamic"] == dynamic


@pytest.mark.parametrize(
    "overrides",
    [
        # GEMV/GEMV_FAST have no per-module width fallback.
        {"format": FORMAT.GEMV},
        # Other methods keep the full dynamic gate.
        {"quant_method": METHOD.GPTQ, "backend": BACKEND.GPTQ_TORCH, "format": FORMAT.GPTQ},
    ],
)
def test_other_formats_and_methods_keep_the_full_gate(monkeypatch, overrides):
    recorded = _record_gate(monkeypatch, _GemmKernel)

    _select(**overrides)
    assert recorded["dynamic"] == {"model.layers.1.mlp.down_proj": {"bits": 8}}
