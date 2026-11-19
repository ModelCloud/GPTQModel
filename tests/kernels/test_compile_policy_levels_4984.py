# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A trace-clean kernel must not be forced eager by the #4669 policy (#4984).

`BaseQModel._kernels_support_compile()` turns transformers' static-cache
auto-compile off unless every quantized linear declares `SUPPORTS_COMPILE`.
That flag is derived from the same measurement as `SUPPORTS_COMPILE_LEVEL`, but
the two declarations drifted apart: `BitsAndBytesLinear` and `MacaqLinear` both
declare `CUDAGRAPH` (and both trace to one graph with zero breaks), yet both
inherited `SUPPORTS_COMPILE = False`. Every bitsandbytes checkpoint on CUDA is
therefore 2.2-3.5x slower under `generate(cache_implementation="static")` than
the compiled path it suppresses (#4984).

The tests here re-measure the trace on real hardware and require the policy to
agree with it: a forward that reaches `INDUCTOR`/`CUDAGRAPH` with zero graph
breaks either opts into compiled generation, or records the measured loss that
justifies staying eager. They use only the pre-#4984 declaration surface
(`SUPPORTS_COMPILE`, `SUPPORTS_COMPILE_LEVEL`, `tests/kernels/compile_probe.py`),
so the same file is red on unmodified `main` and green with the fix.
"""

from __future__ import annotations

import pytest

from gptqmodel.nn_modules.qlinear._compile_support import COMPILE_LEVEL, compile_level
from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from tests.kernels.compile_probe import probe, protocol_kernels, skip_without_measurable_device
from tests.kernels.test_kernel_compile_levels_quant import (
    _build_bnb,
    _build_macaq,
    _build_vptq,
    _pack_qqq,
    _unavailable,
)


#: From this level up, a compiled static-cache decode is expected to pay off.
POLICY_FLOOR = COMPILE_LEVEL.INDUCTOR

CASES = [
    pytest.param(
        BitsAndBytesLinear,
        _build_bnb,
        lambda: _unavailable(BitsAndBytesLinear),
        id="BitsAndBytesLinear",
    ),
    pytest.param(
        MacaqLinear,
        lambda: _build_macaq(MacaqLinear),
        lambda: None,
        id="MacaqLinear",
    ),
    pytest.param(
        QQQTorchLinear,
        lambda: _pack_qqq(QQQTorchLinear),
        lambda: None,
        id="QQQTorchLinear",
    ),
    pytest.param(
        VPTQLinear,
        lambda: _build_vptq(VPTQLinear),
        lambda: None,
        id="VPTQLinear",
    ),
]


def _eager_reason(kernel_cls) -> str:
    """The recorded policy reason, read without the #4984 helper."""

    return (getattr(kernel_cls, "SUPPORTS_COMPILE_EAGER_REASON", "") or "").strip()


@pytest.mark.parametrize("kernel_cls,build,unavailable", CASES)
@pytest.mark.gpu
def test_a_trace_clean_kernel_is_not_forced_eager(kernel_cls, build, unavailable):
    """Measure the forward, then require the policy to follow the measurement."""

    reason = unavailable()
    if reason is not None:
        pytest.skip(reason)
    skip_without_measurable_device(kernel_cls)

    try:
        module, x, _ = build()
    except Exception as error:
        pytest.skip(
            f"{kernel_cls.__name__}: cannot build here: "
            f"{type(error).__name__}: {str(error)[:200]}"
        )

    result = probe(module, (x,), check_cuda_graph=False)
    assert result.eager, f"{kernel_cls.__name__}: eager forward failed: {result.eager_error}"
    assert result.reached(POLICY_FLOOR), (
        f"{kernel_cls.__name__}: expected at least {POLICY_FLOOR.name} on this device, "
        f"probe reported {result.summary()} errors={result.errors}"
    )
    assert result.graph_breaks == 0, (
        f"{kernel_cls.__name__}: probe reported {result.graph_breaks} graph breaks "
        f"({result.break_reasons[:3]})"
    )

    if not getattr(kernel_cls, "SUPPORTS_COMPILE", False):
        assert _eager_reason(kernel_cls), (
            f"{kernel_cls.__name__} traces to {result.max_level.name} with 0 graph "
            "breaks, so the static-cache auto-compile is not a pessimization, yet "
            "SUPPORTS_COMPILE is False and no SUPPORTS_COMPILE_EAGER_REASON is "
            "recorded. Opt the kernel in, or record the measured loss (#4984)."
        )


def test_every_compile_capable_kernel_pairs_its_policy_with_a_reason():
    """The static half of the same contract, over every registered kernel."""

    missing = []
    stale = []
    for kernel_cls in protocol_kernels():
        level = compile_level(kernel_cls)
        if level < POLICY_FLOOR:
            continue
        opted_in = bool(getattr(kernel_cls, "SUPPORTS_COMPILE", False))
        reason = _eager_reason(kernel_cls)
        if not opted_in and not reason:
            missing.append(f"{kernel_cls.__name__} ({level.name})")
        if opted_in and reason:
            stale.append(kernel_cls.__name__)

    assert not missing, (
        "these kernels declare a compile level at or above "
        f"{POLICY_FLOOR.name} but keep SUPPORTS_COMPILE=False with no recorded "
        "loss: " + ", ".join(missing)
    )
    assert not stale, (
        "these kernels opt into compiled generation but still carry a stale "
        "SUPPORTS_COMPILE_EAGER_REASON: " + ", ".join(stale)
    )
