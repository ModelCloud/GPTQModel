# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""TorchFP8Linear cache consistency for the block-decode plan and range facts (#5592).

Two defects shared one root cause: the plan published its key and value as two
separate attributes (a torn read under ``PYTHON_GIL=0``), and the decoded-weight
bound ignored the inverse-scale payload, so a rebind or an in-place ``copy_``
kept a stale FP16 range answer and let an out-of-range payload take the
range-unsafe Triton path.
"""

import sys
import threading

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from tests.fp8_test_utils import pack_fp8_module


def _block_module(rows=256, cols=256) -> TorchFP8Linear:
    torch.manual_seed(11)
    linear = torch.nn.Linear(cols, rows, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(0.0, 0.02)
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=cols,
        out_features=rows,
        bias=False,
        register_buffers=False,
        format="float8_e4m3fn",
        weight_scale_method="block",
        weight_block_size=(128, 128),
    )
    pack_fp8_module(module, linear)
    return module


def _replace_scale(module: TorchFP8Linear, mutation: str) -> None:
    fresh = torch.full_like(module.weight_scale_inv, 1e-3)
    if mutation == "setattr":
        module.weight_scale_inv = fresh
    elif mutation == "buffers":
        module._buffers["weight_scale_inv"] = fresh
    elif mutation == "copy_":
        module.weight_scale_inv.copy_(fresh)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(mutation)


def test_block_decode_plan_is_one_immutable_tuple():
    module = _block_module()
    module._block_decode_plan(torch.device("cpu"), torch.float16)

    plan = module._fp8_block_plan
    assert isinstance(plan, tuple) and len(plan) == 2
    key, value = plan
    assert module._fp8_block_plan_key == key
    assert value == (False, False)

    # The historical test hook still forces a recompute.
    module._fp8_block_plan_key = None
    assert module._fp8_block_plan is None

    with pytest.raises(AttributeError):
        module._fp8_block_plan_key = ("not", "derived")


@pytest.mark.parametrize("mutation", ["setattr", "buffers", "copy_"])
def test_range_bound_tracks_a_changed_scale_payload(mutation):
    module = _block_module()
    module.post_init()
    assert module._weights_fit_activation_dtype(torch.float16)
    assert module._inverse_scales_are_nonzero()

    _replace_scale(module, mutation)

    # 448 / 1e-3 = 448000 > 65504, so FP16 must not narrow the decoded weight.
    assert not module._weights_fit_activation_dtype(torch.float16)
    assert module._inverse_scales_are_nonzero()
    x = torch.zeros(2, module.in_features, dtype=torch.float16)
    assert not module._native_gemm_eligible(x, device_type="cuda")

    # A fresh invalidation agrees with the recomputed answer.
    module._invalidate_compiled()
    assert not module._weights_fit_activation_dtype(torch.float16)
    assert not module._native_gemm_eligible(x, device_type="cuda")


@pytest.mark.parametrize("mutation", ["setattr", "buffers", "copy_"])
def test_zero_inverse_scale_is_detected_after_a_change(mutation):
    module = _block_module()
    module.post_init()
    assert module._inverse_scales_are_nonzero()

    if mutation == "copy_":
        module.weight_scale_inv.zero_()
    else:
        zeros = torch.zeros_like(module.weight_scale_inv)
        if mutation == "setattr":
            module.weight_scale_inv = zeros
        else:
            module._buffers["weight_scale_inv"] = zeros

    assert not module._inverse_scales_are_nonzero()
    assert not module._weights_fit_activation_dtype(torch.float16)


def test_range_bound_ready_tracks_the_payload():
    module = _block_module()
    assert not module._decoded_weight_bound_ready

    module._decoded_weight_absmax_bound()
    assert module._decoded_weight_bound_ready

    module.weight_scale_inv = module.weight_scale_inv.clone()
    assert not module._decoded_weight_bound_ready


def test_block_decode_plan_recomputes_after_a_scale_change(monkeypatch):
    module = _block_module()
    calls = []
    original = module._compute_block_decode_plan

    def counting(**kwargs):
        calls.append(kwargs["dtype"])
        return original(**kwargs)

    monkeypatch.setattr(module, "_compute_block_decode_plan", counting)
    device = torch.device("cpu")

    assert module._block_decode_plan(device, torch.float16) == (False, False)
    assert len(calls) == 1
    module._block_decode_plan(device, torch.float16)
    assert len(calls) == 1

    module.weight_scale_inv.copy_(torch.full_like(module.weight_scale_inv, 1e-3))
    module._block_decode_plan(device, torch.float16)
    assert len(calls) == 2


def test_block_decode_plan_recomputes_after_a_scale_rebind(monkeypatch):
    module = _block_module()
    calls = []
    original = module._compute_block_decode_plan
    monkeypatch.setattr(
        module,
        "_compute_block_decode_plan",
        lambda **kwargs: calls.append(kwargs["dtype"]) or original(**kwargs),
    )
    device = torch.device("cpu")

    module._block_decode_plan(device, torch.float16)
    module.weight_scale_inv = module.weight_scale_inv.clone()
    module._block_decode_plan(device, torch.float16)
    assert len(calls) == 2


@pytest.mark.skipif(
    not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled(),
    reason="torn-publish regression needs a free-threaded build (PYTHON_GIL=0)",
)
def test_block_decode_plan_never_pairs_a_key_with_a_stale_value(monkeypatch):
    module = _block_module()
    plans = {torch.float16: (False, False), torch.bfloat16: (True, True)}
    monkeypatch.setattr(
        module,
        "_compute_block_decode_plan",
        lambda *, device, dtype: plans[dtype],
    )
    device = torch.device("cpu")
    errors = []

    def worker(dtype):
        for _ in range(20000):
            observed = module._block_decode_plan(device, dtype)
            if observed != plans[dtype]:
                errors.append((dtype, observed))

    threads = [
        threading.Thread(target=worker, args=(dtype,))
        for dtype in (torch.float16, torch.bfloat16)
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors[:5]
