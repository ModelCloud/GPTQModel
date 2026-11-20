# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cache clear must not retire a packed executor that a forward is using (#5849).

The fused projection group publishes its guard, executor and operand bindings
without synchronization. Two cold callers could both rebuild, and an explicit
source cache clear could drop ``packed_forward`` (or clear its storage) between
a successful guard and the executor call. The controlled case parks a real
parent forward immediately after its guard, clears one source cache on the main
thread, and resumes it; the paired barrier case repeats the same clear before
two concurrent forwards.
"""

import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn

from gptqmodel.models.fusion import FusionDecision, _FusionGroup
from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.quantization import FORMAT


IN_FEATURES, OUT_FEATURES, GROUP_SIZE = 256, 64, 32
ORACLE = 128.0


def _projection():
    layer = TorchInt8Linear(
        bits=8,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        format=FORMAT.GPTQ_V2,
    ).eval()
    layer.qweight.fill_(-2122219135)  # four copies of code 129 in each word
    layer.qzeros.fill_(-2139062144)  # four copies of zero point 128 in each word
    layer.scales.fill_(0.25)
    layer.g_idx.copy_(torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE)
    layer.qzero_format(2)
    layer.post_init()
    return layer


class _Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = _projection()
        self.up_proj = _projection()

    def forward(self, x):
        return self.gate_proj(x) + self.up_proj(x)


@pytest.fixture(autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _fixture():
    parent = _Pair().eval()
    group = _FusionGroup(
        FusionDecision("test", "", ("gate_proj", "up_proj")),
        parent,
        [parent.gate_proj, parent.up_proj],
        activation_dtype=torch.float32,
    )
    group.install()
    assert group.packed is not None
    return parent, group


def _inputs():
    x = torch.ones((1, IN_FEATURES), dtype=torch.float32)
    return x, torch.full((1, OUT_FEATURES), ORACLE)


def _parked_guard(original, ready, resume):
    # Bind the events as defaults so the compiled guard needs no test-module
    # globals; retirement clears the function's private namespace.
    def parked(_original=original, _ready=ready, _resume=resume):
        if not _original():
            return False
        _ready.set()
        if not _resume.wait(timeout=10):
            raise RuntimeError("controlled interleaving was not resumed")
        return True

    return types.FunctionType(parked.__code__, {}, "parked_guard", parked.__defaults__)


def test_cache_clear_between_guard_and_executor_keeps_the_live_executor():
    parent, group = _fixture()
    x, expected = _inputs()
    ready = threading.Event()
    resume = threading.Event()
    group.guard = _parked_guard(group.guard, ready, resume)
    result = {}

    def call():
        try:
            result["output"] = parent(x)
        except BaseException as error:  # noqa: BLE001 - re-raised on the test thread
            result["error"] = error

    worker = threading.Thread(target=call, name="parked-forward")
    worker.start()
    try:
        assert ready.wait(timeout=10), "forward never reached the post-guard park"
        parent.gate_proj.clear_weight_cache()
    finally:
        resume.set()
    worker.join(timeout=10)
    assert not worker.is_alive(), "forward did not finish after the cache clear"
    if "error" in result:
        raise result["error"]
    torch.testing.assert_close(result["output"], expected, rtol=0, atol=0)
    # The next forward rebuilds from the cleared cache and must still be exact.
    torch.testing.assert_close(parent(x), expected, rtol=0, atol=0)


def test_paired_forwards_after_cache_clear_are_exact_and_do_not_raise():
    parent, _group = _fixture()
    x, expected = _inputs()
    failures = []
    workers = ThreadPoolExecutor(max_workers=2)
    try:
        for index in range(50):
            parent.gate_proj.clear_weight_cache()
            start = threading.Barrier(2)

            def call(_index=index):
                start.wait(timeout=10)
                return parent(x)

            jobs = [workers.submit(call) for _ in range(2)]
            for job in jobs:
                try:
                    torch.testing.assert_close(job.result(timeout=10), expected, rtol=0, atol=0)
                except BaseException as error:  # noqa: BLE001 - aggregate every pair
                    failures.append((index, error))
    finally:
        workers.shutdown(wait=True, cancel_futures=True)
    assert not failures, failures[:3]
