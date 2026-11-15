# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""`StopForward` must not outlive the forward that raised it.

`STOP_FORWARD_EXCEPTION` used to be a module-level singleton. Every
`raise STOP_FORWARD_EXCEPTION.with_traceback(None)` cleared the traceback only
*before* propagation, so the singleton kept the traceback from its last raise:
every frame from the worker down to the MoE block's `forward`, and every tensor
those frames owned. On a 2-GPU MoE quantize that pinned a whole block (~1 GiB)
on `cuda:0`. Raising a fresh exception per call removes the process-lifetime
owner; this test pins that with a weakref instead of a memory number.
"""

import gc
import weakref

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.hooked_linear import (
    STOP_FORWARD_EXCEPTION,
    HookedLinear,
    StopForward,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.cuda,
]


def _forward_from_a_frame_holding_a_tensor(module, x, refs):
    # This frame is on the traceback below the raise site. Nothing else keeps
    # the tensor alive, so it dies exactly when the traceback is released.
    tensor = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32, device=x.device)
    refs.append(weakref.ref(tensor))
    module(x)
    return tensor


def test_stop_forward_does_not_pin_the_raising_traceback():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    linear = nn.Linear(8, 8, bias=False, device=device)
    module = HookedLinear.from_linear(linear).to(device)
    module.forward_hook = lambda _module, _inputs, _output: None
    module.forward_hook_last = True
    x = torch.randn(2, 8, device=device)

    refs = []
    try:
        _forward_from_a_frame_holding_a_tensor(module, x, refs)
    except StopForward:
        pass
    finally:
        module.forward_hook = None
        module.forward_hook_last = False

    gc.collect()

    assert STOP_FORWARD_EXCEPTION.__traceback__ is None
    assert refs and refs[0]() is None, (
        "StopForward retained the raising frame and its 256 MiB tensor"
    )
