# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTQProcessor must not hold the task Hessian across ``g.quantize()`` (#5026).

GPTAQ/FOEM release the task Hessian themselves and then rebind ``H`` to the
activation-order permuted copy. A local ``g_h`` in ``process()`` therefore kept
the original unpermuted ``[C, C]`` fp32 Hessian alive through the whole
quantize, including the peak where ``H``, ``dXXT`` and the inverse are all live:
784 MiB per dense Hessian on an 8B down_proj.

The probe below runs the real ``process()`` preamble and stops inside
``quantize()``, where it records how many strong references the task Hessian
has. One reference is ``task.H`` itself; anything more means the processor is
holding a copy of its own.
"""

import sys
import threading
from types import SimpleNamespace

import torch

import gptqmodel.looper.gptq_processor as gptq_processor


class _StopInQuantize(Exception):
    """Ends the probe once the reference count has been recorded."""


def test_process_does_not_hold_the_task_hessian():
    module = SimpleNamespace(
        name="proj",
        full_name="model.layers.0.proj",
        target_device=torch.device("cpu"),
        state={},
        module=SimpleNamespace(
            weight=torch.zeros(8, 8),
            target_device=torch.device("cpu"),
        ),
    )
    observed = {}

    def quantize():
        # One reference is ``task.H``; the argument to ``getrefcount`` is the
        # second. A local in ``process()`` would make it three.
        observed["refs"] = sys.getrefcount(task.H)
        raise _StopInQuantize()

    task = SimpleNamespace(
        H=torch.zeros(8, 8),
        module=module.module,
        quantize=quantize,
    )

    processor = gptq_processor.GPTQProcessor.__new__(gptq_processor.GPTQProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {module.name: task}
    processor.draw_progress = lambda *args, **kwargs: None

    try:
        processor.process(module)
    except _StopInQuantize:
        pass
    else:  # pragma: no cover - the probe must stop inside quantize()
        raise AssertionError("process() did not reach quantize()")

    assert observed["refs"] == 2, (
        "GPTQProcessor kept an extra reference to the task Hessian across "
        "g.quantize(); GPTAQ/FOEM then cannot free the unpermuted copy (#5026)"
    )
