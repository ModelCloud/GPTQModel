# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUF Torch picks fused-vs-dense statically, so results survive process load.

`GGUFTorchLinear` used to time `_forward_fused_k` against
`_forward_dequant_matmul` per `(device, dtype, shape)` and cache the winner.
The two paths are not bitwise equivalent -- on CPU the fused reduction runs
inside `cpu_bf16x3_matmul()`, the dense one is IEEE FP32 -- so the logits a run
produced depended on what else the process was doing at the first forward
(#4590). A second thread doing unrelated `matmul` work was enough to move one
to three of 168 Qwen2.5-0.5B modules onto the other path and change the output.

These tests pin the replacement contract:

* the decision is a pure function of the module's static shape gates,
* identical work in a second thread cannot change it,
* a fresh process reaches the same decision and the same bytes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear


def _build(in_features: int, out_features: int, bits: str = "q4_k_m", seed: int = 0) -> GGUFTorchLinear:
    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.02)
    module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    ).eval()
    module.pack_original(linear, scales=torch.empty(0), zeros=torch.empty(0), g_idx=None)
    module.post_init()
    module.gguf_fused_cpu_max_rows = 32
    module.gguf_fused_cpu_min_matrix_elements = 0
    return module


def _run_with_background_load(module: GGUFTorchLinear, x: torch.Tensor, *, busy: bool) -> tuple[torch.Tensor, bool]:
    stop = threading.Event()
    thread = None
    if busy:
        def spin() -> None:
            # The product has to feed the next iteration: a bare `work @ work`
            # is a no-effect statement, and binding the result keeps the same
            # continuous matmul load while giving the loop a real dependency.
            work = torch.randn(128, 128)
            while not stop.is_set():
                work = work @ work

        thread = threading.Thread(target=spin)
        thread.start()
    try:
        with torch.inference_mode():
            module(x)  # first forward selects the path
            return module(x).clone(), module._should_use_fused_k_forward(x)
    finally:
        stop.set()
        if thread is not None:
            thread.join()


def test_repeated_first_forward_under_background_load_selects_one_path():
    """A busy neighbour thread must not move modules between the two paths."""

    decisions = set()
    outputs = []
    for busy in (False, True, False, True):
        module = _build(256, 256, seed=11)
        # Seven rows of bf16 activations: the shape that the reported CPU repro
        # drove onto the fused path.
        activation = torch.randn(
            7, module.in_features, generator=torch.Generator().manual_seed(1), dtype=torch.bfloat16
        )
        output, fused = _run_with_background_load(module, activation, busy=busy)
        decisions.add(fused)
        outputs.append(output)

    assert decisions == {True}
    for candidate in outputs[1:]:
        assert torch.equal(outputs[0], candidate)


def test_decision_does_not_depend_on_call_order_or_shape_history():
    module = _build(256, 256, seed=13)
    generator = torch.Generator().manual_seed(17)
    shapes = [(1,), (7,), (33,), (7,), (1,), (40,)]

    first_pass = []
    for (rows,) in shapes:
        x = torch.randn(rows, module.in_features, generator=generator, dtype=torch.bfloat16)
        first_pass.append(module._should_use_fused_k_forward(x))

    # Asking again in a different order must return the same per-shape answer.
    second_pass = {}
    for (rows,) in reversed(shapes):
        x = torch.randn(rows, module.in_features, generator=torch.Generator().manual_seed(rows), dtype=torch.bfloat16)
        second_pass[rows] = module._should_use_fused_k_forward(x)

    assert first_pass == [second_pass[rows] for (rows,) in shapes]
    assert second_pass[33] is False
    assert second_pass[40] is False


_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import torch
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear

    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 256, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.02)
    module = GGUFTorchLinear(
        bits="q4_k_m", group_size=-1, sym=True, desc_act=False,
        in_features=256, out_features=256, bias=False, register_buffers=True,
    ).eval()
    module.pack_original(linear, scales=torch.empty(0), zeros=torch.empty(0), g_idx=None)
    module.post_init()
    module.gguf_fused_cpu_max_rows = 32
    module.gguf_fused_cpu_min_matrix_elements = 0

    x = torch.randn(7, 256, generator=torch.Generator().manual_seed(5), dtype=torch.bfloat16)
    with torch.inference_mode():
        module(x)
        out = module(x)
    print("RESULT" + json.dumps({
        "fused": module._should_use_fused_k_forward(x),
        "checksum": float(out.double().sum().item()),
    }))
    """
)


@pytest.mark.timeout(300)
def test_two_processes_produce_the_same_path_and_bytes():
    repo_root = Path(__file__).resolve().parents[2]
    results = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", _SCRIPT],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert completed.returncode == 0, completed.stderr[-4000:]
        line = next(
            (entry for entry in completed.stdout.splitlines() if entry.startswith("RESULT")),
            None,
        )
        assert line is not None, completed.stdout[-2000:]
        results.append(json.loads(line[len("RESULT"):]))

    assert results[0]["fused"] is True
    assert results[0] == results[1]
