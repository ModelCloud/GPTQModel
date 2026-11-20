# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.exllamav3 import ext
from tests.test_exllamav3_torch_padding import _oracle_inner


pytestmark = pytest.mark.skipif(
    torch.version.hip is not None or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (8, 0),
    reason="requires NVIDIA SM80 or newer",
)


def _fixture(op, bits, *, q4nx=False):
    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(117)
    rows, width, experts, topk = 2, 128, 3, 2
    x = torch.randn((rows, width), dtype=torch.float16, device=device, generator=gen) * 0.04
    ids = torch.tensor([[0, 1], [1, experts]], dtype=torch.int64, device=device)
    routing = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float16, device=device)
    tables, projection_weights, packed_owners = [], [], []
    states = torch.arange(width, device=device)
    parity = states[:, None] & states[None, :]
    signs = torch.ones_like(parity)
    for bit in range(7):
        signs = signs * (1 - 2 * ((parity >> bit) & 1))
    hadamard = signs.double() / (width ** 0.5)
    for _ in range(3):
        trellises, su, sv = [], [], []
        dense = []
        for _ in range(experts):
            packed = torch.randint(-32768, 32768, (8, 8, int(bits * 16)),
                                   dtype=torch.int16, device=device, generator=gen)
            sign = (2 * torch.randint(0, 2, (width,), generator=gen, device=device) - 1).half()
            scale = sign * 0.125
            trellises.append(packed)
            su.append(sign)
            sv.append(scale)
            if q4nx:
                w = _oracle_inner({"trellis": packed, "q4nx": True}).to(device)
            else:
                w = torch.empty((width, width), device=device, dtype=torch.float16)
                ext.exllamav3_ext.reconstruct(w, packed, bits, False, True)
            dense.append((sign.double()[:, None] * hadamard) @ w.double() @ (hadamard * scale.double()))
        projection_weights.append(dense)
        tables.append(tuple(torch.tensor([item.data_ptr() for item in column], device=device)
                            for column in (trellises, su, sv)))
        # The packed tensors own the addresses embedded in the native tables.
        packed_owners.append((trellises, su, sv))

    slots = rows * topk
    shape = (slots, width)
    had_g, had_u, activation = (torch.empty(shape, device=device, dtype=torch.float16) for _ in range(3))
    gu_g, gu_u, down = (torch.empty(shape, device=device, dtype=torch.float32) for _ in range(3))
    counters = torch.zeros((3 * slots + rows + 3,), dtype=torch.int32, device=device)
    out = torch.empty((rows, width), dtype=torch.float32, device=device)

    def run():
        assert packed_owners
        op(x, ids, routing, -1, -1, width,
           *tables[0], *tables[1], *tables[2], None, None, None,
           bits, bits, bits, False, not q4nx, 0, 0.0, True,
           had_g, had_u, gu_g, gu_u, activation, down, counters, out, None, None, q4nx=q4nx)
        return out

    def reference(inputs):
        result = torch.zeros_like(out, dtype=torch.float64)
        for row, selected in enumerate(((0, 1), (1,))):
            for slot, expert in enumerate(selected):
                gate, up, d = (weights[expert] for weights in projection_weights)
                g = inputs[row].double() @ gate
                u = inputs[row].double() @ up
                a = torch.nn.functional.silu(g) * u
                result[row] += (a @ d) * routing[row, slot]
        return result.float()

    return x, run, reference


@pytest.mark.parametrize("bits", (3, 2.5, 3.5))
@pytest.mark.parametrize("q4nx", (False, True))
@torch.inference_mode()
def test_routed_exl3_matches_torch_and_replays_changed_input(bits, q4nx):
    op = ext._runtime_op("exl3_moe_coop")
    x, run, reference = _fixture(op, bits, q4nx=q4nx)
    initial = x.clone()
    changed = -x
    eager = run().clone()
    expected = reference(initial)
    torch.testing.assert_close(eager, expected, atol=2e-3, rtol=2e-3)
    assert (eager - expected).norm() <= 2e-3 * expected.norm()
    assert eager.abs().max() > 1e-3
    stream = torch.cuda.Stream(device=x.device)
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        run()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    assert torch.equal(captured, eager)
    x.copy_(changed)
    graph.replay()
    torch.testing.assert_close(captured, reference(changed), atol=2e-3, rtol=2e-3)
    assert not torch.equal(captured, eager)


@torch.inference_mode()
def test_routed_exl3_uses_independent_scratch_for_live_graphs():
    op = ext._runtime_op("exl3_moe_coop")
    fixtures = [_fixture(op, 3), _fixture(op, 3)]
    main_stream = torch.cuda.current_stream()
    graphs = []
    for index, (x, run, _) in enumerate(fixtures, start=1):
        x.mul_(index)
        expected = run().clone()
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(main_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            run()
            with torch.cuda.graph(graph, stream=stream):
                output = run()
        graphs.append((stream, graph, output, expected))
    for _ in range(4):
        for stream, graph, _, _ in graphs:
            with torch.cuda.stream(stream):
                graph.replay()
    for stream, _, output, expected in graphs:
        main_stream.wait_stream(stream)
        assert torch.equal(output, expected)
