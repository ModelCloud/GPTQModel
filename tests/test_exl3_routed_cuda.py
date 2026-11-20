# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

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


@pytest.mark.parametrize("scratch_dtype", [torch.float16, torch.float32])
@torch.inference_mode()
def test_routed_q4nx_keeps_gate_up_partials_wide_through_output_scale(scratch_dtype):
    device = torch.device("cuda")
    width, bits = 128, 3.5
    trellis = torch.zeros((8, 8, 56), dtype=torch.int16, device=device)
    sign = torch.ones(width, dtype=torch.float16, device=device)
    up_scale = torch.full_like(sign, 2**-14)
    down_scale = torch.full_like(sign, 1 / width)
    up_bias = torch.zeros_like(sign)
    up_bias[0] = -64
    down_bias = torch.full_like(sign, 0.375)
    down_bias[0] = 1.125

    def table(t):
        return torch.tensor([t.data_ptr()], dtype=torch.int64, device=device)

    up_parts = tuple(map(table, (trellis, sign, up_scale)))
    down_parts = tuple(map(table, (trellis, sign, down_scale)))
    up_bias_ptr, down_bias_ptr = table(up_bias), table(down_bias)
    selected = torch.zeros((1, 1), dtype=torch.int64, device=device)
    routing = torch.full((1, 1), 0.5, dtype=torch.float16, device=device)
    x = torch.full((1, width), -4096, dtype=torch.float16, device=device)
    had_gate, had_up, activated = (torch.empty_like(x) for _ in range(3))
    gate_out, up_out = (torch.empty_like(x, dtype=scratch_dtype) for _ in range(2))
    down_out, out = (torch.empty_like(x, dtype=torch.float32) for _ in range(2))
    counters = torch.zeros((7,), dtype=torch.int32, device=device)
    native = ext._runtime_op("exl3_moe_coop")

    def run():
        native(x, selected, routing, -1, -1, width, *up_parts, *up_parts, *down_parts,
               None, up_bias_ptr, down_bias_ptr, bits, bits, bits, False, False, 2, 0, False,
               had_gate, had_up, gate_out, up_out, activated, down_out, counters, out, None, None, q4nx=True)

    def expected(fill):
        # Constant Q4NX codewords make each Hadamard block a DC projection.
        # Model the two specified input-rotation casts and use FP64 elsewhere.
        root = math.sqrt(width)
        code = -15 / math.sqrt(42.5)
        up_dc = torch.tensor(fill * root, dtype=torch.float16).item() * code
        up = up_dc * root * up_scale[0].item() + up_bias[0].item()
        down_input = torch.tensor(max(up, 0) ** 2 / root, dtype=torch.float16).item()
        result = down_bias.double().unsqueeze(0) * routing.item()
        result[0, 0] += down_input * code * root * routing.item()
        return result.float()

    run()
    torch.testing.assert_close(out, expected(-4096), rtol=1e-6, atol=1e-6)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    x.fill_(-2048)
    graph.replay()
    torch.testing.assert_close(out, expected(-2048), rtol=1e-6, atol=1e-6)
