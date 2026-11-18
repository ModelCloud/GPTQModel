# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A nested ``torch_compile`` must not write dynamo config while tracing (#5224).

``torch.compile`` is lazy, so the wrapper ``torch_compile`` installs can be
entered from inside another compile's trace: transformers compiles a
static-cache ``generate()`` forward, and ``VPTQLinear.optimize()`` installs one
of these wrappers on ``dequantize`` at post-init.

#5163 made that wrapper enable ``torch._dynamo.config.suppress_errors`` for the
call. ``torch._dynamo.config`` overrides are ContextVars, so the write is an
``Unsupported: ContextVar mutation not supported`` graph break when the wrapper
is traced. Every VPTQ layer then breaks out of the outer graph, and the
cudagraph static-cache decode hands out outputs the next partition overwrote
(``accessing tensor output of CUDAGraphs that has been overwritten``).

A nested call must stay read-only: the outer trace owns error handling and the
eager body is inlined instead. These tests are red before the fix.
"""

import pytest
import torch

import gptqmodel.utils.torch as torch_utils
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear


pytestmark = [pytest.mark.cpu]


def _require_dynamo():
    return pytest.importorskip("torch._dynamo")


def _reset_dynamo():
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()


def _graph_breaks():
    from torch._dynamo.utils import counters

    return sum(counters["graph_break"].values())


def _build_vptq() -> VPTQLinear:
    return VPTQLinear(
        bits=2,
        in_features=64,
        out_features=64,
        bias=False,
        vector_lens=[-1, 4],
        num_centroids=[-1, 16],
        num_res_centroids=[-1, -1],
        group_size=64,
        register_buffers=True,
        dtype=None,
    )


def test_nested_callable_compile_is_inlined_into_the_outer_graph():
    """The #5224 mechanism without a model: no graph break, same result."""

    _require_dynamo()
    _reset_dynamo()
    inner = torch_utils.torch_compile(lambda t: t * 2 + 1, backend="eager")
    outer = lambda t: inner(t).sum()  # noqa: E731 - the traced callable

    value = torch.randn(8)
    eager = float(outer(value))
    got = float(torch.compile(outer, fullgraph=True, backend="eager")(value))

    assert got == pytest.approx(eager)
    assert _graph_breaks() == 0


def test_nested_module_compile_is_inlined_into_the_outer_graph():
    """``torch_compile(nn.Module)`` takes the same nested path."""

    _require_dynamo()
    _reset_dynamo()

    class Scale(torch.nn.Module):
        def forward(self, value):
            return value * 2 + 1

    inner = torch_utils.torch_compile(Scale(), backend="eager")
    outer = lambda t: inner(t).sum()  # noqa: E731 - the traced callable

    value = torch.randn(8)
    eager = float(outer(value))
    got = float(torch.compile(outer, fullgraph=True, backend="eager")(value))

    assert got == pytest.approx(eager)
    assert _graph_breaks() == 0


def test_vptq_optimize_installs_a_trace_safe_dequantize():
    """The kernel that exposed the bug: ``optimize()`` then a model-level trace."""

    _require_dynamo()
    _reset_dynamo()
    layer = _build_vptq()
    layer.optimize()
    assert layer.optimized

    value = torch.randn(4, 64)
    eager = layer(value)
    got = torch.compile(lambda t: layer(t), fullgraph=True, backend="eager")(value)

    assert torch.allclose(eager, got)
    assert _graph_breaks() == 0


def test_top_level_non_fullgraph_call_still_enables_suppression():
    """#5163's contract: a user-facing non-fullgraph call suppresses errors."""

    _require_dynamo()
    _reset_dynamo()
    seen = {}

    def backend(graph_module, _example_inputs):
        seen["suppress_errors"] = torch._dynamo.config.suppress_errors
        return graph_module.forward

    before = torch._dynamo.config.suppress_errors
    compiled = torch_utils.torch_compile(lambda value: value * 2, backend=backend)
    assert float(compiled(torch.ones(2)).sum()) == 4.0
    assert seen["suppress_errors"] is True
    assert torch._dynamo.config.suppress_errors is before


def test_nested_call_does_not_change_the_process_suppression_flag():
    """Nested tracing must leave the ContextVar exactly as it found it."""

    _require_dynamo()
    _reset_dynamo()
    inner = torch_utils.torch_compile(lambda t: t * 2 + 1, backend="eager")
    outer = lambda t: inner(t).sum()  # noqa: E731 - the traced callable

    before = torch._dynamo.config.suppress_errors
    torch.compile(outer, fullgraph=True, backend="eager")(torch.randn(8))
    assert torch._dynamo.config.suppress_errors is before


def test_compiled_model_forward_proxy_is_trace_clean_inside_an_outer_compile():
    """``optimize()``'s proxy must not collect garbage inside an outer trace.

    transformers' static-cache ``generate()`` wraps ``model.__call__``, so the
    proxy's ``finally`` GC runs while dynamo is tracing. ``gc.collect`` is
    untraceable; the release is deferred to the next eager call instead.
    """

    _require_dynamo()
    _reset_dynamo()
    from gptqmodel.models.base import _CompiledModelForward

    class Scale(torch.nn.Module):
        def forward(self, value):
            return value * 2

    module = Scale()
    compiled_forward = torch_utils.torch_compile(type(module).forward, backend="eager")
    proxy = _CompiledModelForward(compiled_forward, type(module).forward, module)

    value = torch.randn(4)
    eager = proxy(value)
    got = torch.compile(
        lambda tensor: proxy(tensor), fullgraph=True, backend="eager"
    )(value)

    assert torch.allclose(eager, got)
    assert _graph_breaks() == 0
