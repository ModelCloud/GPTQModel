# SPDX-License-Identifier: Apache-2.0
"""GPU acceptance tests; no mocked arithmetic or allocator in this file."""
import pytest
import torch

from gptqmodel.utils import marlin as marlin_utils
from gptqmodel.utils.marlin_scratch import MarlinScratchContext
from scripts.marlin_scratch_fixture import gemm_arguments, make_layer

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")]


def require_dtype(dtype, method="gptq"):
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (7, 5) or ((dtype == torch.bfloat16 or method == "awq") and major < 8):
        pytest.skip("Unsupported Marlin device/dtype combination")
    if not marlin_utils.marlin_runtime_available(dtype):
        pytest.fail(marlin_utils.marlin_runtime_error(dtype))


def scratch(args):
    c_count, a_count = marlin_utils.marlin_scratch_sizes(
        args["a"], args["size_m"], args["size_k"], args["use_fp32_reduce"],
        bool(args["g_idx"].numel() and args["perm"].numel()))
    return (torch.empty(c_count, dtype=torch.float32, device=args["a"].device),
            torch.empty(a_count, dtype=args["a"].dtype, device=args["a"].device))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("method,act_order,bits,k,n", [
    ("gptq", False, 4, 256, 128), ("gptq", True, 4, 256, 128),
    ("gptq", True, 8, 256, 128), ("gptq", False, 8, 288, 200),
    ("awq", False, 4, 256, 128), ("awq", False, 4, 288, 200),
    ("awq", False, 8, 256, 128)])
@pytest.mark.parametrize("fp32", [True, False])
def test_scratch_matches_independent_reference(dtype, method, act_order, bits, k, n, fp32):
    require_dtype(dtype, method)
    layer, dense, bias = make_layer(method, dtype, k, n, act_order=act_order, bits=bits)
    # Prepare maximum capacity once, then deliberately vary M and dirty contents.
    max_args = gemm_arguments(layer, torch.zeros(64, k, device="cuda", dtype=dtype), fp32=fp32)
    c_tmp, a_tmp = scratch(max_args)
    for m in (1, 8, 16, 17, 32, 64, 2, 0):
        x = torch.randn(1, 1, m, k, device="cuda", dtype=dtype) / k**.5
        args = gemm_arguments(layer, x, fp32=fp32)
        c_tmp.fill_(float("nan"))
        a_tmp.fill_(float("nan"))
        old = marlin_utils.gptq_marlin_gemm(**args)[:, :n].reshape(1, 1, m, n)
        actual = marlin_utils.gptq_marlin_gemm(**args, c_tmp=c_tmp, a_tmp=a_tmp)[:, :n].reshape(1, 1, m, n)
        expected = x @ dense + bias
        torch.testing.assert_close(actual, expected, rtol=.05, atol=.05)
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
        assert actual.data_ptr() != old.data_ptr() or m == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("target", ["c_tmp", "a_tmp"])
@pytest.mark.parametrize("fault", ["dtype", "device", "capacity", "noncontiguous", "alignment", "alias"])
def test_invalid_scratch_rejected(dtype, target, fault):
    require_dtype(dtype)
    layer, _, _ = make_layer("gptq", dtype, act_order=True)
    args = gemm_arguments(layer, torch.randn(17, 256, device="cuda", dtype=dtype))
    c_tmp, a_tmp = scratch(args)
    good = c_tmp if target == "c_tmp" else a_tmp
    length = good.numel()
    if fault == "dtype":
        bad = good.to(torch.int32)
    elif fault == "device":
        bad = good.cpu()
    elif fault == "capacity":
        bad = good[:-1]
    elif fault == "noncontiguous":
        bad = torch.empty(length * 2, dtype=good.dtype, device=good.device)[::2]
    elif fault == "alignment":
        bad = torch.empty(length + 1, dtype=good.dtype, device=good.device)[1:]
    else:
        # Ensure capacity is sufficient, so alias validation is what fails.
        storage = torch.empty(max(length * good.element_size(), args["a"].numel() * dtype.itemsize),
                              device="cuda", dtype=torch.uint8)
        args["a"] = storage[:args["a"].numel() * dtype.itemsize].view(dtype).reshape_as(args["a"])
        bad = storage[:length * good.element_size()].view(good.dtype)
    buffers = dict(c_tmp=c_tmp, a_tmp=a_tmp)
    buffers[target] = bad
    with pytest.raises(RuntimeError, match=target):
        marlin_utils.gptq_marlin_gemm(**args, **buffers)


@pytest.mark.parametrize("target", ["c_tmp", "a_tmp"])
@pytest.mark.parametrize("alias", ["b_q_weight", "b_scales", "c", "workspace", "other_scratch"])
def test_scratch_alias_with_other_arguments_rejected(target, alias):
    require_dtype(torch.float16)
    layer, _, _ = make_layer("gptq", torch.float16, act_order=True)
    args = gemm_arguments(layer, torch.randn(17, 256, device="cuda", dtype=torch.float16))
    args["c"] = torch.empty(17, 128, device="cuda", dtype=torch.float16)
    c_tmp, a_tmp = scratch(args)
    buffers = dict(c_tmp=c_tmp, a_tmp=a_tmp)
    good = buffers[target]
    if alias == "other_scratch":
        name = "a_tmp" if target == "c_tmp" else "c_tmp"
        other = buffers[name]
    else:
        name, other = alias, args[alias]
    byte_count = max(good.numel() * good.element_size(), other.numel() * other.element_size())
    storage = torch.empty(byte_count, device="cuda", dtype=torch.uint8)
    shared_other = storage[:other.numel() * other.element_size()].view(other.dtype).reshape_as(other)
    shared_other.copy_(other)
    if alias == "other_scratch":
        buffers[name] = shared_other
    else:
        args[name] = shared_other
    buffers[target] = storage[:good.numel() * good.element_size()].view(good.dtype)
    with pytest.raises(RuntimeError, match="c_tmp|a_tmp"):
        marlin_utils.gptq_marlin_gemm(**args, **buffers)


def test_disabled_scratch_requirements_and_validation():
    require_dtype(torch.float16)
    layer, _, _ = make_layer("gptq", torch.float16)
    args = gemm_arguments(layer, torch.randn(8, 256, device="cuda", dtype=torch.float16), fp32=False)
    c_tmp, a_tmp = scratch(args)
    assert c_tmp.numel() == a_tmp.numel() == 0
    marlin_utils.gptq_marlin_gemm(**args, c_tmp=c_tmp, a_tmp=a_tmp)
    with pytest.raises(RuntimeError, match="c_tmp"):
        marlin_utils.gptq_marlin_gemm(**args, c_tmp=torch.empty(0, device="cuda", dtype=torch.int32))


@pytest.mark.parametrize("target", ["c_tmp", "a_tmp"])
def test_scratch_from_other_cuda_device_rejected(target):
    require_dtype(torch.float16)
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices required for wrong-device validation")
    layer, _, _ = make_layer("gptq", torch.float16, act_order=True)
    args = gemm_arguments(layer, torch.randn(17, 256, device="cuda:0", dtype=torch.float16))
    c_tmp, a_tmp = scratch(args)
    buffers = dict(c_tmp=c_tmp, a_tmp=a_tmp)
    buffers[target] = buffers[target].to("cuda:1")
    with pytest.raises(RuntimeError, match=target):
        marlin_utils.gptq_marlin_gemm(**args, **buffers)


def test_raw_act_order_permutation_capacity_uses_padded_k():
    require_dtype(torch.float16)
    # The layer API intentionally excludes padded act-order. Raw GEMM can
    # execute an already-aligned 320-wide fixture after caller-side padding.
    layer, dense, bias = make_layer("gptq", torch.float16, k=320, n=256, act_order=True)
    x = torch.randn(17, 288, device="cuda", dtype=torch.float16) / 16
    padded = torch.nn.functional.pad(x, (0, 32))
    args = gemm_arguments(layer, padded)
    c_tmp, a_tmp = scratch(args)
    assert a_tmp.numel() == 17 * 320
    result = marlin_utils.gptq_marlin_gemm(**args, c_tmp=c_tmp, a_tmp=a_tmp)
    torch.testing.assert_close(result, padded @ dense + bias, rtol=.05, atol=.05)
    short = torch.empty(17 * 288, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="a_tmp"):
        marlin_utils.gptq_marlin_gemm(**args, c_tmp=c_tmp, a_tmp=short)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_explicit_scratch_graph_repeated_replay(dtype):
    require_dtype(dtype)
    layer, dense, bias = make_layer("gptq", dtype, act_order=True)
    x = torch.randn(17, 256, device="cuda", dtype=dtype) / 16
    args = gemm_arguments(layer, x)
    c_tmp, a_tmp = scratch(args)
    # Compile and warm up outside capture. This graph exclusively owns all buffers.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            marlin_utils.gptq_marlin_gemm(**args, c_tmp=c_tmp, a_tmp=a_tmp)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = marlin_utils.gptq_marlin_gemm(**args, c_tmp=c_tmp, a_tmp=a_tmp)
    for _ in range(10):
        x.copy_(torch.randn_like(x) / 16)
        graph.replay()
        torch.testing.assert_close(output, x @ dense + bias, rtol=.05, atol=.05)


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_context_layers_growth_clear_and_state_dict(method):
    require_dtype(torch.float16, method)
    first, dense, bias = make_layer(method, torch.float16)
    second, dense2, bias2 = make_layer(method, torch.float16, k=288, n=200)
    keys = set(first.state_dict())
    with MarlinScratchContext("cuda:0") as context:
        for m in (1, 17, 64, 2, 128, 8):
            for layer, weight, offset in ((first, dense, bias), (second, dense2, bias2)):
                x = torch.randn(2, m, layer.in_features, device="cuda", dtype=torch.float16) / 16
                torch.testing.assert_close(layer(x), x @ weight + offset, rtol=.05, atol=.05)
        context.clear()
        x = torch.randn(8, 256, device="cuda", dtype=torch.float16) / 16
        torch.testing.assert_close(first(x), x @ dense + bias, rtol=.05, atol=.05)
        # A freshly initialized/reloaded layer cannot depend on previous scratch contents.
        reloaded, weight, offset = make_layer(method, torch.float16, seed=23)
        torch.testing.assert_close(reloaded(x), x @ weight + offset, rtol=.05, atol=.05)
    assert set(first.state_dict()) == keys
