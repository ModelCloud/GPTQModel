# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""BF16 inference coverage for GPTQModel's lightweight NF12 API.

The expected values come from dense Torch BF16 weights, independently of the
NF12 decoder and the native inference operators.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import gptqmodel_nf12 as nf12


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("NF12 needs SM80 or newer")
    return device


@pytest.mark.parametrize("shape", [
    (16, 64), (32, 5120), (128, 512), (128, 5120), (384, 5120), (512, 5120),
])
def test_native_linear_and_exact_weight_roundtrip(shape):
    device = _cuda()
    n, k = shape
    generator = torch.Generator(device=device).manual_seed(540 + n + k)
    weight = (torch.randn(shape, generator=generator, device=device) * 0.02).bfloat16()
    # A signed zero is part of the weight format's exactness contract.
    weight[0, 0] = -0.0
    x = (torch.randn((3, k), generator=generator, device=device) * 0.05).bfloat16()
    bias = (torch.randn((n,), generator=generator, device=device) * 0.01).bfloat16()

    packed = nf12.encode(weight)
    restored = nf12.decode(packed)
    assert torch.equal(restored.view(torch.int16), weight.view(torch.int16))
    assert packed.packed.nbytes == weight.nbytes * 3 // 4

    actual = nf12.linear(x, packed, bias)
    oracle_fp32 = F.linear(x.float(), weight.float(), bias.float())
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), oracle_fp32, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual, oracle_fp32.bfloat16(), rtol=2e-3, atol=2e-3)
    module = nf12.NF12Linear.from_weight(packed, bias)
    assert torch.equal(module(x), actual)
    if n >= 32:
        sliced = packed.narrow_rows(16, 16)
        assert torch.equal(nf12.decode(sliced).view(torch.int16),
                           weight[16:32].contiguous().view(torch.int16))


def test_native_embedding_preserves_row_bits_and_shape():
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(121)
    weight = torch.randn((128, 256), generator=generator, device=device).bfloat16()
    weight[3, 0] = -0.0
    ids = torch.tensor([[3, 127], [0, 9]], device=device, dtype=torch.int64)
    packed = nf12.encode(weight)
    actual = nf12.embedding(ids, packed)
    oracle = F.embedding(ids, weight)
    assert actual.shape == (2, 2, 256)
    assert torch.equal(actual.view(torch.int16), oracle.view(torch.int16))
    assert torch.equal(nf12.NF12Embedding.from_weight(packed)(ids), actual)


@pytest.mark.parametrize("n,m", [(384, 128), (384, 512), (512, 128), (512, 512)])
def test_native_linear_prefill_matches_torch_oracle(n, m):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(304)
    weight = (torch.randn((n, 5120), generator=generator, device=device) * 0.02).bfloat16()
    x = (torch.randn((m, 5120), generator=generator, device=device) * 0.05).bfloat16()
    packed = nf12.encode(weight)
    actual = nf12.linear(x, packed)
    dense = F.linear(x, weight)
    fp64 = F.linear(x.double(), weight.double())
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), fp64.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual.float(), dense.float(), rtol=2e-3, atol=2e-3)
    # This shape takes call-local decode + cuBLAS on SM80, matching the dense
    # Torch call bit for bit. SM89 keeps the native reduction, whose FP32 sum
    # order can round to a different BF16 value without exceeding the oracle
    # error bound above.
    if torch.cuda.get_device_capability(device) == (8, 0):
        assert torch.equal(actual, dense)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


def test_native_compressor_decode_matches_dense_and_sampled_fp64():
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(2512)
    weight = (torch.randn((512, 5120), generator=generator, device=device) * 0.02).bfloat16()
    x = (torch.randn((6, 5120), generator=generator, device=device) * 0.05).bfloat16()
    packed = nf12.encode(weight)
    actual = nf12.linear(x, packed)
    dense = F.linear(x, weight)
    torch.testing.assert_close(actual, dense, rtol=2e-3, atol=2e-3)
    sample = torch.tensor([0, 7, 128, 255, 511], device=device)
    fp64 = F.linear(x.double(), weight.index_select(0, sample).double())
    torch.testing.assert_close(actual.index_select(1, sample).float(), fp64.float(),
                               rtol=2e-3, atol=2e-3)


def test_native_full_vocabulary_head_matches_dense_torch():
    device = _cuda()
    if torch.cuda.mem_get_info(device)[0] < 4_000_000_000:
        pytest.skip("full vocabulary head needs at least 4 GB free GPU memory")
    generator = torch.Generator(device=device).manual_seed(310)
    weight = (torch.randn((129280, 5120), generator=generator, device=device) * 0.02).bfloat16()
    x = (torch.randn((5, 5120), generator=generator, device=device) * 0.05).bfloat16()
    packed = nf12.encode(weight)
    actual = nf12.linear(x, packed)
    oracle = F.linear(x.float(), weight.float())
    torch.testing.assert_close(actual.float(), oracle, rtol=2e-3, atol=2e-3)
    assert torch.equal(actual.topk(8, dim=-1).indices,
                       oracle.bfloat16().topk(8, dim=-1).indices)
    ids = torch.tensor([0, 21, 129279], device=device)
    assert torch.equal(nf12.embedding(ids, packed).view(torch.int16),
                       F.embedding(ids, weight).view(torch.int16))


@pytest.mark.parametrize("n,k", [(384, 5120), (512, 5120), (5120, 5120), (129280, 5120)])
def test_staged_b1_matches_independent_rows_and_torch(n, k):
    device = _cuda()
    if n >= 129280 and torch.cuda.mem_get_info(device)[0] < 4_000_000_000:
        pytest.skip("full vocabulary head needs at least 4 GB free GPU memory")
    generator = torch.Generator(device=device).manual_seed(851 + n)
    dense = (torch.randn((n, k), generator=generator, device=device) * 0.02).bfloat16()
    dense[0, 0] = -0.0
    dense[-1, -1] = 2.0**-20  # exercise escape decoding without a large outlier
    packed = nf12.encode(dense)
    assert torch.equal(nf12.decode(packed).view(torch.int16), dense.view(torch.int16))
    for m in (1, 2, 6, 8):
        x = (torch.randn((m, k), generator=generator, device=device) * 0.05).bfloat16()
        bias = (torch.randn((n,), generator=generator, device=device) * 0.01).bfloat16()
        expected = torch.cat([nf12.linear(x[i:i + 1], packed, bias) for i in range(m)])
        actual = nf12.linear_staged_b1(x, packed, bias)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        torch.testing.assert_close(actual.float(), F.linear(x.float(), dense.float(), bias.float()),
                                   rtol=2e-3, atol=2e-3)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}
    with pytest.raises(ValueError, match="requires a BF16"):
        nf12.linear_staged_b1(torch.empty((9, k), device=device, dtype=torch.bfloat16), packed)


@pytest.mark.parametrize("n", [128, 384, 512, 5120])
@pytest.mark.parametrize("m", [2, 6, 8])
def test_fp32_staged_b1_matches_independent_rows_and_torch(n, m):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(859 + n + m)
    dense = (torch.randn((n, 5120), generator=generator, device=device) * 0.02).bfloat16()
    dense[0, 0] = 2.0**10
    packed = nf12.encode(dense)
    x = torch.randn((m, 5120), generator=generator, device=device) * 0.05
    bias = torch.randn((n,), generator=generator, device=device) * 0.01
    expected = torch.cat([nf12.linear_fp32(x[i:i + 1], packed, bias) for i in range(m)])
    actual = nf12.linear_fp32_staged_b1(x, packed, bias)
    assert torch.equal(actual, expected)
    torch.testing.assert_close(actual, F.linear(x, dense.float(), bias), rtol=2e-3, atol=2e-3)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


@pytest.mark.parametrize("n", (128, 384))
def test_prepared_fp32_gate_matches_b1_and_fp64_oracle(n):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(20261111)
    dense = (torch.randn((n, 5120), generator=generator, device=device) * 0.02).bfloat16()
    dense[0, 0] = -0.0
    dense[0, 7] = 2.0**10
    dense[-1, -1] = -(2.0**-20)
    packed = nf12.encode(dense)
    nf12.prepare_fp32_gate(packed)
    nf12.prepare_fp32_gate(packed)
    from gptqmodel_nf12._fp32 import _get_gate_view
    meta, low, escapes = _get_gate_view(packed)
    assert meta.dtype == torch.int32 and low.dtype == torch.uint8
    assert escapes.dtype == torch.int16
    assert meta.nbytes + low.nbytes == dense.nbytes * 3 // 4
    assert meta.nbytes + low.nbytes + escapes.nbytes < dense.nbytes
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}
    x = torch.randn((6, 5120), generator=generator, device=device) * 0.05
    bias = torch.randn((n,), generator=generator, device=device) * 0.01
    staged = nf12.linear_fp32_staged_b1(x, packed, bias)
    singles = torch.cat([nf12.linear_fp32(x[i:i + 1], packed, bias) for i in range(6)])
    assert torch.equal(staged, singles)
    fp64 = F.linear(x.double(), dense.double(), bias.double()).float()
    torch.testing.assert_close(staged, fp64, rtol=2e-3, atol=2e-3)
    assert torch.equal(staged.topk(8, -1).indices, fp64.topk(8, -1).indices)
    changed = x.clone()
    changed[2].mul_(1.7)
    assert torch.equal(nf12.linear_fp32_staged_b1(changed, packed, bias)[2],
                       nf12.linear_fp32(changed[2:3], packed, bias)[0])


@pytest.mark.parametrize("shape", [
    (32, 512), (32, 5120), (128, 5120), (384, 5120), (512, 5120),
])
def test_fp32_projection_from_packed_weight_matches_dense_torch(shape):
    device = _cuda()
    n, k = shape
    generator = torch.Generator(device=device).manual_seed(416 + n)
    weight = (torch.randn(shape, generator=generator, device=device) * 0.02).bfloat16()
    # Force escape groups at both ends of the weight; the FP32 kernel must
    # reconstruct their original BF16 bits before multiplication.
    weight[0, 0] = 2.0**10
    weight[-1, -1] = -(2.0**-20)
    x = (torch.randn((6, k), generator=generator, device=device) * 0.05).float()
    bias = (torch.randn((n,), generator=generator, device=device) * 0.01).float()
    packed = nf12.encode(weight)
    assert packed.num_escapes > 0

    actual = nf12.linear_fp32(x, packed, bias)
    oracle = F.linear(x, weight.float(), bias)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)
    assert torch.equal(actual.topk(min(4, n), dim=-1).indices,
                       oracle.topk(min(4, n), dim=-1).indices)
    module = nf12.NF12Linear.from_weight(packed, bias.bfloat16())
    torch.testing.assert_close(module(x), F.linear(x, weight.float(), module.bias.float()),
                               rtol=2e-3, atol=2e-3)
    rows = torch.tensor([n - 1, 0, 2, n - 1], device=device, dtype=torch.int64)
    selected = nf12.linear_fp32(x, packed, bias, rows=rows)
    torch.testing.assert_close(selected, F.linear(x, weight.float(), bias)[..., rows],
                               rtol=2e-3, atol=2e-3)
    # An int64 ID outside int32 range must not wrap into a valid packed row.
    invalid = torch.tensor([2**32, -1], device=device, dtype=torch.int64)
    assert torch.equal(nf12.linear_fp32(x, packed, rows=invalid),
                       torch.zeros((x.shape[0], 2), device=device))
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


@pytest.mark.parametrize("shape,m", [((384, 5120), 32), ((128, 512), 256)])
def test_fp32_prefill_crossover_matches_dense_torch_without_retained_weight(shape, m):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(702 + m)
    weight = (torch.randn(shape, generator=generator, device=device) * 0.02).bfloat16()
    x = (torch.randn((m, shape[1]), generator=generator, device=device) * 0.05).float()
    packed = nf12.encode(weight)
    actual = nf12.linear_fp32(x, packed)
    oracle = F.linear(x, weight.float())
    assert torch.equal(actual, oracle)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


@pytest.mark.parametrize("m", (256, 384, 512, 3072))
def test_fp32_main_gate_prefill_tensor_core_matches_torch_and_keeps_weight_packed(m):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(41591 + m)
    weight = (torch.randn((384, 5120), generator=generator, device=device) * 0.02).bfloat16()
    weight[0, 0] = -0.0
    weight[0, 7] = 2.0**10
    weight[-1, -1] = -(2.0**-20)
    packed = nf12.encode(weight)
    assert packed.num_escapes > 0
    x = torch.randn((m, 5120), generator=generator, device=device) * 0.05
    bias = torch.randn((384,), generator=generator, device=device) * 0.01
    expected = F.linear(x, weight.float(), bias)
    actual = nf12.linear_fp32(x, packed, bias)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    assert torch.equal(actual.topk(8, dim=-1).indices, expected.topk(8, dim=-1).indices)
    sample_rows = torch.tensor((0, 1, m // 2, m - 1), device=device)
    oracle = F.linear(x.index_select(0, sample_rows).double(), weight.double(), bias.double())
    torch.testing.assert_close(actual.index_select(0, sample_rows).double(), oracle,
                               rtol=2e-3, atol=2e-3)
    changed = x.clone()
    changed[m // 2].mul_(1.7)
    changed_out = nf12.linear_fp32(changed, packed, bias)
    torch.testing.assert_close(changed_out[m // 2],
                               F.linear(changed[m // 2], weight.float(), bias),
                               rtol=2e-3, atol=2e-3)
    assert not torch.equal(changed_out[m // 2], actual[m // 2])
    assert torch.equal(nf12.decode(packed).view(torch.int16), weight.view(torch.int16))
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}
    torch.cuda.current_stream().synchronize()
    retained_before = torch.cuda.memory_allocated(device)
    repeated = nf12.linear_fp32(x, packed, bias)
    torch.cuda.current_stream().synchronize()
    retained_after = torch.cuda.memory_allocated(device)
    assert retained_after - retained_before <= repeated.nbytes + 1_048_576
    del repeated
    assert torch.cuda.memory_allocated(device) <= retained_before + 1_048_576


def test_fp32_main_gate_prefill_tensor_core_graph_and_3d_input():
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(41599)
    weight = (torch.randn((384, 5120), generator=generator, device=device) * 0.02).bfloat16()
    packed = nf12.encode(weight)
    x = torch.randn((2, 256, 5120), generator=generator, device=device) * 0.05
    reference = F.linear(x, weight.float())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = nf12.linear_fp32(x, packed)
    graph.replay()
    torch.testing.assert_close(captured, reference, rtol=2e-3, atol=2e-3)
    x.mul_(0.5)
    graph.replay()
    torch.testing.assert_close(captured, F.linear(x, weight.float()), rtol=2e-3, atol=2e-3)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


def _moe_oracle(x, w13, w2, topk_weights, topk_ids):
    """Dense Torch arithmetic with the documented BF16 stage boundaries."""
    result = torch.zeros((x.shape[0], w2.shape[1]), dtype=torch.float32, device=x.device)
    intermediate = w13.shape[1] // 2
    for token in range(x.shape[0]):
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            hidden = F.linear(x[token].float(), w13[expert].float()).bfloat16()
            gated = (F.silu(hidden[:intermediate].float())
                     * hidden[intermediate:].float()).bfloat16()
            projected = F.linear(gated.float(), w2[expert].float())
            result[token] += (projected * topk_weights[token, route]).bfloat16().float()
    return result.bfloat16()


def test_native_fused_moe_matches_dense_torch_oracle():
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(23)
    x = (torch.randn((4, 64), generator=generator, device=device) * 0.05).bfloat16()
    w13 = (torch.randn((2, 128, 64), generator=generator, device=device) * 0.02).bfloat16()
    w2 = (torch.randn((2, 64, 64), generator=generator, device=device) * 0.02).bfloat16()
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]], device=device)
    topk_weights = torch.tensor([[0.6, 0.4], [0.8, 0.2], [0.1, 0.9], [0.5, 0.5]], device=device)
    packed13, packed2 = nf12.encode_moe(w13), nf12.encode_moe(w2)
    assert torch.equal(packed13.decode().view(torch.int16), w13.view(torch.int16))
    assert torch.equal(packed2.decode().view(torch.int16), w2.view(torch.int16))

    actual = nf12.fused_moe(x, packed13, packed2, topk_weights, topk_ids)
    oracle = _moe_oracle(x, w13, w2, topk_weights, topk_ids)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


def test_large_markov_shortlist_uses_packed_selected_rows():
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(902)
    weight = (torch.randn((129280, 256), generator=generator, device=device) * 0.02).bfloat16()
    packed = nf12.encode(weight)
    ids = torch.randperm(weight.shape[0], generator=generator, device=device)[:8192]
    x = (torch.randn((1, 256), generator=generator, device=device) * 0.05).float()
    actual = nf12.linear_fp32(x, packed, rows=ids)
    oracle = F.linear(x, weight.index_select(0, ids).float())
    assert actual.shape == (1, 8192)
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


def test_full_markov_fp32_single_row_matches_dense_torch():
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(1101)
    weight = (torch.randn((129280, 256), generator=generator, device=device) * 0.02).bfloat16()
    x = (torch.randn((1, 256), generator=generator, device=device) * 0.05).float()
    packed = nf12.encode(weight)
    actual = nf12.linear_fp32(x, packed)
    oracle = F.linear(x, weight.float())
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)
    assert torch.equal(actual.argmax(-1), oracle.argmax(-1))
    ids = torch.tensor([0, 17, 129279], device=device)
    assert torch.equal(nf12.embedding(ids, packed).view(torch.int16),
                       F.embedding(ids, weight).view(torch.int16))


@pytest.mark.parametrize("m", [1, 5, 16])
def test_full_markov_bf16_input_fp32_output_matches_dense_torch(m):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(1201 + m)
    weight = (torch.randn((129280, 256), generator=generator, device=device) * 0.02).bfloat16()
    weight[0, 0] = 2.0**10
    weight[-1, -1] = -(2.0**-20)
    x = (torch.randn((m, 256), generator=generator, device=device) * 0.05).bfloat16()
    bias = (torch.randn((129280,), generator=generator, device=device) * 0.001).float()
    packed = nf12.encode(weight)
    assert packed.num_escapes > 0
    actual = nf12.linear_bf16_fp32(x, packed, bias)
    oracle = F.linear(x.float(), weight.float(), bias)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)
    assert torch.equal(actual.argmax(-1), oracle.argmax(-1))
    sample = torch.cat((torch.arange(0, 129280, 1024, device=device),
                        torch.tensor([129279], device=device)))
    fp64 = F.linear(x.double(), weight.index_select(0, sample).double(),
                    bias.index_select(0, sample).double())
    torch.testing.assert_close(actual.index_select(1, sample).double(), fp64,
                               rtol=2e-5, atol=2e-6)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


def test_shape_contract_rejects_non_matrix_bf16_weights():
    device = _cuda()
    with pytest.raises(ValueError, match="multiple of 16"):
        nf12.encode(torch.zeros((1, 64), dtype=torch.bfloat16, device=device))
    with pytest.raises(ValueError, match="multiple of 64"):
        nf12.encode(torch.zeros((16, 65), dtype=torch.bfloat16, device=device))
