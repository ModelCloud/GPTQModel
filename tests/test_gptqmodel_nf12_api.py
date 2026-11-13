# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""BF16 inference coverage for GPTQModel's lightweight NF12 API.

The expected values come from dense Torch BF16 weights, independently of the
NF12 decoder and the native inference operators.
"""

from __future__ import annotations

import shutil
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import gptqmodel_nf12 as nf12
from tests.accelerator_gates import nf12_skip_reason


def _cuda():
    reason = nf12_skip_reason()
    if reason:
        pytest.skip(reason)
    return torch.device("cuda:0")


def test_native_runtime_stays_testable_without_build_tools(monkeypatch):
    device = _cuda()
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert nf12_skip_reason() is None
    weight = torch.ones((16, 64), device=device, dtype=torch.bfloat16)
    x = torch.ones((2, 64), device=device, dtype=torch.bfloat16)
    packed = nf12.encode(weight)
    actual = nf12.linear(x, packed)
    assert actual.dtype == x.dtype
    assert torch.equal(actual, torch.full((2, 16), 64, device=device, dtype=x.dtype))


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


def test_repeat4_uses_single_copy_from_preloaded_legacy_namespace(monkeypatch):
    from gptqmodel_nf12 import _library, api

    bits = torch.tensor([[-32768, 0x7FC1, 1, -1], [0, 1, 2, 3]], dtype=torch.int16)
    single = bits.view(torch.bfloat16).reshape(2, 1, 4)
    legacy = SimpleNamespace(**{name: lambda *args: None for name in (
        "encode", "decode", "linear", "encode_moe", "moe_linear", "_fused_moe")})
    legacy.embedding = lambda *args: single
    monkeypatch.setattr(torch.ops, "nf12", legacy)
    monkeypatch.setattr(torch.version, "hip", None)
    loaded = threading.Event()
    monkeypatch.setattr(_library, "_loaded", loaded)
    monkeypatch.setattr(api, "_check_cuda", lambda *_: None)
    ids = torch.tensor([[0], [-1]], dtype=torch.int64)
    weight = SimpleNamespace(packed=torch.empty((16, 96), dtype=torch.uint8),
                             escapes=torch.empty((0, 8), dtype=torch.int16), exp_base=0)

    actual = api.embedding_repeat4(ids, weight)

    assert loaded.is_set()
    assert actual.shape == (2, 1, 4, 4)
    assert torch.equal(actual.view(torch.int16),
                       single.unsqueeze(-2).repeat(1, 1, 4, 1).view(torch.int16))


@pytest.mark.parametrize("m,id_dtype", [(6, torch.int32), (312, torch.int64),
                                          (1536, torch.int64), (3072, torch.int64)])
def test_native_embedding_repeat4_matches_dense_torch_and_keeps_packed_weight(m, id_dtype):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(14100 + m)
    weight = torch.randn((512, 5120), generator=generator, device=device).bfloat16()
    weight[0, 0] = -0.0
    weight[0, 1] = 2.0**-20
    ids = torch.randint(weight.shape[0], (m,), generator=generator, device=device, dtype=id_dtype)
    ids[0] = 0
    ids[-1] = weight.shape[0] - 1
    packed = nf12.encode(weight)
    expected = F.embedding(ids, weight).unsqueeze(1).repeat(1, 4, 1)
    for _ in range(2):
        actual = nf12.embedding_repeat4(ids, packed)
        assert actual.shape == (m, 4, 5120)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    invalid = ids.clone()
    invalid[0], invalid[-1] = -1, weight.shape[0]
    result = nf12.embedding_repeat4(invalid, packed)
    assert torch.count_nonzero(result[0].view(torch.int16)) == 0
    assert torch.count_nonzero(result[-1].view(torch.int16)) == 0
    assert torch.equal(result[1:-1].view(torch.int16), expected[1:-1].view(torch.int16))
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


@pytest.mark.parametrize("m,id_dtype", [
    (256, torch.int64), (312, torch.int32), (313, torch.int64),
    (512, torch.int32), (513, torch.int64), (1536, torch.int64),
    (2048, torch.int32), (2049, torch.int64), (3072, torch.int32),
    (3072, torch.int64), (4095, torch.int64),
])
def test_native_embedding_observed_lite_prefill_rows_are_exact_and_packed(m, id_dtype):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(10101 + m)
    weight = (torch.randn((512, 5120), generator=generator, device=device) * 0.02).bfloat16()
    weight[0, 0] = -0.0
    weight[-1, -1] = 2.0**-20
    ids = torch.randint(weight.shape[0], (m,), generator=generator, device=device, dtype=id_dtype)
    ids[0], ids[-1] = 0, weight.shape[0] - 1
    packed = nf12.encode(weight)
    module = nf12.NF12Embedding.from_weight(packed)
    oracle = F.embedding(ids, weight)
    for _ in range(2):
        actual = module(ids)
        assert torch.equal(actual.view(torch.int16), oracle.view(torch.int16))
    if m == 512:
        invalid = ids.clone()
        invalid[1], invalid[2] = -1, weight.shape[0]
        actual = module(invalid)
        assert torch.equal(actual[1:3].view(torch.int16), torch.zeros_like(actual[1:3]).view(torch.int16))
        assert torch.equal(actual[3:].view(torch.int16), oracle[3:].view(torch.int16))
    assert packed.nbytes < weight.nbytes
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}
    assert {name for name, _ in module.named_buffers()} == {"nf12_packed", "nf12_escapes"}
    assert all(buffer.dtype != torch.bfloat16 for _, buffer in module.named_buffers())


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


_CUDA_SYNC_SHAPES = [
    (1, 128, 2048),   # SM103 one-row GEMV
    (1, 512, 512),    # SM103 four-row GEMV
    (2, 512, 64),     # direct, minimum K
    (8, 512, 192),    # direct, non-256-aligned K
    (6, 128, 7168),   # clustered direct
    (2, 4096, 768),   # asynchronous, three stages
    (8, 3072, 4096),  # asynchronous, eight stages
    (8, 6144, 4096),  # escape staging when three CTAs per SM fit
    (9, 512, 512),    # outside specialized M range
]


@pytest.mark.parametrize("m,n,k", _CUDA_SYNC_SHAPES)
@pytest.mark.parametrize("static_weights", [False, True])
def test_cuda_sync_linear_matches_twin_and_torch(m, n, k, static_weights):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(20261113 + m + n + k)
    dense = (torch.randn((n, k), generator=generator, device=device) * 0.02).bfloat16()
    dense[0, 0], dense[-1, -1] = -0.0, 2.0**-20
    packed = nf12.encode(dense)
    assert packed.num_escapes > 0
    assert torch.equal(nf12.decode(packed).view(torch.int16), dense.view(torch.int16))
    x = (torch.randn((m, k), generator=generator, device=device) * 0.05).bfloat16()
    bias = (torch.randn((n,), generator=generator, device=device) * 0.01).bfloat16()
    actual = nf12.linear(x, packed, bias, static_weights=static_weights)
    twin = torch.ops.nf12.linear_bf16t(x, nf12.pack_bf16t(dense), bias)
    assert actual.dtype == x.dtype
    assert torch.equal(actual.view(torch.int16), twin.view(torch.int16))
    oracle = F.linear(x.double(), dense.double(), bias.double())
    torch.testing.assert_close(actual.double(), oracle, rtol=2e-3, atol=2e-3)
    assert torch.equal(nf12.linear(x, packed, bias, static_weights=static_weights), actual)
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


@pytest.mark.parametrize("static_weights", [False, True])
def test_cuda_sync_short_k_prefill_matches_torch(static_weights):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(20261115)
    dense = (torch.randn((512, 128), generator=generator, device=device) * 0.02).bfloat16()
    packed = nf12.encode(dense)
    x = (torch.randn((64, 128), generator=generator, device=device) * 0.05).bfloat16()
    actual = nf12.linear(x, packed, static_weights=static_weights)
    oracle = F.linear(x.double(), dense.double())
    torch.testing.assert_close(actual.double(), oracle, rtol=2e-3, atol=2e-3)
    if torch.cuda.get_device_capability(device) == (10, 3):
        assert torch.equal(actual.view(torch.int16), F.linear(x, dense).view(torch.int16))


@pytest.mark.parametrize("m,n,k", [(1, 128, 2048), (2, 512, 192), (2, 4096, 768)])
@pytest.mark.parametrize("static_weights", [False, True])
def test_cuda_sync_graph_replays_changed_inputs(m, n, k, static_weights):
    device = _cuda()
    generator = torch.Generator(device=device).manual_seed(20261114 + m + n + k)
    dense = (torch.randn((n, k), generator=generator, device=device) * 0.02).bfloat16()
    dense[-1, -1] = 2.0**-20
    packed = nf12.encode(dense)
    x = (torch.randn((m, k), generator=generator, device=device) * 0.05).bfloat16()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            nf12.linear(x, packed, static_weights=static_weights)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = nf12.linear(x, packed, static_weights=static_weights)
    torch.cuda.current_stream(device).wait_stream(stream)
    for _ in range(3):
        x.neg_()
        graph.replay()
        oracle = F.linear(x.double(), dense.double())
        torch.testing.assert_close(actual.double(), oracle, rtol=2e-3, atol=2e-3)
    module = nf12.NF12Linear.from_weight(packed)
    module(x)
    module(x)
    assert {name for name, _ in module.named_buffers()} == {"nf12_packed", "nf12_escapes"}
    assert set(vars(packed)) == {"packed", "escapes", "exp_base", "shape"}


@pytest.mark.parametrize("n,k", [(384, 5120), (512, 5120), (5120, 5120), (129280, 5120), (128, 2048)])
def test_staged_b1_matches_independent_rows_and_torch(n, k):
    device = _cuda()
    if (n, k) == (128, 2048) and torch.cuda.get_device_capability(device) != (10, 3):
        pytest.skip("the 128x2048 staged-B1 GEMV route requires SM103")
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


def test_prepared_fp32_gate_discards_replaced_and_modified_weight():
    from gptqmodel_nf12._fp32 import _get_gate_view

    device = _cuda()
    dense = torch.full((128, 5120), 0.03125, dtype=torch.bfloat16, device=device)
    dense[1::2].neg_()
    original = nf12.encode(dense)
    replacement = nf12.encode(-dense)
    replacement_bytes = replacement.packed.clone()
    assert original.num_escapes == replacement.num_escapes == 0
    assert original.exp_base == replacement.exp_base
    weight = nf12.NF12Weight(original.packed.clone(), original.escapes.clone(), original.exp_base)
    x = torch.ones((6, 5120), dtype=torch.float32, device=device)

    nf12.prepare_fp32_gate(weight)
    torch.testing.assert_close(nf12.linear_fp32(x[:1], weight),
                               F.linear(x[:1], dense.float()), rtol=2e-3, atol=2e-3)

    weight.packed = replacement.packed
    weight.escapes = replacement.escapes
    assert _get_gate_view(weight) is None
    torch.testing.assert_close(nf12.linear_fp32(x[:1], weight),
                               F.linear(x[:1], -dense.float()), rtol=2e-3, atol=2e-3)
    nf12.prepare_fp32_gate(weight)
    torch.testing.assert_close(nf12.linear_fp32_staged_b1(x, weight),
                               F.linear(x, -dense.float()), rtol=2e-3, atol=2e-3)

    weight.packed.copy_(original.packed)
    assert _get_gate_view(weight) is None
    torch.testing.assert_close(nf12.linear_fp32(x[:1], weight),
                               F.linear(x[:1], dense.float()), rtol=2e-3, atol=2e-3)
    nf12.prepare_fp32_gate(weight)
    torch.testing.assert_close(nf12.linear_fp32_staged_b1(x, weight),
                               F.linear(x, dense.float()), rtol=2e-3, atol=2e-3)
    weight.packed.data.copy_(replacement_bytes)
    # ``.data`` writes bypass the version counter; explicit preparation is the
    # contract for this form of mutation.
    nf12.prepare_fp32_gate(weight)
    torch.testing.assert_close(nf12.linear_fp32_staged_b1(x, weight),
                               F.linear(x, -dense.float()), rtol=2e-3, atol=2e-3)
    assert set(vars(weight)) == {"packed", "escapes", "exp_base", "shape"}


def test_prepared_fp32_gate_reprepares_unversioned_inference_weight():
    device = _cuda()
    dense = torch.full((128, 5120), 0.03125, dtype=torch.bfloat16, device=device)
    replacement = nf12.encode(-dense)
    with torch.inference_mode():
        source = nf12.encode(dense)
        weight = nf12.NF12Weight(source.packed.clone(), source.escapes.clone(), source.exp_base)
    assert weight.packed.is_inference()
    x = torch.ones((1, 5120), dtype=torch.float32, device=device)
    nf12.prepare_fp32_gate(weight)
    with torch.inference_mode():
        weight.packed.copy_(replacement.packed)
    # Inference tensors have no version counter. Repeating this explicit
    # preparation is the cache-clear contract after an in-place edit.
    nf12.prepare_fp32_gate(weight)
    torch.testing.assert_close(nf12.linear_fp32(x, weight),
                               F.linear(x, -dense.float()), rtol=2e-3, atol=2e-3)


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
