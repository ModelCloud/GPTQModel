"""Exact differential tests for the experimental ROCm banked recurrence."""

import pytest
import torch

from gptqmodel.quantization.qvq import batched_v2b2_p32_viterbi_quantize

pytestmark = pytest.mark.skipif(torch.version.hip is None or not torch.cuda.is_available(), reason="ROCm required")


@pytest.mark.parametrize("bits", [2.0, 2.5, 3.0, 3.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("closed", [False, True])
def test_native_matches_torch(bits, dtype, closed):
    from gptqmodel.utils.qvq_yaqa_amd import banked_viterbi_trusted

    generator = torch.Generator(device="cuda").manual_seed(20260905)
    x = torch.randn((1, 128, 2), device="cuda", generator=generator)
    c = torch.randn((2, 65536, 2), device="cuda", generator=generator).to(dtype)
    overlap = torch.tensor([13], device="cuda") if closed else None
    weights = torch.rand((1, 128), device="cuda", generator=generator) if closed else None
    kwargs = {"bits": bits, "overlap": overlap, "step_weights": weights}
    expected = batched_v2b2_p32_viterbi_quantize(x, c, **kwargs)
    actual = banked_viterbi_trusted(x, c, **kwargs)
    for name in ("states", "segment_bank_ids", "values", "squared_error"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


@pytest.mark.parametrize("bits", [2.0, 2.5, 3.0, 3.5])
@pytest.mark.parametrize("banked", [False, True])
def test_public_tail_biting_and_packed_words(monkeypatch, bits, banked):
    from gptqmodel.quantization.qvq import (
        pack_trellis_states,
        tail_biting_v2b2_p32_quantize,
        tail_biting_viterbi_quantize,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260906)
    x = torch.randn((3, 128, 2), device="cuda", generator=generator)
    c = torch.randn((2, 65536, 2) if banked else (65536, 2), device="cuda", generator=generator)
    quantize = tail_biting_v2b2_p32_quantize if banked else tail_biting_viterbi_quantize
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "0")
    expected = quantize(x, c, bits=bits)
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "1")
    actual = quantize(x, c, bits=bits)
    for name in ("states", "values", "squared_error") + (("segment_bank_ids",) if banked else ()):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name
    assert torch.equal(pack_trellis_states(actual.states, bits=bits), pack_trellis_states(expected.states, bits=bits))


@pytest.mark.parametrize("graph", [False, True])
def test_full_yaqa_family_selection_matches_rocm_eager(monkeypatch, graph):
    from gptqmodel.quantization.qvq import yaqa_inner_v2b2_p32

    generator = torch.Generator(device="cuda").manual_seed(20260907)
    weight = torch.randn((32, 32), device="cuda", generator=generator)
    left = torch.randn((32, 32), device="cuda", generator=generator)
    right = torch.randn((32, 32), device="cuda", generator=generator)
    input_hessian = left @ left.T + torch.eye(32, device="cuda")
    output_hessian = right @ right.T + torch.eye(32, device="cuda")
    library = tuple(torch.randn((65536, 2), device="cuda", generator=generator) for _ in range(4))
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "0")
    expected = yaqa_inner_v2b2_p32(weight, input_hessian, output_hessian, library, bits=2.0)
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "1")
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_GRAPH", str(int(graph)))
    actual = yaqa_inner_v2b2_p32(weight, input_hessian, output_hessian, library, bits=2.0)
    assert len(actual) == len(expected) == 4
    for a, e in zip(actual, expected, strict=True):
        assert torch.equal(a, e)


@pytest.mark.parametrize("bits", [2.0, 2.5, 3.0, 3.5])
@pytest.mark.parametrize("closed", [False, True])
def test_native_zero_weight_ties(monkeypatch, bits, closed):
    generator = torch.Generator(device="cuda").manual_seed(20260909)
    x = torch.randn((2, 128, 2), device="cuda", generator=generator)
    c = torch.randn((2, 65536, 2), device="cuda", generator=generator)
    weights = torch.zeros((2, 128), device="cuda")
    overlap = torch.tensor([0, (1 << (16 - int(2 * bits))) - 1], device="cuda") if closed else None
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "0")
    expected = batched_v2b2_p32_viterbi_quantize(x, c, bits=bits, overlap=overlap, step_weights=weights)
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "1")
    actual = batched_v2b2_p32_viterbi_quantize(x, c, bits=bits, overlap=overlap, step_weights=weights)
    for name in ("states", "values", "squared_error", "segment_bank_ids"):
        assert torch.equal(getattr(actual, name), getattr(expected, name)), name


def test_native_nondefault_stream_and_graph_mutated_inputs(monkeypatch):
    from gptqmodel.utils.qvq_yaqa_amd import banked_viterbi_trusted

    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "0")
    generator = torch.Generator(device="cuda").manual_seed(20260910)
    x = torch.randn((1, 128, 2), device="cuda", generator=generator)
    c = torch.randn((2, 65536, 2), device="cuda", generator=generator)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        banked_viterbi_trusted(x, c, bits=2.0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = banked_viterbi_trusted(x, c, bits=2.0)
        for scale in (1.0, -0.5, 2.0):
            x.mul_(scale)
            c.add_(0.01)
            graph.replay()
            expected = batched_v2b2_p32_viterbi_quantize(x, c, bits=2.0)
            for name in ("states", "values", "squared_error", "segment_bank_ids"):
                assert torch.equal(getattr(actual, name), getattr(expected, name)), name
    torch.cuda.current_stream().wait_stream(stream)


def test_public_cached_graph_preserves_prior_results_and_mutated_inputs(monkeypatch):
    from gptqmodel.utils.qvq_yaqa_amd import clear_banked_viterbi_graph_cache

    clear_banked_viterbi_graph_cache()
    generator = torch.Generator(device="cuda").manual_seed(20260911)
    x = torch.randn((2, 128, 2), device="cuda", generator=generator)
    c = torch.randn((2, 65536, 2), device="cuda", generator=generator)
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "0")
    expected_first = batched_v2b2_p32_viterbi_quantize(x, c, bits=2.0)
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "1")
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_GRAPH", "1")
    first = batched_v2b2_p32_viterbi_quantize(x, c, bits=2.0)
    x.mul_(-0.75)
    expected_second = None
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "0")
    expected_second = batched_v2b2_p32_viterbi_quantize(x, c, bits=2.0)
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "1")
    second = batched_v2b2_p32_viterbi_quantize(x, c, bits=2.0)
    for actual, expected in ((first, expected_first), (second, expected_second)):
        for name in ("states", "values", "squared_error", "segment_bank_ids"):
            assert torch.equal(getattr(actual, name), getattr(expected, name)), name
    clear_banked_viterbi_graph_cache()


@pytest.mark.parametrize("case", ["fp64", "strided", "strict_pruning"])
def test_opt_in_keeps_unsupported_fallbacks_and_strict_policy(monkeypatch, case):
    from gptqmodel.utils import qvq_yaqa_amd

    x = torch.zeros((1, 128, 2), device="cuda")
    c = torch.zeros((2, 65536, 2), device="cuda")
    if case == "fp64":
        x = x.double()
    if case == "strided":
        x = torch.zeros((1, 256, 2), device="cuda")[:, ::2]

    def reject_native(*args, **kwargs):
        pytest.fail("Unsupported geometry/precision or strict pruning reached native recurrence")

    monkeypatch.setattr(qvq_yaqa_amd, "banked_viterbi_trusted", reject_native)
    monkeypatch.setenv("GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION", "1")
    if case == "strict_pruning":
        with pytest.raises(RuntimeError, match="ROCm"):
            batched_v2b2_p32_viterbi_quantize(x, c, bits=2.0, viterbi_pruning={"mode": "required"})
    else:
        actual = batched_v2b2_p32_viterbi_quantize(x, c, bits=2.0)
        assert torch.count_nonzero(actual.states) == 0
        assert torch.count_nonzero(actual.segment_bank_ids) == 0
        assert actual.squared_error.dtype == (torch.float64 if case == "fp64" else torch.float32)


@pytest.mark.parametrize("bits", [2.0, 2.5, 3.0, 3.5])
@pytest.mark.parametrize("closed", [False, True])
def test_family_batched_native_matches_independent_solves(bits, closed):
    from gptqmodel.utils.qvq_yaqa_amd import (
        banked_viterbi_trusted,
        family_banked_viterbi_trusted,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260912)
    sequences = torch.randn((3, 2, 128, 2), device="cuda", generator=generator)
    codebooks = torch.randn((3, 2, 65536, 2), device="cuda", generator=generator)
    overlap = torch.tensor([[0, 1], [2, 3], [4, 5]], device="cuda") if closed else None
    actual = family_banked_viterbi_trusted(sequences, codebooks, bits=bits, overlap=overlap)
    for family in range(3):
        expected = banked_viterbi_trusted(
            sequences[family], codebooks[family], bits=bits,
            overlap=None if overlap is None else overlap[family],
        )
        for name in ("states", "values", "squared_error", "segment_bank_ids"):
            assert torch.equal(getattr(actual, name)[family], getattr(expected, name)), name


@pytest.mark.parametrize("bits", [2.0, 2.5, 3.0, 3.5])
def test_family_batched_midpoint_matches_full_provisional(bits):
    from gptqmodel.utils.qvq_yaqa_amd import (
        family_banked_viterbi_midpoint_trusted,
        family_banked_viterbi_trusted,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260913)
    sequences = torch.randn((4, 2, 128, 2), device="cuda", generator=generator)
    codebooks = torch.randn((4, 2, 65536, 2), device="cuda", generator=generator)
    provisional = family_banked_viterbi_trusted(sequences, codebooks, bits=bits)
    actual = family_banked_viterbi_midpoint_trusted(sequences, codebooks, bits=bits)
    suffix_mask = (1 << (16 - int(2 * bits))) - 1
    assert torch.equal(actual, provisional.states[:, :, 63] & suffix_mask)
