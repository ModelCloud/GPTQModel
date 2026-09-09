import io

import pytest
import torch

from gptqmodel.quantization.qvq import (
    decode_p32_window_tiles,
    repack_p32_planar_to_window,
    repack_p32_window_to_planar,
    rht_preprocess_weight,
)
from gptqmodel.quantization.qvq_gsq import _candidate_probabilities, refine_p32_candidates
from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU


@pytest.mark.parametrize("bits", [1, 1.5, 2, 2.5, 3, 3.5])
def test_hard_export_and_baseline_guard(bits):
    gen = torch.Generator().manual_seed(12)
    candidates = torch.randint(-(2**31), 2**31 - 1, (3, 1, int(bits * 8)), generator=gen, dtype=torch.int32)
    bank = torch.tensor([0b10101010], dtype=torch.uint8)
    alt = torch.tensor([2], dtype=torch.int32)
    target = decode_p32_window_tiles(candidates[1], bits=bits, bank_ids=bank, bank_alt_id=alt).reshape(16, 16)
    inputs = torch.eye(16)
    original = candidates.clone()
    kwargs = dict(bits=bits, bank_ids=bank, bank_alt_id=alt, target=target, inputs=inputs,
                  enabled=True, steps=40, seed=9)
    result = refine_p32_candidates(candidates, **kwargs)
    repeated = refine_p32_candidates(candidates, **kwargs)
    assert torch.equal(result.window_words, repeated.window_words)
    assert torch.equal(candidates, original)
    assert result.calibration_after <= result.calibration_before
    assert result.calibration_after == 0  # the exact target is a legal candidate
    assert torch.equal(result.window_words, candidates[result.choices, torch.arange(1)])
    assert torch.equal(result.window_words, repack_p32_planar_to_window(
        repack_p32_window_to_planar(result.window_words, bits=bits), bits=bits
    ))
    buffer = io.BytesIO()
    torch.save(result.window_words, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert torch.equal(loaded, result.window_words)
    assert loaded.numel() * loaded.element_size() == candidates[0].numel() * 4


def test_reject_nonfinite_and_noop():
    candidates = torch.zeros(2, 1, 16, dtype=torch.int32)
    kwargs = dict(bits=2, bank_ids=torch.zeros(1, dtype=torch.uint8), bank_alt_id=torch.tensor([1]),
                  target=torch.ones(16, 16), inputs=torch.eye(16), enabled=True, steps=0)
    result = refine_p32_candidates(candidates, **kwargs)
    assert result.calibration_before == result.calibration_after
    kwargs["target"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        refine_p32_candidates(candidates, **kwargs)


def test_full_layer_inner_objective_matches_deployed_output():
    """The real-layer fitter may use inner NMSE only with constant |SV|."""
    gen = torch.Generator().manual_seed(7)
    weight = torch.randn(32, 64, generator=gen)
    inputs = torch.randn(40, 64, generator=gen)
    su = torch.where(torch.arange(64) % 2 == 0, 1.0, -1.0)
    sv = torch.where(torch.arange(32) % 2 == 0, 0.125, -0.125)
    target = rht_preprocess_weight(weight, su.reciprocal(), sv.reciprocal())
    transformed = matmul_hadU(inputs * su)
    teacher = inputs @ weight.T
    torch.testing.assert_close(matmul_hadU(transformed @ target) * sv, teacher, atol=1e-5, rtol=1e-5)
    candidate = target + torch.randn(target.shape, generator=gen) * 0.1
    inner_nmse = (transformed @ (candidate - target)).square().sum() / (transformed @ target).square().sum()
    deployed_nmse = (matmul_hadU(transformed @ candidate) * sv - teacher).square().sum() / teacher.square().sum()
    torch.testing.assert_close(inner_nmse, deployed_nmse)


@pytest.mark.parametrize("temperature", [0.1, 1.0, 2.0])
def test_gumbel_math_and_gradient(temperature):
    logits = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64, requires_grad=True)
    uniform = torch.tensor([0.17, 0.61, 0.89], dtype=torch.float64)
    probabilities = _candidate_probabilities(logits, uniform, temperature)
    # Independent scalar formula; also pins the noise sign.
    import math
    expected = torch.tensor([math.exp((value - math.log(-math.log(u))) / temperature)
                             for value, u in zip(logits.tolist(), uniform.tolist(), strict=True)], dtype=torch.float64)
    expected /= expected.sum()
    torch.testing.assert_close(probabilities, expected)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: _candidate_probabilities(value, uniform, temperature), logits)
    torch.testing.assert_close(jacobian, (torch.diag(expected) - expected.outer(expected)) / temperature)
    assert torch.autograd.gradcheck(lambda value: _candidate_probabilities(value, uniform, temperature), (logits,))
    # Refactoring the audited formula preserves the previous FP32 experiment.
    old = ((logits.float() - (-uniform.float().log()).log()) / temperature).softmax(-1)
    assert torch.equal(_candidate_probabilities(logits.float(), uniform.float(), temperature), old)


@pytest.mark.parametrize("explicit", [False, True])
def test_disabled_preserves_payload_without_fitting(monkeypatch, explicit):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled GSQ must not decode, optimize, or sample")

    import gptqmodel.quantization.qvq_gsq as gsq
    candidates = torch.arange(40, dtype=torch.int32).reshape(2, 1, 20)
    original = candidates.clone()
    rng = torch.random.get_rng_state().clone()
    monkeypatch.setattr(gsq, "decode_p32_window_tiles", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch, "rand", forbidden)
    kwargs = {"enabled": False} if explicit else {}
    result = refine_p32_candidates(
        candidates, bits=2.5, bank_ids=torch.empty(0), bank_alt_id=torch.empty(0),
        target=torch.tensor(float("nan")), inputs=torch.empty(0), progress=forbidden, **kwargs)
    assert torch.equal(result.window_words, original[0])
    assert result.window_words.data_ptr() != candidates.data_ptr()
    assert torch.equal(candidates, original)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert result.choices.tolist() == [0]
    assert result.calibration_before is result.calibration_after is None
    assert result.history == []


def test_control_requires_boolean():
    with pytest.raises(TypeError, match="enabled must be boolean"):
        refine_p32_candidates(torch.zeros(2, 1, 20, dtype=torch.int32), bits=2.5,
                              bank_ids=torch.empty(0), bank_alt_id=torch.empty(0),
                              target=torch.empty(0), inputs=torch.empty(0), enabled="false")


def test_fisher_loss_matches_independent_quadratic():
    gen = torch.Generator().manual_seed(7)
    candidates = torch.randint(-(2**31), 2**31-1, (2, 2, 20), dtype=torch.int32, generator=gen)
    bank, alt = torch.zeros(2, dtype=torch.uint8), torch.tensor([2])
    target = torch.randn(16, 32, generator=gen)
    a, b = torch.randn(23, 16, generator=gen), torch.randn(41, 32, generator=gen)
    h, g = a.T @ a + torch.eye(16), b.T @ b + torch.eye(32)
    weight = decode_p32_window_tiles(candidates[0], bits=2.5, bank_ids=bank, bank_alt_id=alt)
    weight = weight.reshape(1, 2, 16, 16).permute(0, 2, 1, 3).reshape(16, 32)
    result = refine_p32_candidates(candidates, enabled=True, steps=0, bits=2.5, bank_ids=bank,
                                   bank_alt_id=alt, target=target, inputs=torch.linalg.cholesky(h).T,
                                   right_factor=torch.linalg.cholesky(g))
    error = (weight-target).double()
    denominator = torch.trace(g.double() @ target.double().T @ h.double() @ target.double())
    expected = torch.trace(g.double() @ error.T @ h.double() @ error) / denominator
    assert result.calibration_before == pytest.approx(expected.item(), rel=2e-6)


def test_fisher_refinement_in_inference_mode_and_budget():
    from gptqmodel.quantization import GSQConfig
    from gptqmodel.quantization.qvq_gsq import refine_p32_fisher

    with torch.inference_mode():
        baseline = torch.zeros(1, 20, dtype=torch.int32)
        kwargs = dict(target=torch.eye(16), input_hessian=torch.eye(16), output_hessian=torch.eye(16),
                      bits=2.5, bank_ids=torch.zeros(1, dtype=torch.uint8), bank_alt_id=torch.tensor([2]))
        result = refine_p32_fisher(baseline, config=GSQConfig(enabled=True, steps=2, candidates=3), **kwargs)
        assert result.calibration_after <= result.calibration_before
        with pytest.raises(ValueError, match="max_candidate_bytes"):
            refine_p32_fisher(baseline, config=GSQConfig(enabled=True, max_candidate_bytes=1), **kwargs)
        with pytest.raises(ValueError, match="enabled GSQConfig"):
            refine_p32_fisher(baseline, config=None, **kwargs)


def test_reject_invalid_right_factor():
    with pytest.raises(ValueError, match="right_factor"):
        refine_p32_candidates(torch.zeros(2, 1, 20, dtype=torch.int32), enabled=True, bits=2.5,
                              bank_ids=torch.zeros(1, dtype=torch.uint8), bank_alt_id=torch.tensor([1]),
                              target=torch.eye(16), inputs=torch.eye(16), right_factor=torch.ones(15, 15))


@pytest.mark.parametrize("bits", [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8])
def test_nonbank_planar_adapter_fit_and_roundtrip(bits):
    from gptqmodel.quantization.qvq_gsq import TrellisCandidateAdapter, refine_trellis_candidates

    gen = torch.Generator().manual_seed(7)
    candidates = torch.randint(-(2**31), 2**31-1, (3, 1, int(bits*8)), dtype=torch.int32, generator=gen)
    adapter = TrellisCandidateAdapter("qvq_planar", bits)
    target = adapter.inner(candidates[1], 16, 16)
    result = refine_trellis_candidates(candidates, layout="qvq_planar", bits=bits, target=target,
                                       inputs=torch.eye(16), enabled=True, steps=40, seed=7)
    assert result.calibration_after == 0
    assert torch.equal(result.words, candidates[1])
    assert torch.equal(adapter.pack(adapter.unpack(result.words)), result.words)
    assert result.words.numel() == candidates[0].numel()
    # The exact PGC16 reference is independent of the adapter's packing path.
    from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
    torch.testing.assert_close(adapter.inner(result.words, 16, 16), reconstruct_qvq_inner_weight(
        result.words, bits=bits, in_features=16, out_features=16), atol=0, rtol=0)


@pytest.mark.parametrize("layout,bits", [("p32_window", 4), ("qvq_planar", 3.5), ("unknown", 4)])
def test_adapter_rejects_wrong_format(layout, bits):
    from gptqmodel.quantization.qvq_gsq import TrellisCandidateAdapter
    with pytest.raises(ValueError, match="adapter requires"):
        TrellisCandidateAdapter(layout, bits)


def test_adapter_rejects_mismatched_bank_metadata():
    from gptqmodel.quantization.qvq_gsq import TrellisCandidateAdapter
    with pytest.raises(ValueError, match="requires selectors"):
        TrellisCandidateAdapter("p32_window", 2.5).decode(torch.zeros(1, 20, dtype=torch.int32))
    with pytest.raises(ValueError, match="cannot accept bank"):
        TrellisCandidateAdapter("qvq_planar", 4).decode(torch.zeros(1, 32, dtype=torch.int32), torch.ones(1))


@pytest.mark.parametrize("words", [20, 32, 64])
def test_shared_pool_preserves_historical_draws(words):
    from gptqmodel.quantization.qvq_gsq import baseline_bitflip_candidates

    baseline = torch.arange(3 * words, dtype=torch.int32).reshape(3, words)
    original = baseline.clone()
    state = torch.random.get_rng_state().clone()
    actual = baseline_bitflip_candidates(baseline, count=33, seed=7)
    expected = baseline.unsqueeze(0).repeat(33, 1, 1)
    rng = torch.Generator().manual_seed(7)
    tiles = torch.arange(3)
    for index in range(1, 33):
        bit = torch.randint(words * 32, (3,), generator=rng)
        expected[index, tiles, bit // 32] ^= (torch.ones_like(bit) << (bit % 32)).to(torch.int32)
    assert torch.equal(actual, expected)
    assert torch.equal(baseline, original)
    assert torch.equal(state, torch.random.get_rng_state())
    xor = (actual[1:].long() ^ baseline.long()) & 0xffffffff
    for candidate in xor:
        for tile in candidate:
            assert sum(int(word).bit_count() for word in tile) == 1


@pytest.mark.parametrize("bits,layout", [(2.5, "p32_window"), (4, "qvq_planar"), (8, "qvq_planar")])
def test_deterministic_fisher_search_matches_full_recomputation(bits, layout):
    from gptqmodel.quantization.qvq_gsq import TrellisCandidateAdapter, deterministic_trellis_candidates

    rng = torch.Generator().manual_seed(71)
    candidates = torch.randint(-(2**31), 2**31 - 1, (3, 4, int(bits * 8)),
                               generator=rng, dtype=torch.int32)
    bank = torch.zeros(4, dtype=torch.uint8) if layout == "p32_window" else None
    alt = torch.tensor([1]) if bank is not None else None
    adapter = TrellisCandidateAdapter(layout, bits)
    decoded = torch.stack([adapter.inner(c, 32, 32, bank, alt) for c in candidates])
    target = decoded[1] * 0.6 + decoded[2] * 0.4
    x = torch.randn(40, 32, generator=rng)
    right = torch.randn(32, 32, generator=rng)  # deliberately couples output blocks
    result = deterministic_trellis_candidates(candidates, target=target, inputs=x, right_factor=right,
                                              bits=bits, layout=layout, bank_ids=bank, bank_alt_id=alt, sweeps=2)
    current = decoded[0].clone()
    choices = torch.zeros(4, dtype=torch.long)
    def loss(weight):
        return (x @ (weight - target) @ right).square().sum()
    for _ in range(2):
        for tile in range(4):
            ib, jb = divmod(tile, 2)
            i, j = slice(16 * ib, 16 * (ib + 1)), slice(16 * jb, 16 * (jb + 1))
            costs = []
            for index in range(3):
                trial = current.clone()
                trial[i, j] = decoded[index, i, j]
                costs.append(loss(trial))
            best = int(torch.stack(costs).argmin())
            if costs[best] < loss(current):
                current[i, j] = decoded[best, i, j]
                choices[tile] = best
    assert torch.equal(result.choices, choices)
    assert torch.equal(result.words, candidates[choices, torch.arange(4)])
    assert result.calibration_after < result.calibration_before
    assert result.calibration_after == pytest.approx(float(loss(current) / (x @ target @ right).square().sum()), rel=1e-5)


def test_matched_experiment_records_exact_pool_and_restores_hook(tmp_path):
    import gptqmodel.quantization.qvq_gsq as fitter
    from gptqmodel.quantization import GSQConfig
    from scripts.validate_qvq_gsq_layers import digest, quantize_with_matched_search

    original = fitter.refine_trellis_fisher
    baseline = torch.arange(32, dtype=torch.int32).reshape(1, 32)
    config = GSQConfig(enabled=True, steps=3, candidates=3, seed=7)
    candidates = fitter.baseline_bitflip_candidates(baseline, count=3, seed=7)
    target = fitter.TrellisCandidateAdapter("qvq_planar", 4).inner(candidates[1], 16, 16)
    def quantize():
        return fitter.refine_trellis_fisher(
            baseline, target=target, input_hessian=torch.eye(16), output_hessian=torch.eye(16),
            config=config, bits=4, layout="qvq_planar", bank_ids=None, bank_alt_id=None,
            codebook_version=fitter.PGC16_CODEBOOK_VERSION)
    path, comparison = tmp_path / "shared.pt", {}
    result = quantize_with_matched_search(quantize, comparison, path)
    assert fitter.refine_trellis_fisher is original
    saved = torch.load(path, weights_only=True)
    assert torch.equal(saved["candidates"], candidates)
    assert torch.equal(saved["target"], target)
    assert comparison["shared_artifact_sha256"] == digest(path)
    assert comparison["before"] == pytest.approx(result.calibration_before)
    assert comparison["after"] <= comparison["before"]
