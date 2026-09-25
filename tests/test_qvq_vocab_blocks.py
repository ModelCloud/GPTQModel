# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Check that vocabulary partitioning preserves logits and Fisher blocks."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from gptqmodel.quantization.qvq_yaqa import YaqaGramSketch, capture_yaqa_sketch_b
from optimize.qvq_vocab_blocks import VocabBlockLinear, factored_head_fisher_loss
from scripts.experiments.qvq_vocab_block_probe import (
    load_shared_factor_cache,
    save_shared_factor_cache,
)
from scripts.experiments.qvq_vocab_head_arbitrate import choose_guarded_blocks
from scripts.experiments.qvq_vocab_head_artifact import _oracle, _oracle_contribution


class _TinyCausal(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 16)
        self.decoder = nn.Linear(16, 16, bias=False)
        self.lm_head = nn.Linear(16, 48, bias=False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        from types import SimpleNamespace

        return SimpleNamespace(logits=self.lm_head(self.decoder(self.embed(input_ids))))


def test_vocab_blocks_preserve_full_distribution_and_principal_fisher_blocks():
    torch.manual_seed(778)
    control = _TinyCausal().eval()
    partitioned = copy.deepcopy(control)
    partitioned.lm_head = VocabBlockLinear(partitioned.lm_head, block_rows=16)
    batches = [{"input_ids": torch.tensor([[1, 3, 5], [2, 4, 6]]),
                "attention_mask": torch.ones(2, 3, dtype=torch.long)}]
    with torch.no_grad():
        baseline_logits = control(**batches[0]).logits
        block_logits = partitioned(**batches[0]).logits
    torch.testing.assert_close(block_logits, baseline_logits, atol=0, rtol=0)

    full_input, full_output, _ = capture_yaqa_sketch_b(
        control, batches, {"lm_head": control.lm_head}, device=torch.device("cpu"),
        seed=778, minimum_sequences=2, first_decoder_layer=control.lm_head,
    )
    block_input, block_output, _ = capture_yaqa_sketch_b(
        partitioned, batches, partitioned.lm_head.yaqa_targets(), device=torch.device("cpu"),
        seed=778, minimum_sequences=2, first_decoder_layer=partitioned.lm_head,
    )
    # YAQA normalizes an input Gram by its module's output width. Recover
    # the full-head normalization when combining independently captured blocks.
    combined_input = sum(
        block_input[name] * (module.out_features / control.lm_head.out_features)
        for name, module in partitioned.lm_head.yaqa_targets().items()
    )
    torch.testing.assert_close(combined_input, full_input["lm_head"], atol=2e-6, rtol=2e-6)
    for index in range(3):
        name = f"lm_head.blocks.{index}"
        expected = full_output["lm_head"][index * 16:(index + 1) * 16,
                                           index * 16:(index + 1) * 16]
        torch.testing.assert_close(block_output[name], expected, atol=2e-6, rtol=2e-6)


def test_factored_head_fisher_oracle_retains_cross_block_terms():
    torch.manual_seed(471)
    error = torch.randn(5, 12, dtype=torch.float64)
    source = torch.randn(12, 3, dtype=torch.float64)
    input_source = torch.randn(5, 5, dtype=torch.float64)
    hessian = input_source @ input_source.T
    dense_output = source @ source.T
    expected = torch.einsum("in,ij,jm,nm->", error, hessian, error, dense_output)
    blocks = [error[:, :7], error[:, 7:]]
    factor_blocks = [source[:7], source[7:]]
    actual64 = factored_head_fisher_loss(blocks, factor_blocks, hessian, dtype=torch.float64)
    actual32 = factored_head_fisher_loss(blocks, factor_blocks, hessian, dtype=torch.float32)
    torch.testing.assert_close(actual64, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual32.double(), expected, atol=1e-4, rtol=1e-5)


def test_shared_head_factor_preserves_principal_and_cross_blocks():
    torch.manual_seed(934)
    source = torch.randn(48, 8, dtype=torch.float32)
    diagonal = source.square().sum(1).mul_(1.125 / 29)
    sketch = YaqaGramSketch(source=source, diagonal=diagonal, normalizer=29, seed=7)
    factor = sketch.factor(device=torch.device("cpu"))
    full = sketch.materialize(device=torch.device("cpu"))
    torch.testing.assert_close(factor @ factor.T, full, atol=2e-6, rtol=2e-6)
    for start in (0, 16, 32):
        block = factor[start:start + 16]
        torch.testing.assert_close(block @ block.T, full[start:start + 16, start:start + 16],
                                   atol=2e-6, rtol=2e-6)
    cross = factor[:16] @ factor[16:32].T
    torch.testing.assert_close(cross, full[:16, 16:32], atol=2e-6, rtol=2e-6)


def test_shared_head_factor_cache_preserves_identity_and_exact_diagonal(tmp_path):
    torch.manual_seed(391)
    input_source = torch.randn(16, 4)
    output_source = torch.randn(48, 4)
    # A transferred GPU reduction need not equal a fresh CPU reduction bitwise.
    input_sketch = YaqaGramSketch(
        input_source, input_source.square().sum(1), 1.0, 7,
        source_diagonal=input_source.square().sum(1) * 1.001,
        _source_diagonal_validated=True,
    )
    output_sketch = YaqaGramSketch(
        output_source, output_source.square().sum(1), 1.0, 7,
        source_diagonal=output_source.square().sum(1) * 1.001,
        _source_diagonal_validated=True,
    )
    identity = {
        "schema": "qvq.yaqa.shared-head-factor.v2", "model_path": "/fixture/model",
        "calibration_sha256": "a" * 64, "requested_sequences": "2", "gram_rank": "4",
        "batch_size": "1", "seed": "7", "input_features": "16", "output_features": "48",
    }
    path = tmp_path / "factors.safetensors"
    save_shared_factor_cache(path, {"lm_head": input_sketch}, {"lm_head": output_sketch},
                             {**identity, "independent_sequences": "2", "valid_tokens": "19"})
    inputs, outputs, stats = load_shared_factor_cache(path, identity)
    assert stats == {"independent_sequences": 2, "valid_output_samples": 19}
    torch.testing.assert_close(inputs["lm_head"].factor(device=torch.device("cpu")),
                               input_sketch.factor(device=torch.device("cpu")), atol=0, rtol=0)
    torch.testing.assert_close(outputs["lm_head"].factor(device=torch.device("cpu")),
                               output_sketch.factor(device=torch.device("cpu")), atol=0, rtol=0)
    with pytest.raises(ValueError, match="provenance"):
        load_shared_factor_cache(path, {**identity, "calibration_sha256": "b" * 64})
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_shared_factor_cache(path, {"lm_head": input_sketch}, {"lm_head": output_sketch},
                                 {**identity, "independent_sequences": "2", "valid_tokens": "19"})


def test_streaming_whole_head_oracle_retains_cross_blocks_and_damping():
    torch.manual_seed(521)
    error = torch.randn(13, 5, dtype=torch.float64)
    factor = torch.randn(13, 3, dtype=torch.float64)
    h_source = torch.randn(5, 5, dtype=torch.float64)
    h = h_source @ h_source.T
    input_damping = 0.07
    output_damping = 0.11
    for dtype in (torch.float32, torch.float64):
        contributions = [
            _oracle_contribution(error[start:stop], factor[start:stop], h,
                                 input_damping, dtype=dtype)
            for start, stop in ((0, 6), (6, 13))
        ]
        z = sum(item[0] for item in contributions)
        penalty = sum(item[1] for item in contributions)
        undamped, damped = _oracle(z, penalty, h, input_damping, output_damping, dtype=dtype)
        e, s, hd = error.to(dtype), factor.to(dtype), h.to(dtype)
        gd = s @ s.T
        expected_undamped = torch.einsum("oi,ij,pj,op->", e, hd, e, gd)
        hd = hd + input_damping * torch.eye(5, dtype=dtype)
        gd = gd + output_damping * torch.eye(13, dtype=dtype)
        expected_damped = torch.einsum("oi,ij,pj,op->", e, hd, e, gd)
        tolerance = 1e-6 if dtype == torch.float32 else 1e-12
        assert abs(undamped - float(expected_undamped)) < tolerance * abs(float(expected_undamped))
        assert abs(damped - float(expected_damped)) < tolerance * abs(float(expected_damped))


def test_whole_head_arbitration_rejects_cross_block_overshoot_and_damped_loss():
    h = torch.ones((1, 1), dtype=torch.float64)
    base_z = torch.ones((1, 1), dtype=torch.float64)
    base_penalty = torch.zeros((), dtype=torch.float64)
    delta = torch.full((1, 1), -0.8, dtype=torch.float64)
    zero = torch.zeros((), dtype=torch.float64)
    selected, z, _ = choose_guarded_blocks(
        base_z, base_penalty, [(delta, zero), (delta, zero)], h, 0.0, 0.0,
    )
    assert len(selected) == 1
    torch.testing.assert_close(z.square().sum(), torch.tensor(0.04, dtype=torch.float64))
    selected, _, _ = choose_guarded_blocks(
        base_z, base_penalty,
        [(delta, torch.tensor(100.0, dtype=torch.float64))], h, 0.0, 0.1,
    )
    assert selected == []
