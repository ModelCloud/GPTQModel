# GPU=-1
import numpy as np
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.slq import (
    build_dynamic_bits,
    expected_acceptance_rate,
    linear_sensitivity,
)


def test_dynamic_bits_round_trip_with_quantize_config():
    torch.manual_seed(0)
    groups = ["model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj"]
    bitwidths = [4, 5, 6, 7, 8]
    bitwidths_arr = np.array(bitwidths)
    weights = [torch.randn(128) for _ in groups]
    costs = linear_sensitivity(weights, bitwidths, symmetric=False)
    assignment = np.argmin(costs, axis=1)
    dynamic = build_dynamic_bits(groups, bitwidths, assignment)

    cfg = QuantizeConfig(dynamic=dynamic, bits=4, group_size=128, sym=False)
    for name in groups:
        resolved = cfg.dynamic_get(name, "bits", cfg.bits)
        assert resolved == bitwidths_arr[assignment[groups.index(name)]]


def test_ear_on_slq_quantized_logits():
    torch.manual_seed(42)
    # Simulate reference and quantized next-token logits.
    logits_p = torch.randn(8, 100)
    logits_q = logits_p + torch.randn(8, 100) * 0.05
    p = torch.softmax(logits_p, dim=-1)
    q = torch.softmax(logits_q, dim=-1)
    ear = expected_acceptance_rate(p, q, top_k=10)
    assert 0.0 <= ear.item() <= 1.0


def test_slq_end_to_end_bit_assignment():
    """Sanity-check that SLQ utilities can produce a valid per-layer config."""

    torch.manual_seed(1)
    groups = [f"layer_{i}" for i in range(8)]
    bitwidths = [2, 3, 4, 5, 6, 7, 8]
    weights = [torch.randn(64 * (i + 1)) for i in range(len(groups))]
    costs = linear_sensitivity(weights, bitwidths, symmetric=False)

    # Target an average budget of 4.5 bits using ILP.
    from gptqmodel.quantization.slq import allocate_bitwidth_ilp

    assignment = allocate_bitwidth_ilp(costs, bitwidths, budget=4.5)
    dynamic = build_dynamic_bits(groups, bitwidths, assignment)
    avg = np.mean([dynamic[f"^{g}$"]["bits"] for g in groups])
    assert avg <= 4.5 + 0.5  # Allow MILP granularity slack.
    assert all(2 <= dynamic[f"^{g}$"]["bits"] <= 8 for g in groups)
