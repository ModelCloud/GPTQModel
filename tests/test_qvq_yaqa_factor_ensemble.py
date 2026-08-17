# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from scripts.combine_qvq_yaqa_factor_caches import combine_yaqa_factor_payloads


def _payload(seed: int, value: float) -> dict:
    factor = torch.full((2, 2), value, dtype=torch.float32)
    return {
        "metadata": {
            "model": "llama",
            "dataset": "calibration",
            "rows": 2,
            "row_offset": 4,
            "batch_size": 1,
            "module_shapes": {"q_proj": [2, 2]},
            "seed": seed,
        },
        "input_hessians": {"q_proj": factor.clone()},
        "output_hessians": {"q_proj": factor.clone()},
        "stats": {"independent_sequences": 2, "monte_carlo_samples_per_output": 1},
    }


def test_combine_yaqa_factor_payloads_is_exact_two_sample_estimator():
    combined = combine_yaqa_factor_payloads([_payload(0, 2.0), _payload(1, 4.0)])

    assert torch.equal(combined["input_hessians"]["q_proj"], torch.full((2, 2), 3.0))
    assert torch.equal(combined["output_hessians"]["q_proj"], torch.full((2, 2), 3.0))
    assert combined["metadata"]["seed"] == "ensemble:0,1"
    assert combined["stats"]["monte_carlo_samples_per_output"] == 2


def test_combine_yaqa_factor_payloads_rejects_mismatched_evidence_contract():
    mismatched = _payload(1, 4.0)
    mismatched["metadata"]["row_offset"] = 5

    with pytest.raises(ValueError, match="share rows"):
        combine_yaqa_factor_payloads([_payload(0, 2.0), mismatched])
