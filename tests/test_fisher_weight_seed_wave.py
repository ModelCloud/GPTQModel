from __future__ import annotations

import torch

from scripts.build_calibration_fisher_composition import (
    _weight_label,
    _weighted_config,
    _weighted_coverage,
)
from scripts.compare_qvq_seed_checkpoints import _difference, _layer_index


def test_fractional_yaqa_weight_is_preserved_without_row_duplication():
    base = {"yaqa": {"seed": 0}}
    config = _weighted_config(base, yaqa_weight=1.5)
    assert config["yaqa"]["source_weight_column"] == "source_name"
    assert config["yaqa"]["source_weights"] == [["yaqa", 1.5], ["nm", 1.0]]
    assert _weight_label(1.5) == "15"
    assert _weight_label(3.0) == "3"


def test_fractional_weight_coverage_reports_unique_and_effective_counts():
    coverage = {
        "independent_sequences": 4277,
        "valid_fisher_output_token_samples": 1_794_387,
        "by_source": {
            "yaqa": {
                "rows_kept_after_dedup": 182,
                "kept_valid_fisher_output_token_samples": 302_193,
            },
            "nm": {
                "rows_kept_after_dedup": 4095,
                "kept_valid_fisher_output_token_samples": 1_492_194,
            },
        },
    }
    weighted = _weighted_coverage(coverage, yaqa_weight=1.5)
    assert weighted["unique_sequences"] == 4277
    assert weighted["raw_valid_tokens"] == 1_794_387
    assert weighted["effective_weighted_sequences"] == 4368.0
    assert weighted["effective_weighted_tokens"] == 1_945_483.5


def test_checkpoint_distance_helpers_report_element_disagreement_and_layer():
    stats = _difference(torch.tensor([0, 1, 2]), torch.tensor([0, 3, 2]))
    assert stats == {
        "compatible": True,
        "elements": 3,
        "differing_elements": 1,
        "disagreement_fraction": 1 / 3,
    }
    assert _layer_index("model.layers.12.mlp.up_proj") == 12
    assert _layer_index("lm_head") is None
