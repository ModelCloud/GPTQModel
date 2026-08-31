# SPDX-License-Identifier: Apache-2.0

from scripts.build_calibration_fisher_composition import _weighted_config, _weighted_coverage
from scripts.build_calibration_fisher_scaling import select_random_token_matched_rows
from scripts.build_calibration_union import PreparedRow


def _row(source_row: int, tokens: int) -> PreparedRow:
    return PreparedRow(
        source_name="nm",
        source_path="/nm.parquet",
        source_row=source_row,
        source_priority=1,
        messages=[{"role": "user", "content": f"prompt {source_row}"}],
        raw_example_sha256=f"raw-{source_row}",
        normalized_user_sha256=f"user-{source_row}",
        prepared_example_sha256=f"prepared-{source_row}",
        valid_input_tokens=tokens,
        valid_fisher_output_token_samples=tokens,
    )


def test_random_token_match_is_deterministic_and_close_to_target():
    rows = [_row(index_value, 80 + index_value) for index_value in range(20)]

    first = select_random_token_matched_rows(rows, target_tokens=503, seed=7)
    second = select_random_token_matched_rows(rows, target_tokens=503, seed=7)

    assert [row.source_row for row in first] == [row.source_row for row in second]
    assert abs(sum(row.valid_input_tokens for row in first) - 503) <= max(row.valid_input_tokens for row in rows)
    assert [row.source_row for row in first] == sorted(row.source_row for row in first)


def test_fisher_composition_records_unique_and_effective_coverage_separately():
    coverage = {
        "independent_sequences": 12,
        "valid_fisher_output_token_samples": 1200,
        "by_source": {
            "yaqa": {
                "rows_kept_after_dedup": 2,
                "kept_valid_fisher_output_token_samples": 400,
            },
            "nm": {
                "rows_kept_after_dedup": 10,
                "kept_valid_fisher_output_token_samples": 800,
            },
        },
    }

    weighted = _weighted_coverage(coverage, yaqa_weight=2.0)

    assert weighted["unique_sequences"] == 12
    assert weighted["raw_valid_tokens"] == 1200
    assert weighted["effective_weighted_sequences"] == 14
    assert weighted["effective_weighted_tokens"] == 1600
    assert weighted["YAQA_weight"] == 2.0
    assert weighted["NM_weight"] == 1.0


def test_fisher_composition_config_uses_source_weights_not_duplicate_rows():
    config = _weighted_config({"yaqa": {"seed": 0}}, yaqa_weight=2.0)

    assert config["yaqa"]["source_weight_column"] == "source_name"
    assert config["yaqa"]["source_weights"] == [["yaqa", 2.0], ["nm", 1.0]]
