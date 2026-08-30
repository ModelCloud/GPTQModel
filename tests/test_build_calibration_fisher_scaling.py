# SPDX-License-Identifier: Apache-2.0

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
