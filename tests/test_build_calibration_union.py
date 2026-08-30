# SPDX-License-Identifier: Apache-2.0

from scripts.build_calibration_union import PreparedRow, deduplicate_rows


def _row(source: str, source_row: int, priority: int, user_digest: str) -> PreparedRow:
    return PreparedRow(
        source_name=source,
        source_path=f"/{source}.parquet",
        source_row=source_row,
        source_priority=priority,
        messages=[{"role": "user", "content": f"prompt {source_row}"}],
        raw_example_sha256=f"raw-{source}-{source_row}",
        normalized_user_sha256=user_digest,
        prepared_example_sha256=f"prepared-{source}-{source_row}",
        valid_input_tokens=100 + source_row,
        valid_fisher_output_token_samples=100 + source_row,
    )


def test_deduplicate_rows_prefers_yaqa_and_preserves_stable_source_order():
    rows = [
        _row("nm", 1, 1, "shared"),
        _row("nm", 0, 1, "nm-only"),
        _row("yaqa", 1, 0, "yaqa-only"),
        _row("yaqa", 0, 0, "shared"),
    ]

    winners, duplicate_groups = deduplicate_rows(rows)

    assert [(row.source_name, row.source_row) for row in winners] == [
        ("yaqa", 0),
        ("yaqa", 1),
        ("nm", 0),
    ]
    assert len(duplicate_groups) == 1
    assert duplicate_groups[0]["winner"]["source_name"] == "yaqa"
    assert [member["source_name"] for member in duplicate_groups[0]["members"]] == ["yaqa", "nm"]
