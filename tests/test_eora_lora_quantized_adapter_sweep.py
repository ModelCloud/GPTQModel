# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from scripts.eora_lora_quantized_adapter_sweep import _metric_summary, _table_rows


def test_metric_summary_counts_evalution_samples_when_sample_count_is_absent():
    result = {
        "tests": [
            {
                "name": "gsm8k_platinum_cot",
                "metrics": {"acc,num": 0.5},
                "samples": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}],
            }
        ]
    }

    summary = _metric_summary(result)

    assert summary["sample_count"] == 4
    assert summary["correct"] == 2


def test_table_rows_recomputes_rows_per_second_from_eval_seconds():
    rows = _table_rows(
        [
            {
                "bits": 4,
                "group_size": 32,
                "file_mb": 10.0,
                "rel_l2": 0.1,
                "sqnr_db": 20.0,
                "eval": {
                    "eval_seconds": 5.0,
                    "peak_allocated_gb": 1.25,
                    "summary": {
                        "sample_count": 20,
                        "acc_num": 0.5,
                        "correct": 10,
                    },
                },
            }
        ],
        100 * 1024**2,
    )

    assert rows[0][2] == 20
    assert rows[0][5] == "4.000"
