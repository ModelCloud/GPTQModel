# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from pathlib import Path

from datasets import load_dataset


# 20-row imdb-shaped (text, label) sample: keeps the loader check hermetic.
FIXTURE = Path(__file__).parent / "fixtures" / "datasets" / "imdb_tiny.jsonl"


def test_dataset_loader():
    # Load a percentage slice, the way calibration callers request small splits.
    dataset = load_dataset("json", data_files=str(FIXTURE), split="train[:10%]")

    assert dataset.num_rows == 2
    assert set(dataset.column_names) == {"text", "label"}

    first_row = dataset[0]
    assert first_row["text"].startswith("Review 0:")
    assert first_row["label"] == 0


if __name__ == "__main__":
    test_dataset_loader()
