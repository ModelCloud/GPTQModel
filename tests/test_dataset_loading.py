# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from datasets import load_dataset


def test_dataset_loader():
    # Load a small split of a dataset from Hugging Face
    dataset = load_dataset("imdb", split="train[:1%]")  # load only 1% to keep it small

    # Print dataset info
    print(dataset)

    # Print the first row
    first_row = dataset[0]
    print("First row:", first_row)

if __name__ == "__main__":
    test_dataset_loader()
