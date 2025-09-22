# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

def batched(iterable, n: int, process_func):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, "batch size must be at least one"
    from itertools import islice

    iterator = iter(iterable)

    while batch := tuple(islice(iterator, n)):
        if process_func is None:
            yield batch
        else:
            yield [process_func(item) for item in batch]
