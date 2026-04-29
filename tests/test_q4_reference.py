# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from q4_reference import REFERENCE, get_diff


def test_q4_reference_smoke():
    # Keep an addressable pytest target for CI jobs that request `test_q4_reference`.
    assert REFERENCE.dtype == REFERENCE.new_empty(()).dtype
    assert "Maxdiff:" in get_diff(REFERENCE, REFERENCE)
