# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.looper.loop_processor import _format_gib


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.00, "0G"),
        (1.00, "1G"),
        (1.50, "1.5G"),
        (1.25, "1.25G"),
        (0.05, "0.05G"),
        (10.00, "10G"),
    ],
)
def test_format_gib(value, expected):
    # Regression: whole-GiB values rendered with a trailing dot ("1.G")
    # because only the zeros were stripped from the fixed-point text.
    assert _format_gib(value) == expected
