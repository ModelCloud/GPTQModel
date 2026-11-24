# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""``encoder_stages_on_host`` accepts every method spelling the loader passes.

The staging decision used to test ``qcfg.method`` for set membership directly.
A stub carrying a ``SimpleNamespace`` ``value`` is unhashable, so the decision
raised ``TypeError`` instead of reporting whether the encoder runs on the host,
and a plain string spelling was never recognized.
"""

from types import SimpleNamespace

import pytest

from gptqmodel.quantization.config import (
    CPU_BACKED_ENCODER_METHODS,
    encoder_stages_on_host,
)


@pytest.mark.parametrize(
    "method", sorted(CPU_BACKED_ENCODER_METHODS, key=lambda item: item.value)
)
def test_every_cpu_backed_method_is_recognized_in_every_spelling(method):
    assert encoder_stages_on_host(SimpleNamespace(method=method))
    assert encoder_stages_on_host(SimpleNamespace(method=SimpleNamespace(value=method)))
    assert encoder_stages_on_host(SimpleNamespace(method=method.value))


def test_absent_unknown_and_empty_methods_report_no_host_staging():
    assert encoder_stages_on_host(SimpleNamespace()) is False
    assert encoder_stages_on_host(SimpleNamespace(method=None)) is False
    assert encoder_stages_on_host(SimpleNamespace(method="not-a-method")) is False
    assert encoder_stages_on_host(SimpleNamespace(method=SimpleNamespace())) is False
