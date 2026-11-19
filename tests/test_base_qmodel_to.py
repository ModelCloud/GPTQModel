# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``BaseQModel.to`` must fail with a real exception when the wrapped model
has no ``to`` method."""

import pytest

from gptqmodel.models.base import BaseQModel


def test_base_qmodel_to_raises_not_implemented_without_to():
    class ModelWithoutTo:
        pass

    holder = object.__new__(BaseQModel)
    holder.model = ModelWithoutTo()

    with pytest.raises(NotImplementedError, match=r"does not support the to\(\) method"):
        BaseQModel.to(holder, "cpu")
