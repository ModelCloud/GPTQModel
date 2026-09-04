# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import inspect

from gptqmodel.models.base import BaseQModel


def test_get_model_with_quantize_uses_instance_attrs():
    # ModelWriter is applied exactly once, to BaseQModel itself, so inside
    # its method bodies the decorator-closure ``cls`` is always BaseQModel.
    # Reading ``cls.loader`` / ``cls.lm_head`` there silently discards any
    # definition-subclass override (custom loader class, custom lm_head
    # name); such attributes must be read from ``self``.
    src = inspect.getsource(BaseQModel.get_model_with_quantize)
    assert "cls.loader" not in src
    assert "cls.lm_head" not in src
