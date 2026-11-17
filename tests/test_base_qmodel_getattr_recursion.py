# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``BaseQModel.__getattr__`` must not re-enter itself while ``_modules`` is absent.

``BaseQModel`` forwards unknown attributes to its inner ``model``. To find that
inner module it has to read ``self._modules`` even when the attribute has not
been created yet: during ``__init__`` and on test fixtures built with
``object.__new__``. Asking ``hasattr(self, "_modules")`` re-enters this hook
forever, so the first unrelated attribute probe raised ``RecursionError`` and a
test teardown spent minutes formatting the traceback.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.base import BaseQModel


def test_unset_attribute_probe_on_a_bare_instance_raises_attribute_error():
    instance = object.__new__(BaseQModel)

    with pytest.raises(AttributeError):
        instance.missing_attribute
    assert not hasattr(instance, "missing_attribute")


def test_missing_modules_container_still_raises_attribute_error():
    instance = object.__new__(BaseQModel)
    instance._parameters = {}
    instance._buffers = {}

    with pytest.raises(AttributeError):
        instance.missing_attribute


def test_attribute_forwarding_to_the_inner_model_still_works():
    instance = object.__new__(BaseQModel)
    nn.Module.__init__(instance)
    instance.model = nn.Module()
    instance.model.config = SimpleNamespace(model_type="llama")

    assert instance.config.model_type == "llama"
    assert instance.model is instance._modules["model"]


def test_probe_during_init_style_construction_does_not_recurse():
    """A subclass that probes attributes before ``super().__init__`` finishes."""

    class ProbesEarly(BaseQModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.probed = hasattr(self, "_modules")
            self.model = nn.Module()
            self.model.marker = torch.tensor(1)

    instance = ProbesEarly()

    assert instance.probed is True
    torch.testing.assert_close(instance.marker, torch.tensor(1))
