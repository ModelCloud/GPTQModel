# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from gptqmodel.utils.vram import get_non_persistent_buffers, named_module_tensors


class _Child(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.register_buffer("cache", torch.zeros(64), persistent=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("root_cache", torch.zeros(8), persistent=False)
        self.register_buffer("root_persistent", torch.zeros(8))
        self.child = _Child()


def test_get_non_persistent_buffers_returns_qualified_names_without_mutation():
    model = _Model()

    assert get_non_persistent_buffers(model, recurse=False) == {"root_cache"}
    assert get_non_persistent_buffers(model, recurse=True) == {"root_cache", "child.cache"}
    assert model._non_persistent_buffers_set == {"root_cache"}
    assert get_non_persistent_buffers(model, recurse=False) == {"root_cache"}


def test_named_module_tensors_filters_nested_non_persistent_buffers():
    model = _Model()

    all_names = {name for name, _ in named_module_tensors(model, recurse=True)}
    filtered_names = {name for name, _ in named_module_tensors(model, recurse=True, remove_non_persistent=True)}

    assert all_names == {"child.weight", "root_cache", "root_persistent", "child.cache"}
    assert filtered_names == {"child.weight", "root_persistent"}
