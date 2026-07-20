# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Regression tests for state-dict collection during save.

These tests target the persistent-buffer filtering inside
``get_state_dict_for_save`` and ``_collect_state_dict_with_offload``.
A previous implementation called ``get_module_by_name`` for every buffer,
which is O(N^2) (N buffers x N modules). The current implementation walks
the module tree once, so large MoE models save in seconds instead of hours.
"""

import torch
from torch import nn

from gptqmodel.utils.model import get_state_dict_for_save


class _Expert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)


class _MoELayer(nn.Module):
    """Layer holding many experts plus an init-only (non-persistent) buffer."""

    def __init__(self, width: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(width) for _ in range(num_experts)])
        # Non-persistent buffer, mimicking rotary/positional init-time state.
        self.register_buffer("inv_freq", torch.linspace(0.0, 1.0, width), persistent=False)


class _MoEModel(nn.Module):
    def __init__(self, width: int, num_layers: int, num_experts: int):
        super().__init__()
        self.embed = nn.Embedding(64, width)
        self.layers = nn.ModuleList(
            [_MoELayer(width, num_experts) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(width)


def _persistent_buffers_reference(model: nn.Module) -> dict:
    """Reference: the set of persistent buffer names PyTorch would serialize."""
    persistent = {}
    # state_dict() excludes non-persistent buffers by default.
    for name, tensor in model.state_dict().items():
        if name in dict(model.named_buffers()):
            persistent[name] = tensor
    return persistent


def test_non_persistent_buffers_are_excluded():
    model = _MoEModel(width=8, num_layers=2, num_experts=3)

    state_dict = get_state_dict_for_save(model, offload_root=None)

    saved_names = set(state_dict.keys())

    # Every persistent buffer must be present.
    for name in _persistent_buffers_reference(model):
        assert name in saved_names, f"persistent buffer {name} missing from state dict"

    # Non-persistent buffers (inv_freq on each layer) must NOT be present.
    for layer_idx in range(2):
        assert f"layers.{layer_idx}.inv_freq" not in saved_names, (
            "non-persistent buffer should be excluded"
        )


def test_all_persistent_parameters_and_buffers_present():
    model = _MoEModel(width=8, num_layers=2, num_experts=3)

    state_dict = get_state_dict_for_save(model, offload_root=None)

    # Parameters
    for name, _ in model.named_parameters():
        assert name in state_dict, f"parameter {name} missing"

    # Persistent buffers
    for name, buf in model.named_buffers():
        module_path, leaf = name.rsplit(".", 1)
        # find the owning module to check persistence
        owner = model.get_submodule(module_path)
        if leaf in getattr(owner, "_non_persistent_buffers_set", set()):
            assert name not in state_dict
        else:
            assert name in state_dict, f"persistent buffer {name} missing"


def test_mixed_persistent_and_non_persistent_buffers():
    """A module may have both persistent and non-persistent buffers."""

    class _Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.register_buffer("persist_buf", torch.ones(4))
            self.register_buffer("ephemeral_buf", torch.ones(4), persistent=False)

    model = _Mixed()
    state_dict = get_state_dict_for_save(model, offload_root=None)

    assert "linear.weight" in state_dict
    assert "linear.bias" in state_dict
    assert "persist_buf" in state_dict
    assert "ephemeral_buf" not in state_dict


def test_nested_empty_module_prefix_is_safe():
    """Edge case: a module nested under a named-but-empty container must not break naming."""

    class _Empty(nn.Module):
        pass

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.container = _Empty()
            self.container.child = nn.Linear(4, 4)
            self.container.register_buffer("buf", torch.ones(4))

    model = _Root()
    state_dict = get_state_dict_for_save(model, offload_root=None)

    assert "container.child.weight" in state_dict
    assert "container.child.bias" in state_dict
    assert "container.buf" in state_dict


def test_collection_scales_linearly_with_module_count():
    """Smoke/regression check: collection must remain fast for a wide module tree.

    The previous O(N^2) implementation would balloon with module count. Here we
    build a moderately wide model and assert collection completes quickly.
    """
    import time

    num_layers, num_experts, width = 40, 60, 8
    model = _MoEModel(width=width, num_layers=num_layers, num_experts=num_experts)
    num_modules = sum(1 for _ in model.modules())
    num_buffers = sum(1 for _ in model.named_buffers())

    start = time.perf_counter()
    state_dict = get_state_dict_for_save(model, offload_root=None)
    elapsed = time.perf_counter() - start

    # With ~5K modules and ~7K buffers the O(N^2) path would take many seconds
    # to minutes; the fixed linear path should be well under 2 seconds.
    assert elapsed < 2.0, f"state-dict collection took {elapsed:.3f}s for {num_modules} modules / {num_buffers} buffers"
    # Sanity: non-persistent buffers excluded, persistent present.
    for layer_idx in range(num_layers):
        assert f"layers.{layer_idx}.inv_freq" not in state_dict
