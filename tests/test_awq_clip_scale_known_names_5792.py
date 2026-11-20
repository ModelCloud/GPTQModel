# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ clip/scale search must use the projection names the caller already has.

``_search_best_clip`` used to resolve each projection's model-level name four
times (bits, group size, format, symmetry), and every lookup walked the whole
model tree to recover the parent layer's name plus the layer subtree to recover
the projection name. ``_search_best_scale`` called the same wrappers while
preparing per-member options. The caller already holds
``NamedModule.full_name``; these tests pin the known-name route to zero model
walks and bitwise-identical results, with a control case for unnamed modules.
"""

from statistics import median
from time import perf_counter
from unittest.mock import patch

import pytest
import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor, _normalize_mse
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig


pytestmark = [pytest.mark.cpu]

DEPTHS = (0, 24)
PROJECTIONS = 8
SIBLINGS = 384
WIDTH = 16


class _Shell(nn.Module):
    """Synthetic decoder shell with inert siblings to reproduce tree traversal."""

    def __init__(self, layers=48, siblings=SIBLINGS, projections=PROJECTIONS):
        super().__init__()
        self.layers = nn.ModuleList(nn.Module() for _ in range(layers))
        for layer in self.layers:
            layer.linears = nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(projections))
            layer.expert_modules = nn.ModuleList(nn.Identity() for _ in range(siblings))
        self.count_visits = False
        self.visits = 0

    def named_modules(self, *args, **kwargs):
        for name, module in super().named_modules(*args, **kwargs):
            if self.count_visits:
                self.visits += 1
            yield name, module


def _dynamic(index):
    prefix = f"layers.{index}.linears"
    return {
        f"{prefix}.1": {"bits": 3},
        f"{prefix}.2": {"sym": False},
        f"{prefix}.3": {"group_size": 8},
    }


def _processor(model, *, dynamic=None):
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.model = model
    processor.gptq_model = None
    processor.qcfg = AWQConfig(
        bits=4,
        group_size=16,
        sym=True,
        offload_to_disk=False,
        device="cpu",
        dynamic=dynamic,
    )
    return processor


def _named_linears(model, index):
    layer = model.layers[index]
    return layer, {
        f"linears.{position}": NamedModule(linear, f"linears.{position}", f"layers.{index}.linears.{position}", index)
        for position, linear in enumerate(layer.linears)
    }


def _previous_clip_bounds(processor, layer, index, named, features):
    """Clip bounds through the old per-option identity lookups (origin/main)."""

    full_names = {module: f"layers.{index}.{name}" for name, module in named.items()}
    bounds = []
    with patch.object(processor, "_full_module_name", lambda _parent, module: full_names[module]):
        for name, linear in named.items():
            resolved = processor.qcfg.resolve_module_config(full_names[linear])
            mse = _normalize_mse(resolved.get("mse", None), base=processor.qcfg.mse)
            bounds.append(
                (
                    name,
                    processor._compute_best_clip(
                        linear.weight,
                        features[name],
                        bits=processor._bits_for(layer, linear),
                        group_size=processor._group_size_for(layer, linear),
                        format=processor._format_for(layer, linear),
                        sym=processor._sym_for(layer, linear),
                        clip_grid_budget=mse.clip_grid_budget,
                    ),
                )
            )
    return bounds


def _clip_search(processor, model, index, named, features):
    model.visits = 0
    model.count_visits = True
    try:
        bounds = processor._search_best_clip(model.layers[index], named, features)
    finally:
        model.count_visits = False
    return model.visits, bounds


def test_clip_search_known_names_walk_no_model_tree():
    torch.manual_seed(5792)
    model = _Shell()
    x = torch.rand(WIDTH, WIDTH)
    visits_by_depth = {}

    for index in DEPTHS:
        processor = _processor(model, dynamic=_dynamic(index))
        layer, named = _named_linears(model, index)
        features = dict.fromkeys(named, x)
        raw = {name: wrapped.module for name, wrapped in named.items()}

        visits, bounds = _clip_search(processor, model, index, named, features)
        visits_by_depth[index] = visits

        # One lookup per projection is the budget; the known names need none.
        assert visits <= len(named), (index, visits)
        expected = _previous_clip_bounds(processor, layer, index, raw, features)
        assert [name for name, _ in bounds] == [name for name, _ in expected]
        for (name, bound), (expected_name, expected_bound) in zip(bounds, expected):
            assert name == expected_name
            assert torch.equal(bound, expected_bound), (index, name)

    assert visits_by_depth[0] == visits_by_depth[24] == 0


def test_clip_search_unnamed_fallback_keeps_previous_bounds():
    """An unnamed caller still resolves names by identity and matches the grid."""

    torch.manual_seed(5793)
    model = _Shell(layers=2)
    x = torch.rand(WIDTH, WIDTH)
    index = 1
    processor = _processor(model, dynamic=_dynamic(index))
    layer, named = _named_linears(model, index)
    features = dict.fromkeys(named, x)
    raw = {name: wrapped.module for name, wrapped in named.items()}

    _, bounds = _clip_search(processor, model, index, raw, features)
    expected = _previous_clip_bounds(processor, layer, index, raw, features)

    for (name, bound), (expected_name, expected_bound) in zip(bounds, expected):
        assert name == expected_name
        assert torch.equal(bound, expected_bound), (index, name)


def test_clip_search_depth_24_timing_does_not_scale_with_tree_size():
    """Growing the inert sibling count must not grow the known-name search."""

    torch.manual_seed(5794)
    x = torch.rand(WIDTH, WIDTH)
    medians = {}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for siblings in (SIBLINGS, 8 * SIBLINGS):
            model = _Shell(layers=25, siblings=siblings)
            processor = _processor(model, dynamic=_dynamic(24))
            layer, named = _named_linears(model, 24)
            features = dict.fromkeys(named, x)
            processor._search_best_clip(layer, named, features)  # warm up
            measured = []
            for _ in range(5):
                start = perf_counter()
                processor._search_best_clip(layer, named, features)
                measured.append(perf_counter() - start)
            medians[siblings] = median(measured)
    finally:
        torch.set_num_threads(previous_threads)

    small, large = medians[SIBLINGS], medians[8 * SIBLINGS]
    # The old path walked every sibling eight times per projection, so an 8x
    # larger tree cost ~8x more. Known names do the same clip math on both.
    assert large <= 3.0 * max(small, 1e-4), (small, large)


class _ScaleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.linears = nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(PROJECTIONS))

    def forward(self, x):
        y = self.norm(x)
        output = 0
        for linear in self.linears:
            output = output + linear(y)
        return output


class _ScaleShell(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ScaleLayer()])
        self.count_visits = False
        self.visits = 0

    def named_modules(self, *args, **kwargs):
        for name, module in super().named_modules(*args, **kwargs):
            if self.count_visits:
                self.visits += 1
            yield name, module


def _scale_processor(model):
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(
            device="cpu",
            bits=4,
            group_size=16,
            sym=True,
            dynamic={
                "layers.0.linears.1": {"bits": 3},
                "layers.0.linears.2": {"sym": False},
            },
        ),
        calibration=CalibrationConfig(),
        gptq_model=model,
        model=model,
        calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args: None
    processor.log_new_row = lambda *args: None
    # Resolve the model-level rotary once so the count below covers the
    # per-member name lookups, not the unrelated one-time root scan.
    processor._get_root_rotary()
    return processor


def test_scale_search_known_names_walk_no_model_tree_and_match_fallback():
    torch.manual_seed(5795)
    model = _ScaleShell()
    layer = model.layers[0]
    processor = _scale_processor(model)
    x = torch.rand(12, WIDTH)
    named = {
        f"linears.{position}": NamedModule(linear, f"linears.{position}", f"layers.0.linears.{position}", 0)
        for position, linear in enumerate(layer.linears)
    }
    args = (layer, layer.norm, list(layer.linears), x)

    model.visits = 0
    model.count_visits = True
    try:
        known = processor._search_best_scale(*args, module2inspect=layer, named_childs=named)
    finally:
        model.count_visits = False
    known_visits = model.visits

    fallback = processor._search_best_scale(*args, module2inspect=layer)

    assert known_visits <= len(named), known_visits
    assert known[0] == fallback[0]
    assert known[1] == fallback[1]
    assert torch.equal(known[2], fallback[2])
    assert known[3] == fallback[3]
