# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Routed and shared expert reservoir captures must stay with their module (#5267).

#5236 made the routed-expert reservoir merge in place: ``entry["inputs"]`` is a
fresh ``buffer[:used].unsqueeze(0)`` view per module, materialized only when
``_layer_input_features`` runs. The sibling-share caches were keyed on
``tuple(id(tensor) ...)``, but the loop drops the previous module's view when it
rebinds ``tensors``, so CPython could hand the same id to the next module's
view. The cache then returned the previous module's rows and its packed mask
and positions, silently scoring one expert on another expert's activations.

Sibling captures that genuinely share one activation object (q/k/v, gate/up
under the plan path) must still share one concatenated tensor and one set of
replay kwargs.
"""

import threading
from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.quantization.config import CalibrationConfig


def _inputs_cache(batch_lengths, hidden):
    cache = SimpleNamespace(
        layer_inputs=[],
        attention_masks=[],
        position_ids=[],
        layer_input_kwargs=[],
    )
    for length in batch_lengths:
        cache.layer_inputs.append([torch.randn(1, length, hidden)])
        cache.attention_masks.append(torch.ones(1, length, dtype=torch.bool))
        cache.position_ids.append(torch.arange(length, dtype=torch.long).unsqueeze(0))
        cache.layer_input_kwargs.append({})
    return cache


def _processor(max_tokens, hidden, batch_lengths=None):
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor._mask_tls = threading.local()
    processor.tasks = {}
    processor._reservoir_locks = {}
    processor._capture_host_cache = {}
    processor._capture_sampled_cache = {}
    processor._token_row_plans = {}
    processor._capture_host_batch = None
    processor._module_forward_kwargs = {}
    processor._feature_task_names = set()
    processor._feature_stats = {}
    processor._awq_feature_kwargs = {}
    processor._scale_feature_by_module = {}
    processor._nsamples_total = 1
    processor.inputs_cache = (
        _inputs_cache(batch_lengths, hidden)
        if batch_lengths is not None
        else SimpleNamespace(
            layer_inputs=None,
            attention_masks=[],
            position_ids=[],
            layer_input_kwargs=[],
        )
    )
    processor.calibration = CalibrationConfig(exclude_attention_sinks=False)
    gptq_model = SimpleNamespace()
    gptq_model.awq_input_feature_aggregation = lambda name: {
        "mode": "token_rows",
        "max_tokens": max_tokens,
    }
    processor.gptq_model = gptq_model
    return processor


def test_reservoir_routed_and_shared_experts_keep_their_own_rows():
    """Every module must replay exactly the rows it captured.

    The routed experts and the shared expert are all on the dynamic reservoir
    path (their per-batch row counts do not match a precomputed plan), so each
    module owns one fresh materialized view. A view's recycled ``id()`` must
    not turn into a cache hit for the next module.
    """

    hidden = 4
    processor = _processor(max_tokens=64, hidden=hidden)
    names = [
        *(
            f"mlp.experts.{expert}.{proj}"
            for expert in range(10)
            for proj in ("gate_proj", "up_proj", "down_proj")
        ),
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
    ]

    captured_rows = {}
    for batch in range(3):
        processor._set_current_batch_index(batch)
        for index, name in enumerate(names):
            length = 2 + (index + batch) % 5
            captured_rows[name] = captured_rows.get(name, 0) + length
            # Every module's rows carry its own index, so receiving another
            # module's rows is visible in the values and in the row count.
            processor._record_input_feature(
                name, torch.full((1, length, hidden), float(index))
            )

    state = _AWQLayerState(modules={name: nn.Linear(hidden, hidden) for name in names})
    features = processor._layer_input_features(state)

    bad = []
    shape_bad = []
    for index, name in enumerate(names):
        values = features[name][0, :, 0].tolist()
        if set(values) != {float(index)} or len(values) != captured_rows[name]:
            bad.append((name, sorted(set(values)), len(values), captured_rows[name]))
        if features[name].shape != (1, captured_rows[name], hidden):
            shape_bad.append((name, tuple(features[name].shape), captured_rows[name]))
        if len(processor.tasks[name]["provenance"]) != captured_rows[name]:
            shape_bad.append((name, "provenance", len(processor.tasks[name]["provenance"])))

    assert bad == [], f"{len(bad)}/{len(names)} modules got another module's rows: {bad}"
    assert shape_bad == [], f"{len(shape_bad)}/{len(names)} modules lost or duplicated rows: {shape_bad}"


def test_reservoir_replay_kwargs_stay_aligned_with_each_modules_rows():
    """Packed positions and mask must describe the module's own retained rows."""

    hidden = 4
    processor = _processor(max_tokens=64, hidden=hidden, batch_lengths=[8, 8, 8])
    names = [
        f"mlp.experts.{expert}.{proj}"
        for expert in range(6)
        for proj in ("gate_proj", "up_proj")
    ]

    captured_rows = {}
    for batch in range(3):
        processor._set_current_batch_index(batch)
        for index, name in enumerate(names):
            length = 2 + (index + batch) % 5
            captured_rows[name] = captured_rows.get(name, 0) + length
            processor._record_input_feature(
                name, torch.full((1, length, hidden), float(index))
            )

    state = _AWQLayerState(modules={name: nn.Linear(hidden, hidden) for name in names})
    features = processor._layer_input_features(state)

    bad = []
    for index, name in enumerate(names):
        if set(features[name][0, :, 0].tolist()) != {float(index)}:
            bad.append((name, "rows", sorted(set(features[name][0, :, 0].tolist()))))
            continue
        kwargs = processor._awq_feature_kwargs[name]
        positions = kwargs.get("position_ids")
        mask = kwargs.get("attention_mask")
        if positions is None or int(positions.shape[-1]) != captured_rows[name]:
            bad.append(
                (name, "positions", None if positions is None else tuple(positions.shape), captured_rows[name])
            )
        elif mask is not None and int(mask.shape[-1]) != captured_rows[name]:
            bad.append((name, "mask", tuple(mask.shape), captured_rows[name]))

    assert bad == [], f"{len(bad)}/{len(names)} modules got another module's replay kwargs: {bad}"


def test_reservoir_cap_keeps_rows_aligned_with_their_provenance():
    """The in-place merge must keep each module's own rows, without loss or duplication."""

    hidden = 2
    cap = 12
    processor = _processor(max_tokens=cap, hidden=hidden)
    names = [f"mlp.experts.{expert}.down_proj" for expert in range(5)]
    lengths = [5, 4, 6]

    for batch, length in enumerate(lengths):
        processor._set_current_batch_index(batch)
        for index, name in enumerate(names):
            # Unique value per (module, batch, row): module 1000s, batch 100s.
            values = torch.arange(length, dtype=torch.float32) + 100.0 * batch + 1000.0 * index
            processor._record_input_feature(
                name, values.reshape(1, length, 1).expand(1, length, hidden).clone()
            )

    state = _AWQLayerState(modules={name: nn.Linear(hidden, hidden) for name in names})
    features = processor._layer_input_features(state)

    for index, name in enumerate(names):
        feature = features[name]
        provenance = processor.tasks[name]["provenance"]
        assert feature.shape == (1, cap, hidden)
        assert len(provenance) == cap
        assert len(set(provenance)) == cap, "a retained row must not be replayed twice"
        for packed_row, (source, row) in enumerate(provenance):
            expected = 1000.0 * index + 100.0 * int(source) + float(row)
            assert float(feature[0, packed_row, 0]) == expected
            assert int(source) < len(lengths)
            assert 0 <= int(row) < lengths[int(source)]


def test_plan_path_siblings_still_share_one_capture():
    """q/k/v read one activation object, so they must still share tensor and kwargs."""

    hidden = 8
    sequence = 16
    batches = 4
    processor = _processor(
        max_tokens=16, hidden=hidden, batch_lengths=[sequence] * batches
    )
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

    for batch in range(batches):
        processor._set_current_batch_index(batch)
        activation = torch.randn(1, sequence, hidden)
        for name in names:
            processor._record_input_feature(name, activation)

    for tensors in zip(*(processor.tasks[name]["inputs"] for name in names)):
        assert tensors[0] is tensors[1] is tensors[2]

    state = _AWQLayerState(modules={name: nn.Linear(hidden, hidden) for name in names})
    features = processor._layer_input_features(state)
    assert features[names[0]] is features[names[1]] is features[names[2]]
    kwargs = processor._awq_feature_kwargs
    assert kwargs[names[0]] is kwargs[names[1]] is kwargs[names[2]]
