# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for bounded activation token capture (Issue #5137).

Verifies that:
1. Peak host RAM under AWQ `token_rows` stays bounded by the sampling budget
   and never grows linearly with calibration corpus size (docs/tokens).
2. Sibling projections (`q_proj`, `k_proj`, `v_proj`) share a single host copy.
3. Provenance and original unpadded batch lengths rebuild block-diagonal causal
   attention masks and position IDs with exact fidelity.
4. Dynamic streaming reservoir bounds dynamically routed MoE expert activations.
5. ParoQuant bounded capture bounds activations per batch and layer.
"""

import threading
from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQLayerState
from gptqmodel.looper.paro_processor import ParoProcessor, _ParoLayerState
from gptqmodel.quantization.config import CalibrationConfig


def _create_mock_cache(batch_lengths, hidden=64):
    """Create a mock inputs_cache with specified batch lengths and attention masks."""
    cache = SimpleNamespace()
    cache.layer_inputs = []
    cache.attention_masks = []
    cache.position_ids = []
    cache.layer_input_kwargs = []

    for b, length in enumerate(batch_lengths):
        # 1-batch tensor with shape [1, length, hidden]
        t = torch.randn(1, length, hidden)
        cache.layer_inputs.append([t])
        # Planar bool keep mask [1, length]
        mask = torch.ones(1, length, dtype=torch.bool)
        cache.attention_masks.append(mask)
        # Position ids [1, length]
        pos = torch.arange(length, dtype=torch.long).unsqueeze(0)
        cache.position_ids.append(pos)
        cache.layer_input_kwargs.append({})

    return cache


def _create_awq_processor(cache, policy_mode="token_rows", max_tokens=None):
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor._mask_tls = threading.local()
    processor.tasks = {}
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
    processor.inputs_cache = cache
    processor.calibration = CalibrationConfig(exclude_attention_sinks=False)

    gptq_model = SimpleNamespace()
    gptq_model.awq_input_feature_aggregation = lambda name: {
        "mode": policy_mode,
        **({"max_tokens": max_tokens} if max_tokens is not None else {}),
    }
    processor.gptq_model = gptq_model
    return processor


def test_awq_token_rows_bounded_capture_does_not_scale_with_batch_count():
    """Peak host tokens in entry['inputs'] must not grow linearly with batch count."""
    batch_count = 32
    seq_len = 512
    hidden = 64
    lengths = [seq_len] * batch_count  # 16,384 total tokens
    cache = _create_mock_cache(lengths, hidden=hidden)

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=256)
    module_name = "self_attn.q_proj"

    # Simulate calibration passes across all 32 batches
    for b in range(batch_count):
        processor._set_current_batch_index(b)
        feat = torch.randn(1, seq_len, hidden)
        processor._record_input_feature(module_name, feat)

    entry = processor.tasks[module_name]
    assert entry["raw_tokens"] == 16384
    # The stored slices must sum to at most max_tokens (256), NOT 16,384!
    captured_tokens = sum(t.shape[-2] for t in entry["inputs"])
    assert captured_tokens == 256

    # Test layer aggregation collapse
    state = _AWQLayerState(modules={module_name: nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    assert features[module_name].shape == (1, 256, hidden)
    assert processor._feature_stats[module_name]["raw_tokens"] == 16384
    assert processor._feature_stats[module_name]["retained_tokens"] == 256
    assert processor._feature_stats[module_name]["batches"] == 32


def test_awq_token_rows_bounded_capture_sibling_sharing():
    """q_proj, k_proj, and v_proj must share the exact same sampled host tensors."""
    batch_count = 8
    seq_len = 256
    hidden = 32
    lengths = [seq_len] * batch_count
    cache = _create_mock_cache(lengths, hidden=hidden)

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=64)

    modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

    for b in range(batch_count):
        processor._set_current_batch_index(b)
        # Sibling modules receive the exact same activation tensor in forward
        shared_activation = torch.randn(1, seq_len, hidden)
        for mod in modules:
            processor._record_input_feature(mod, shared_activation)

    # Check that captured slices are shared across sibling tasks
    q_entry = processor.tasks["self_attn.q_proj"]
    k_entry = processor.tasks["self_attn.k_proj"]
    v_entry = processor.tasks["self_attn.v_proj"]

    for t_q, t_k, t_v in zip(q_entry["inputs"], k_entry["inputs"], v_entry["inputs"]):
        assert t_q is t_k is t_v

    # Check that collapsed features are also shared
    state = _AWQLayerState(modules={m: nn.Linear(hidden, hidden) for m in modules})
    features = processor._layer_input_features(state)
    assert features["self_attn.q_proj"] is features["self_attn.k_proj"] is features["self_attn.v_proj"]


def test_awq_token_rows_provenance_and_mask_reconstruction():
    """Reconstructed block-diagonal attention mask and positions must be faithful."""
    lengths = [100, 200, 150]
    hidden = 32
    cache = _create_mock_cache(lengths, hidden=hidden)

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=60)
    module_name = "self_attn.q_proj"

    for b, length in enumerate(lengths):
        processor._set_current_batch_index(b)
        feat = torch.randn(1, length, hidden)
        processor._record_input_feature(module_name, feat)

    state = _AWQLayerState(modules={module_name: nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    assert features[module_name].shape[-2] == 60
    kwargs = processor._awq_feature_kwargs[module_name]

    assert "attention_mask" in kwargs
    mask = kwargs["attention_mask"]
    # Mask shape should match packed tokens along the sequence dimension
    assert mask.shape[-1] == 60

    if "position_ids" in kwargs and kwargs["position_ids"] is not None:
        pos = kwargs["position_ids"]
        assert pos.shape[-1] == 60
        # Position ids should reflect original offsets within documents (< max(lengths))
        assert int(pos.max()) < max(lengths)


def test_awq_token_rows_streaming_reservoir_dynamic_moe():
    """Dynamic token counts (routed experts) must stay bounded via streaming reservoir."""
    hidden = 32
    # Cache without pre-known batch lengths to trigger dynamic reservoir
    cache = SimpleNamespace()
    cache.layer_inputs = None
    cache.attention_masks = []
    cache.position_ids = []
    cache.layer_input_kwargs = []

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=100)
    module_name = "mlp.experts.0.gate_proj"

    # Feed 50 batches of variable token counts (totaling 2,500 tokens)
    for b in range(50):
        processor._set_current_batch_index(b)
        token_count = 50 + (b % 5)
        feat = torch.randn(1, token_count, hidden)
        processor._record_input_feature(module_name, feat)

    entry = processor.tasks[module_name]
    captured_tokens = sum(t.shape[-2] for t in entry["inputs"])
    # Reservoir must have bounded accumulated tokens to at most budget (100)
    assert captured_tokens <= 100

    state = _AWQLayerState(modules={module_name: nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    assert features[module_name].shape[-2] <= 100


def test_paro_bounded_capture_and_layer_aggregation():
    """ParoProcessor must bound per-batch capture and layer features to target_rows."""
    qcfg = SimpleNamespace(opt_train_samples=16, opt_validation_samples=8)
    processor = ParoProcessor.__new__(ParoProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor.tasks = {}
    processor.qcfg = qcfg

    batch_count = 20
    seq_len = 100
    hidden = 16
    lengths = [seq_len] * batch_count  # 2,000 tokens
    cache = _create_mock_cache(lengths, hidden=hidden)
    processor.inputs_cache = cache

    module_name = "mlp.down_proj"
    for b in range(batch_count):
        processor._set_current_batch_index(b)
        feat = torch.randn(1, seq_len, hidden)
        processor._record_input_feature(module_name, feat)

    entry = processor.tasks[module_name]
    # Each batch was downsampled to per_batch_quota
    # target_rows = 16 + 8 = 24.
    # per_batch_quota = max(1, 2 * (24 + 19) // 20) = 4.
    for tensor in entry["inputs"]:
        assert tensor.shape[-2] <= 4

    state = _ParoLayerState(modules={module_name: nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    # Output must be downsampled to exactly target_rows (24)
    assert features[module_name].shape[-2] == 24


def test_awq_token_rows_out_of_order_capture_keeps_provenance_aligned():
    """Out-of-order captures must replay rows with their own batch's positions.

    Capture hooks can run in device/runtime order rather than calibration-batch
    order. ``_layer_input_features`` reorders the per-batch tensors by batch
    index, so the recorded provenance and per-batch row counts must follow the
    same permutation. Otherwise the rebuilt position ids (and block-diagonal
    mask) describe a different calibration batch than the rows they accompany.
    """

    lengths = [4, 6]
    hidden = 8
    cache = _create_mock_cache(lengths, hidden=hidden)

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=10)
    module_name = "self_attn.q_proj"

    batch0 = torch.arange(lengths[0] * hidden, dtype=torch.float32).reshape(1, lengths[0], hidden)
    batch1 = (
        torch.arange(lengths[1] * hidden, dtype=torch.float32).reshape(1, lengths[1], hidden)
        + 1000.0
    )

    # Batch 1 arrives before batch 0.
    processor._set_current_batch_index(1)
    processor._record_input_feature(module_name, batch1)
    processor._set_current_batch_index(0)
    processor._record_input_feature(module_name, batch0)

    state = _AWQLayerState(modules={module_name: nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    replayed = features[module_name]

    # Rows are concatenated in calibration-batch order: batch 0 (4) then batch 1 (6).
    assert replayed.shape == (1, 10, hidden)
    assert torch.equal(replayed[0, : lengths[0]], batch0[0])
    assert torch.equal(replayed[0, lengths[0] :], batch1[0])

    position_ids = processor._awq_feature_kwargs[module_name]["position_ids"]
    assert position_ids.tolist() == [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5]]


def test_awq_token_rows_streaming_reservoir_single_row_budget():
    """A one-row sampling budget must not divide by ``M - 1`` in the reservoir."""

    hidden = 8
    cache = SimpleNamespace()
    cache.layer_inputs = None
    cache.attention_masks = []
    cache.position_ids = []
    cache.layer_input_kwargs = []

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=1)
    module_name = "mlp.experts.0.gate_proj"

    for b in range(4):
        processor._set_current_batch_index(b)
        processor._record_input_feature(module_name, torch.randn(1, 3, hidden))

    entry = processor.tasks[module_name]
    assert sum(t.shape[-2] for t in entry["inputs"]) == 1

    state = _AWQLayerState(modules={module_name: nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    assert features[module_name].shape == (1, 1, hidden)


def test_awq_token_row_lengths_memo_tracks_layer_inputs_identity():
    """Cached batch lengths must refresh when the looper installs new layer inputs."""

    cache = _create_mock_cache([4, 6], hidden=8)
    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=10)

    first = processor._extract_cache_batch_lengths()
    assert first == [4, 6]
    assert processor._extract_cache_batch_lengths() is first

    next_cache = _create_mock_cache([3], hidden=8)
    cache.layer_inputs = next_cache.layer_inputs
    cache.attention_masks = next_cache.attention_masks
    cache.position_ids = next_cache.position_ids

    assert processor._extract_cache_batch_lengths() == [3]


def test_awq_token_rows_out_of_order_sampled_rows_keep_provenance_aligned():
    """Sampled (quota < length) out-of-order captures keep rows and positions paired."""

    lengths = [4, 6]
    hidden = 8
    cache = _create_mock_cache(lengths, hidden=hidden)

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=6)
    module_name = "self_attn.q_proj"

    batch0 = torch.arange(lengths[0] * hidden, dtype=torch.float32).reshape(1, lengths[0], hidden)
    batch1 = (
        torch.arange(lengths[1] * hidden, dtype=torch.float32).reshape(1, lengths[1], hidden)
        + 1000.0
    )

    # Batch 1 arrives before batch 0; each batch keeps an evenly sampled 3 rows.
    processor._set_current_batch_index(1)
    processor._record_input_feature(module_name, batch1)
    processor._set_current_batch_index(0)
    processor._record_input_feature(module_name, batch0)

    state = _AWQLayerState(modules={module_name: nn.Linear(hidden, hidden)})
    features = processor._layer_input_features(state)
    replayed = features[module_name]

    # Batch 0 keeps rows [0, 1, 3]; batch 1 keeps rows [0, 2, 5].
    expected_rows = [0, 8, 24, 1000, 1016, 1040]
    assert [int(value) for value in replayed[0, :, 0].tolist()] == expected_rows

    position_ids = processor._awq_feature_kwargs[module_name]["position_ids"]
    assert position_ids.tolist() == [[0, 1, 3, 0, 2, 5]]


def test_awq_token_row_quotas_skip_empty_batches():
    """Empty batches must not consume sampling budget or select zero rows."""

    quotas, local = AWQProcessor._compute_token_row_quotas([0, 7, 3], 1)
    assert quotas == [0, 1, 0]
    assert local[1].tolist() == [0]

    quotas, local = AWQProcessor._compute_token_row_quotas([0, 7, 3], 4)
    assert quotas == [0, 2, 2]
    assert local[1].tolist() == [0, 6]
    assert local[2].tolist() == [0, 2]


def test_awq_token_rows_streaming_reservoir_keeps_every_quarter_proportional():
    """A routed expert's bounded sample must not drift toward the tail.

    Once the buffer fills, resampling the kept rows evenly together with each
    new batch (the pre-#5204 behaviour) makes the kept rows weigh the same as
    the fresh batch, so earlier batches decay geometrically and the sample ends
    up dominated by the last few documents. A uniform priority reservoir keeps a
    share proportional to each batch's length, so every quarter of the
    calibration stream contributes about a quarter of the rows.
    """

    hidden = 8
    cache = SimpleNamespace()
    cache.layer_inputs = None  # cache-less: routed expert with varying rows
    cache.attention_masks = []
    cache.position_ids = []
    cache.layer_input_kwargs = []

    processor = _create_awq_processor(cache, policy_mode="token_rows", max_tokens=906)
    module_name = "mlp.experts.1.gate_proj"

    batch_count = 128
    lengths = [40 + (b % 7) * 6 for b in range(batch_count)]
    for b, length in enumerate(lengths):
        processor._set_current_batch_index(b)
        processor._record_input_feature(module_name, torch.randn(1, length, hidden))

    entry = processor.tasks[module_name]
    # Provenance is kept as integer tensors during capture and materialized for
    # replay, so build the tuple list the same way `_layer_input_features` does.
    processor._materialize_reservoir_provenance(entry)
    provenance = entry["provenance"]
    counts = entry["batch_row_counts"]

    # The capture stays bounded by the sampling budget.
    assert len(provenance) == 906
    assert sum(int(t.shape[-2]) for t in entry["inputs"]) == 906

    # Every quarter of the calibration stream keeps about a quarter of the rows.
    kept = [0, 0, 0, 0]
    raw = [0, 0, 0, 0]
    for slot, count in enumerate(counts):
        raw[min(3, 4 * slot // len(counts))] += count
    for slot, _row in provenance:
        kept[min(3, 4 * slot // len(counts))] += 1

    for quarter in range(4):
        share = kept[quarter] / len(provenance)
        assert 0.15 <= share <= 0.35, (
            f"quarter {quarter} kept {share:.1%} of the rows "
            f"(raw share {raw[quarter] / sum(counts):.1%}); kept={kept}"
        )

    # No part of the stream is dropped entirely.
    contributing = {slot for slot, _row in provenance}
    assert len(contributing) == batch_count
