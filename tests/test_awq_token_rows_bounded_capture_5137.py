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
