# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Bounded staged-batch construction must not change any training input."""

import pytest
import torch

from gptqmodel.quantization.gsq_batching import (
    LazyLlamaStageBatches,
    llama_stage_batches,
)


def documents(count, tokens, width, rotary_width=4):
    generator = torch.Generator().manual_seed(0)
    captured = []
    for _ in range(count):
        hidden = torch.randn(1, tokens, width, generator=generator)
        captured.append((hidden, {
            'position_ids': torch.arange(tokens).unsqueeze(0),
            'attention_mask': None,
            'position_embeddings': (
                torch.randn(1, tokens, rotary_width, generator=generator),
                torch.randn(1, tokens, rotary_width, generator=generator),
            ),
        }))
    return captured


def assert_batches_identical(eager, lazy):
    assert isinstance(lazy, LazyLlamaStageBatches)
    assert len(eager) == len(lazy)
    for batch_index in range(len(eager)):
        eager_microbatches, lazy_microbatches = eager[batch_index], lazy[batch_index]
        assert len(eager_microbatches) == len(lazy_microbatches)
        for (eager_batch, eager_count), (lazy_batch, lazy_count) in zip(
                eager_microbatches, lazy_microbatches):
            assert eager_count == lazy_count
            eager_hidden, eager_kwargs, eager_mask = eager_batch
            lazy_hidden, lazy_kwargs, lazy_mask = lazy_batch
            assert torch.equal(eager_hidden, lazy_hidden)
            assert (eager_mask is None) == (lazy_mask is None)
            if eager_mask is not None:
                assert torch.equal(eager_mask, lazy_mask)
            assert set(eager_kwargs) == set(lazy_kwargs)
            for name, eager_value in eager_kwargs.items():
                lazy_value = lazy_kwargs[name]
                if eager_value is None or lazy_value is None:
                    assert eager_value is lazy_value
                elif isinstance(eager_value, bool) or isinstance(lazy_value, bool):
                    assert eager_value == lazy_value
                elif isinstance(eager_value, torch.Tensor):
                    assert torch.equal(eager_value, lazy_value)
                else:
                    assert len(eager_value) == len(lazy_value)
                    for left, right in zip(eager_value, lazy_value):
                        assert torch.equal(left, right)


@pytest.mark.parametrize('implicit_causal', [True, False])
def test_lazy_staging_matches_eager_staging(implicit_causal):
    captured = documents(8, 16, 8)
    kwargs = dict(batch_size=4, microbatch_size=2, device='cpu',
                  implicit_causal=implicit_causal)
    eager = llama_stage_batches(captured, lazy=False, **kwargs)
    lazy = llama_stage_batches(captured, lazy=True, **kwargs)
    assert_batches_identical(eager, lazy)


def test_lazy_staging_preserves_partial_trailing_batch():
    captured = documents(6, 16, 8)
    kwargs = dict(batch_size=4, microbatch_size=3, device='cpu', implicit_causal=True)
    eager = llama_stage_batches(captured, lazy=False, **kwargs)
    lazy = llama_stage_batches(captured, lazy=True, **kwargs)
    assert len(eager) == len(lazy) == 2
    assert_batches_identical(eager, lazy)


def test_lazy_staging_materializes_one_batch_at_a_time():
    """Peak staging must be one optimizer batch, not the whole corpus."""
    import tracemalloc

    captured = documents(128, 64, 32)
    kwargs = dict(batch_size=32, microbatch_size=16, device='cpu', implicit_causal=True)

    tracemalloc.start()
    lazy = llama_stage_batches(captured, lazy=True, **kwargs)
    assert isinstance(lazy, LazyLlamaStageBatches)
    # Construction must not touch the documents at all, so nothing large is held.
    _, lazy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    llama_stage_batches(captured, lazy=False, **kwargs)
    _, eager_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert lazy_peak < eager_peak / 4, (lazy_peak, eager_peak)


def test_input_cache_releases_captured_tensors():
    """A per-layer caller must be able to reclaim a GPU-resident capture."""
    from gptqmodel.looper.input_cache import InputCache

    cache = InputCache(
        [[torch.zeros(2, 4, 4)]],
        [{'use_cache': False}],
        [torch.zeros(1, 4)],
        [None],
    )
    assert cache.layer_inputs and cache.layer_input_kwargs
    cache.cleanup()
    assert cache.layer_inputs == []
    assert cache.layer_input_kwargs == []
    assert cache.position_ids == []
    assert cache.attention_masks == []
    assert cache.src_inputs == []
