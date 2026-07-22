import gc
import weakref

import pytest
import torch
from transformers import Qwen3Config, StaticCache

from gptqmodel.nn_modules.triton_utils.q2_cache import install_prism_q2_static_cache


class Qwen3Attention(torch.nn.Module):
    def __init__(self, *, prism_q2: bool):
        super().__init__()
        self.q_proj = torch.nn.Identity()
        self._gptqmodel_prism_q2_qkv = prism_q2


class _CacheModel(torch.nn.Module):
    def __init__(self, config: Qwen3Config, *, prism_q2: bool):
        super().__init__()
        self.config = config
        self.attention = Qwen3Attention(prism_q2=prism_q2)


def _config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=1,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )


def test_prism_q2_static_cache_installer_preserves_non_q2_fallback():
    config = _config()
    model = _CacheModel(config, prism_q2=False).eval()
    cache = StaticCache(config=config, max_cache_len=84)

    assert install_prism_q2_static_cache(model, cache, device=torch.device("cpu")) == 0
    assert not hasattr(cache.layers[0], "_gptqmodel_prism_q2_cache")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="Prism Q2 fused static-cache specialization requires sm80 CUDA",
)
def test_prism_q2_static_cache_sm80_matches_transformers_and_replays_graph():
    pytest.importorskip("triton")
    torch.manual_seed(1234)
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config()
    model = _CacheModel(config, prism_q2=True).eval()
    reference = StaticCache(config=config, max_cache_len=84)
    candidate = StaticCache(config=config, max_cache_len=84)

    assert install_prism_q2_static_cache(model, candidate, device=device) == 1
    assert install_prism_q2_static_cache(model, candidate, device=device) == 0

    prefill_key = torch.randn(1, 8, 64, 128, device=device, dtype=torch.float16)
    prefill_value = torch.randn_like(prefill_key)
    reference.update(prefill_key, prefill_value, 0)
    candidate.update(prefill_key, prefill_value, 0)
    assert torch.equal(candidate.layers[0].keys, reference.layers[0].keys)
    assert torch.equal(candidate.layers[0].values, reference.layers[0].values)
    assert candidate.layers[0].cumulative_length.item() == 64

    scratch = StaticCache(config=config, max_cache_len=84)
    assert install_prism_q2_static_cache(model, scratch, device=device) == 1
    scratch.update(prefill_key, prefill_value, 0)
    scratch_layer_ref = weakref.ref(scratch.layers[0])
    del scratch
    gc.collect()
    assert scratch_layer_ref() is None

    for position in range(64, 67):
        key = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
        value = torch.randn_like(key)
        expected = reference.update(key, value, 0)
        actual = candidate.update(key, value, 0)
        torch.cuda.synchronize(device)
        assert actual[0].data_ptr() == candidate.layers[0].keys.data_ptr()
        assert actual[1].data_ptr() == candidate.layers[0].values.data_ptr()
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])
        assert candidate.layers[0].cumulative_length.item() == position + 1

    graph_key = torch.empty(1, 8, 1, 128, device=device, dtype=torch.float16)
    graph_value = torch.empty_like(graph_key)
    candidate.layers[0].cumulative_length.fill_(64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate.update(graph_key, graph_value, 0)
    replay_key = torch.randn_like(graph_key)
    replay_value = torch.randn_like(graph_value)
    candidate.layers[0].cumulative_length.fill_(64)
    graph_key.copy_(replay_key)
    graph_value.copy_(replay_value)
    graph.replay()
    torch.cuda.synchronize(device)
    assert candidate.layers[0].cumulative_length.item() == 65
    assert torch.equal(candidate.layers[0].keys[:, :, 64:65, :], replay_key)
    assert torch.equal(candidate.layers[0].values[:, :, 64:65, :], replay_value)

    model.train()
    fallback_key = torch.randn_like(graph_key)
    fallback_value = torch.randn_like(graph_value)
    candidate.update(fallback_key, fallback_value, 0)
    torch.cuda.synchronize(device)
    assert candidate.layers[0].cumulative_length.item() == 66
    assert torch.equal(candidate.layers[0].keys[:, :, 65:66, :], fallback_key)
    assert torch.equal(candidate.layers[0].values[:, :, 65:66, :], fallback_value)
