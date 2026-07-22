import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.triton_utils.q2_residual import (
    _PRISM_Q2_RESIDUAL_CONTEXT,
    _is_supported_decoder_forward,
    install_prism_q2_residuals,
)


class Qwen3Attention(torch.nn.Module):
    def __init__(self, output_projection: torch.nn.Module):
        super().__init__()
        self.o_proj = output_projection
        self._gptqmodel_prism_q2_qkv = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, position_ids, past_key_values, use_cache, position_embeddings, kwargs
        return self.o_proj(hidden_states), None


class Qwen3MLP(torch.nn.Module):
    def __init__(self, down_projection: torch.nn.Module):
        super().__init__()
        self.down_proj = down_projection
        self._gptqmodel_prism_q2_swiglu = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(hidden_states.repeat(1, 1, 3))


class Qwen3DecoderLayer(torch.nn.Module):
    def __init__(self, output_projection: torch.nn.Module, down_projection: torch.nn.Module):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.self_attn = Qwen3Attention(output_projection)
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = Qwen3MLP(down_projection)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _ResidualModel(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=2048,
            intermediate_size=6144,
            hidden_act="silu",
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=1,
        )
        self.layer = layer


def _random_q2_projection(in_features: int, out_features: int, seed: int) -> GGUFTritonKernel:
    generator = torch.Generator().manual_seed(seed)
    projection = GGUFTritonKernel(
        bits="q2_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    blocks = projection.qweight.reshape(out_features, in_features // 128, 34)
    scales = torch.full((out_features, in_features // 128), 0.002, dtype=torch.float16)
    with torch.no_grad():
        blocks[..., :2].copy_(scales.view(torch.uint8).reshape(out_features, in_features // 128, 2))
        blocks[..., 2:].copy_(
            torch.randint(0, 256, blocks[..., 2:].shape, dtype=torch.uint8, generator=generator)
        )
    return projection.cuda().eval()


def test_prism_q2_residual_installer_preserves_non_q2_fallback_and_checks_live_forward():
    model = _ResidualModel(Qwen3DecoderLayer(torch.nn.Identity(), torch.nn.Identity())).eval()

    assert _is_supported_decoder_forward(model.layer.forward)
    assert install_prism_q2_residuals(model) == 0
    assert not hasattr(model.layer, "_gptqmodel_prism_q2_residual")

    def changed_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del attention_mask, position_ids, past_key_values, use_cache, position_embeddings, kwargs
        return self.input_layernorm(hidden_states)

    model.layer.forward = MethodType(changed_forward, model.layer)
    assert not _is_supported_decoder_forward(model.layer.forward)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="Prism Q2 fused residual specialization requires sm80 CUDA",
)
def test_prism_q2_residual_sm80_is_exact_and_replays_graph_with_local_fallbacks():
    pytest.importorskip("triton")
    torch.manual_seed(1234)
    output_projection = _random_q2_projection(2048, 2048, 1)
    down_projection = _random_q2_projection(6144, 2048, 2)
    model = _ResidualModel(Qwen3DecoderLayer(output_projection, down_projection)).eval()
    layer = model.layer
    hidden_states = torch.randn(1, 1, 2048, device="cuda", dtype=torch.float16) * 0.02

    with torch.inference_mode():
        expected = layer(hidden_states)
        direct_expected = output_projection(hidden_states)
        assert install_prism_q2_residuals(model) == 1
        assert install_prism_q2_residuals(model) == 0
        actual = layer(hidden_states)
        direct_actual = output_projection(hidden_states)
        torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(direct_actual, direct_expected)
    assert actual.dtype == expected.dtype == torch.float16
    assert not inspect.ismethod(layer._gptqmodel_prism_q2_residual_original_forward)
    assert not inspect.ismethod(output_projection._gptqmodel_prism_q2_residual_original_forward)
    assert _PRISM_Q2_RESIDUAL_CONTEXT.get() is None

    layer._gptqmodel_prism_q2_residual = False
    with torch.inference_mode():
        disabled = layer(hidden_states)
    layer._gptqmodel_prism_q2_residual = True
    assert torch.equal(disabled, expected)

    prefill = torch.randn(1, 2, 2048, device="cuda", dtype=torch.float16) * 0.02
    with torch.inference_mode():
        prefill_expected = layer._gptqmodel_prism_q2_residual_original_forward(layer, prefill)
        prefill_actual = layer(prefill)
    assert torch.equal(prefill_actual, prefill_expected)

    replay_input = torch.randn_like(hidden_states)
    with torch.inference_mode():
        replay_expected = layer._gptqmodel_prism_q2_residual_original_forward(layer, replay_input)
        layer(hidden_states)
        torch.cuda.synchronize()
        graph_input = hidden_states.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = layer(graph_input)
        graph_input.copy_(replay_input)
        graph.replay()
        torch.cuda.synchronize()
    assert torch.equal(graph_output, replay_expected)

    model.training = True
    training_input = torch.randn_like(hidden_states)
    with torch.inference_mode():
        training_expected = layer._gptqmodel_prism_q2_residual_original_forward(layer, training_input)
        training_actual = layer(training_input)
    assert torch.equal(training_actual, training_expected)
