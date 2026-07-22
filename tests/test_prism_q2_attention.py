from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.triton_utils.q2_attention import install_prism_q2_gqa_attention


ATTENTION_CALLS = 0


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    batch, heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, heads, repeats, sequence, head_dim)
    return expanded.reshape(batch, heads * repeats, sequence, head_dim)


def _reference_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    del module, kwargs
    global ATTENTION_CALLS
    ATTENTION_CALLS += 1
    output = F.scaled_dot_product_attention(
        query,
        _repeat_kv(key, 2),
        _repeat_kv(value, 2),
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
    )
    return output.transpose(1, 2).contiguous(), None


class _AttentionInterfaces:
    def get_interface(self, key, fallback):
        del key, fallback
        return _reference_attention


ALL_ATTENTION_FUNCTIONS = _AttentionInterfaces()


def eager_attention_forward(*args, **kwargs):
    return _reference_attention(*args, **kwargs)


class Qwen3Attention(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.q_proj = torch.nn.Identity()
        self.q_proj._gptqmodel_prism_q2_qkv_device = device
        self.scaling = 128**-0.5
        self.sliding_window = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", eager_attention_forward)
        return attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0,
            scaling=self.scaling,
            **kwargs,
        )


def _model_with_attention(attention: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        model_type="qwen3",
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
    )
    model.attention = attention
    return model.eval()


def test_prism_q2_attention_installer_preserves_non_q2_fallback():
    model = _model_with_attention(Qwen3Attention(torch.device("cpu")))

    assert install_prism_q2_gqa_attention(model) == 0
    assert not hasattr(model.attention, "_gptqmodel_prism_q2_attention")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="Prism Q2 fused GQA attention specialization requires sm80 CUDA",
)
def test_prism_q2_attention_sm80_matches_sdpa_and_preserves_local_fallbacks():
    pytest.importorskip("triton")
    global ATTENTION_CALLS
    torch.manual_seed(1234)
    device = torch.device("cuda", torch.cuda.current_device())
    attention = Qwen3Attention(device)
    attention._gptqmodel_prism_q2_qkv = True
    model = _model_with_attention(attention)
    query = torch.randn(1, 16, 1, 128, device=device, dtype=torch.float16)
    key = torch.randn(1, 8, 84, 128, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    attention_mask = torch.zeros(1, 1, 1, 84, device=device, dtype=torch.bool)
    attention_mask[..., :65] = True
    expected, _ = _reference_attention(model.attention, query, key, value, attention_mask, scaling=attention.scaling)
    global_interfaces = Qwen3Attention.forward.__globals__["ALL_ATTENTION_FUNCTIONS"]

    assert install_prism_q2_gqa_attention(model) == 1
    assert install_prism_q2_gqa_attention(model) == 0
    ATTENTION_CALLS = 0
    with torch.inference_mode():
        actual, weights = model.attention(query, key, value, attention_mask)

    difference = (actual.float() - expected.float()).abs()
    assert ATTENTION_CALLS == 0
    assert weights is None
    assert Qwen3Attention.forward.__globals__["ALL_ATTENTION_FUNCTIONS"] is global_interfaces
    assert model.attention.forward.__func__.__globals__["ALL_ATTENTION_FUNCTIONS"] is not global_interfaces
    assert actual.shape == expected.shape == (1, 1, 16, 128)
    assert actual.dtype == expected.dtype == torch.float16
    assert difference.mean().item() <= 0.0001
    assert difference.max().item() <= 0.001

    empty_mask = torch.zeros_like(attention_mask)
    empty_expected, _ = _reference_attention(
        model.attention,
        query,
        key,
        value,
        empty_mask,
        scaling=attention.scaling,
    )
    ATTENTION_CALLS = 0
    with torch.inference_mode():
        empty_actual, _ = model.attention(query, key, value, empty_mask)
    assert ATTENTION_CALLS == 0
    assert torch.equal(empty_actual, empty_expected)

    additive_mask = torch.zeros(1, 1, 1, 84, device=device, dtype=torch.float16)
    additive_mask[..., 65:] = torch.finfo(torch.float16).min
    ATTENTION_CALLS = 0
    with torch.inference_mode():
        fallback, _ = model.attention(query, key, value, additive_mask)
    fallback_expected, _ = _reference_attention(
        model.attention,
        query,
        key,
        value,
        additive_mask,
        scaling=attention.scaling,
    )
    assert ATTENTION_CALLS == 2
    assert torch.equal(fallback, fallback_expected)

    dynamic_key = key[:, :, :65, :].contiguous()
    dynamic_value = value[:, :, :65, :].contiguous()
    ATTENTION_CALLS = 0
    with torch.inference_mode():
        dynamic, _ = model.attention(query, dynamic_key, dynamic_value, None)
    dynamic_expected, _ = _reference_attention(
        model.attention,
        query,
        dynamic_key,
        dynamic_value,
        None,
        scaling=attention.scaling,
    )
    assert ATTENTION_CALLS == 2
    assert torch.equal(dynamic, dynamic_expected)

    model.train()
    ATTENTION_CALLS = 0
    training, _ = model.attention(query, key, value, attention_mask)
    training_expected, _ = _reference_attention(
        model.attention,
        query,
        key,
        value,
        attention_mask,
        scaling=attention.scaling,
    )
    assert ATTENTION_CALLS == 2
    assert torch.equal(training, training_expected)
