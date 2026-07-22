from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.triton_utils.q2_rotary import install_prism_q2_rotary


ROTARY_CALLS = 0


def _reference_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def rotate_half(values: torch.Tensor) -> torch.Tensor:
        midpoint = values.shape[-1] // 2
        return torch.cat((-values[..., midpoint:], values[..., :midpoint]), dim=-1)

    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    global ROTARY_CALLS
    ROTARY_CALLS += 1
    return _reference_rotary(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)


class Qwen3Attention(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.q_proj = torch.nn.Identity()
        self.q_proj._gptqmodel_prism_q2_qkv_device = device

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)


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


def test_prism_q2_rotary_installer_preserves_non_q2_fallback():
    model = _model_with_attention(Qwen3Attention(torch.device("cpu")))

    assert install_prism_q2_rotary(model) == 0
    assert not hasattr(model.attention, "_gptqmodel_prism_q2_rotary")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="Prism Q2 fused rotary specialization requires sm80 CUDA",
)
def test_prism_q2_rotary_sm80_is_exact_and_preserves_local_fallbacks():
    pytest.importorskip("triton")
    global ROTARY_CALLS
    torch.manual_seed(1234)
    device = torch.device("cuda", torch.cuda.current_device())
    attention = Qwen3Attention(device)
    attention._gptqmodel_prism_q2_qkv = True
    model = _model_with_attention(attention)
    query = torch.randn(1, 16, 1, 128, device=device, dtype=torch.float16)
    key = torch.randn(1, 8, 1, 128, device=device, dtype=torch.float16)
    cos = torch.randn(1, 1, 128, device=device, dtype=torch.float16)
    sin = torch.randn(1, 1, 128, device=device, dtype=torch.float16)
    expected = _reference_rotary(query, key, cos, sin)
    global_apply = Qwen3Attention.forward.__globals__["apply_rotary_pos_emb"]

    assert install_prism_q2_rotary(model) == 1
    assert install_prism_q2_rotary(model) == 0
    ROTARY_CALLS = 0
    with torch.inference_mode():
        actual = model.attention(query, key, cos, sin)

    assert ROTARY_CALLS == 0
    assert Qwen3Attention.forward.__globals__["apply_rotary_pos_emb"] is global_apply
    assert model.attention.forward.__func__.__globals__["apply_rotary_pos_emb"] is not global_apply
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])

    prefill_query = torch.randn(1, 16, 2, 128, device=device, dtype=torch.float16)
    prefill_key = torch.randn(1, 8, 2, 128, device=device, dtype=torch.float16)
    prefill_cos = torch.randn(1, 2, 128, device=device, dtype=torch.float16)
    prefill_sin = torch.randn(1, 2, 128, device=device, dtype=torch.float16)
    ROTARY_CALLS = 0
    with torch.inference_mode():
        prefill = model.attention(prefill_query, prefill_key, prefill_cos, prefill_sin)
    prefill_expected = _reference_rotary(prefill_query, prefill_key, prefill_cos, prefill_sin)
    assert ROTARY_CALLS == 1
    assert torch.equal(prefill[0], prefill_expected[0])
    assert torch.equal(prefill[1], prefill_expected[1])

    model.train()
    ROTARY_CALLS = 0
    training = model.attention(query, key, cos, sin)
    assert ROTARY_CALLS == 1
    assert torch.equal(training[0], expected[0])
    assert torch.equal(training[1], expected[1])
