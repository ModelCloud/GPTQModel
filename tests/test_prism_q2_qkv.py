from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.triton_utils.q2_qkv import install_prism_q2_qkv


class Qwen3Attention(torch.nn.Module):
    def __init__(
        self,
        q_proj: torch.nn.Module,
        k_proj: torch.nn.Module,
        v_proj: torch.nn.Module,
    ):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        return query, key, value


def _model_with_attention(attention: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3")
    model.attention = attention
    return model.eval()


def _random_q2_projection(out_features: int, seed: int) -> GGUFTritonKernel:
    generator = torch.Generator().manual_seed(seed)
    projection = GGUFTritonKernel(
        bits="q2_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    blocks = projection.qweight.reshape(out_features, 16, 34)
    scales = torch.full((out_features, 16), 0.01, dtype=torch.float16)
    with torch.no_grad():
        blocks[..., :2].copy_(scales.view(torch.uint8).reshape(out_features, 16, 2))
        blocks[..., 2:].copy_(torch.randint(0, 256, blocks[..., 2:].shape, dtype=torch.uint8, generator=generator))
    return projection.cuda().eval()


def test_prism_q2_qkv_installer_preserves_non_q2_fallback():
    attention = Qwen3Attention(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    )
    model = _model_with_attention(attention)

    assert install_prism_q2_qkv(model) == 0
    assert not hasattr(model.attention, "_gptqmodel_prism_q2_qkv")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="Prism Q2 fused QKV specialization requires sm80 CUDA",
)
def test_prism_q2_qkv_sm80_matches_unfused_decode_and_preserves_prefill_fallback():
    pytest.importorskip("triton")
    torch.manual_seed(1234)
    attention = Qwen3Attention(
        _random_q2_projection(2048, 1),
        _random_q2_projection(1024, 2),
        _random_q2_projection(1024, 3),
    )
    model = _model_with_attention(attention)
    hidden_states = torch.randn(1, 1, 2048, device="cuda", dtype=torch.float16) * 0.1

    with torch.inference_mode():
        expected = model.attention(hidden_states)
        assert install_prism_q2_qkv(model) == 1
        actual = model.attention(hidden_states)

    assert getattr(hidden_states, "_gptqmodel_prism_q2_qkv_cache", None) is None
    for actual_projection, expected_projection in zip(actual, expected):
        assert actual_projection.shape == expected_projection.shape
        assert actual_projection.dtype == expected_projection.dtype
        torch.testing.assert_close(actual_projection, expected_projection, atol=2e-3, rtol=2e-3)

    sentinels = (
        torch.randn(1, 2, 2048, device="cuda", dtype=torch.float16),
        torch.randn(1, 2, 1024, device="cuda", dtype=torch.float16),
        torch.randn(1, 2, 1024, device="cuda", dtype=torch.float16),
    )
    for projection, sentinel in zip(
        (model.attention.q_proj, model.attention.k_proj, model.attention.v_proj),
        sentinels,
    ):
        projection._gptqmodel_prism_q2_qkv_original_forward = lambda _hidden_states, value=sentinel: value
    with torch.inference_mode():
        prefill = model.attention(torch.randn(1, 2, 2048, device="cuda", dtype=torch.float16))
    assert all(actual_projection is sentinel for actual_projection, sentinel in zip(prefill, sentinels))
