from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel
from gptqmodel.nn_modules.triton_utils.q2_swiglu import install_prism_q2_swiglu


class Qwen3MLP(torch.nn.Module):
    def __init__(
        self,
        gate_proj: torch.nn.Module,
        up_proj: torch.nn.Module,
        down_proj: torch.nn.Module,
    ):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _model_with_mlp(mlp: torch.nn.Module) -> torch.nn.Module:
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3", hidden_act="silu")
    model.mlp = mlp
    return model.eval()


def _random_q2_projection(seed: int) -> GGUFTritonKernel:
    generator = torch.Generator().manual_seed(seed)
    projection = GGUFTritonKernel(
        bits="q2_0",
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=2048,
        out_features=6144,
        bias=False,
        register_buffers=True,
    )
    blocks = projection.qweight.reshape(6144, 16, 34)
    scales = torch.full((6144, 16), 0.01, dtype=torch.float16)
    with torch.no_grad():
        blocks[..., :2].copy_(scales.view(torch.uint8).reshape(6144, 16, 2))
        blocks[..., 2:].copy_(torch.randint(0, 256, blocks[..., 2:].shape, dtype=torch.uint8, generator=generator))
    return projection.cuda().eval()


def test_prism_q2_swiglu_installer_preserves_non_q2_fallback():
    mlp = Qwen3MLP(
        torch.nn.Linear(4, 8, bias=False),
        torch.nn.Linear(4, 8, bias=False),
        torch.nn.Linear(8, 4, bias=False),
    )
    model = _model_with_mlp(mlp)

    assert install_prism_q2_swiglu(model) == 0
    assert not hasattr(model.mlp, "_gptqmodel_prism_q2_swiglu")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="Prism Q2 fused SwiGLU specialization requires sm80 CUDA",
)
def test_prism_q2_swiglu_sm80_matches_unfused_decode_and_preserves_prefill_fallback():
    pytest.importorskip("triton")
    torch.manual_seed(1234)
    mlp = Qwen3MLP(
        _random_q2_projection(1),
        _random_q2_projection(2),
        torch.nn.Identity(),
    )
    model = _model_with_mlp(mlp)
    hidden_states = torch.randn(1, 1, 2048, device="cuda", dtype=torch.float16) * 0.1

    with torch.inference_mode():
        expected = model.mlp(hidden_states)
        assert install_prism_q2_swiglu(model) == 1
        actual = model.mlp(hidden_states)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    replay_input = torch.randn_like(hidden_states)
    with torch.inference_mode():
        replay_expected = model.mlp._gptqmodel_prism_q2_swiglu_original_forward(replay_input)
        model.mlp(hidden_states)
        torch.cuda.synchronize()
        graph_input = hidden_states.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = model.mlp(graph_input)
        graph_input.copy_(replay_input)
        graph.replay()
        torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, replay_expected, atol=2e-3, rtol=2e-3)

    sentinel = torch.randn(1, 2, 6144, device="cuda", dtype=torch.float16)
    model.mlp._gptqmodel_prism_q2_swiglu_original_forward = lambda _hidden_states: sentinel
    with torch.inference_mode():
        assert model.mlp(torch.randn(1, 2, 2048, device="cuda", dtype=torch.float16)) is sentinel
