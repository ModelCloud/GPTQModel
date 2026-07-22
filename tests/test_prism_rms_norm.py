from types import SimpleNamespace

import pytest
import torch

from gptqmodel.nn_modules.triton_utils.rms_norm import (
    _select_prism_q2_rms_norm_num_warps,
    _torch_rms_norm,
    install_prism_q2_rms_norms,
)


class _ReferenceRMSNorm(torch.nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _torch_rms_norm(self, hidden_states)


def test_prism_q2_rms_norm_torch_fallback_matches_reference():
    module = _ReferenceRMSNorm(128)
    hidden_states = torch.randn(2, 3, 128)

    actual = _torch_rms_norm(module, hidden_states)
    values = hidden_states.float()
    expected = module.weight * (values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + 1e-6))

    torch.testing.assert_close(actual, expected)


def test_prism_q2_rms_norm_launch_selection_is_limited_to_profiled_fp16_decode_shapes():
    assert _select_prism_q2_rms_norm_num_warps(dtype=torch.float16, n_cols=2048, rows=1) == 8
    assert _select_prism_q2_rms_norm_num_warps(dtype=torch.float16, n_cols=128, rows=8) == 1
    assert _select_prism_q2_rms_norm_num_warps(dtype=torch.float16, n_cols=128, rows=16) == 1
    assert _select_prism_q2_rms_norm_num_warps(dtype=torch.float16, n_cols=2048, rows=64) == 4
    assert _select_prism_q2_rms_norm_num_warps(dtype=torch.float16, n_cols=128, rows=1024) == 4
    assert _select_prism_q2_rms_norm_num_warps(dtype=torch.bfloat16, n_cols=2048, rows=1) == 4


def test_prism_q2_rms_norm_installer_preserves_cpu_fallback():
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3")
    model.norm = type("Qwen3RMSNorm", (_ReferenceRMSNorm,), {})(128)

    assert install_prism_q2_rms_norms(model) == 0
    assert not hasattr(model.norm, "_gptqmodel_prism_q2_rms_norm")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="Prism Q2 fused RMSNorm specialization requires sm80 CUDA",
)
@pytest.mark.parametrize(("rows", "width"), ((1, 2048), (8, 128), (16, 128), (3, 2048)))
def test_prism_q2_rms_norm_sm80_matches_qwen3_reference(rows: int, width: int):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen3")
    model.norm = Qwen3RMSNorm(width, eps=1e-6).cuda().half()
    hidden_states = torch.randn(rows, width, device="cuda", dtype=torch.float16)
    expected = model.norm(hidden_states)

    assert install_prism_q2_rms_norms(model) == 1
    actual = model.norm(hidden_states)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=2e-3)
