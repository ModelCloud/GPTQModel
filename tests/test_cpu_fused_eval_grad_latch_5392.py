# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""An eval-mode grad forward must not latch the CPU fused kernels to dense (#5392).

``TorchFusedLinear``/``TorchAtenLinear``/``TorchCpuAvx512Linear`` (and the AWQ
variants) latch ``linear_mode="train"`` when a forward is taken with
``linear_mode is None``. After one eval-mode grad forward restores the
checkpoint layout, the second grad forward took that branch and pinned the
module to the dense-dequant path for good: ``model.eval()`` returned early, so
only ``train(); eval()`` recovered, and later inference was 8-15x slower per
layer. The latch now only happens in training mode.
"""

import pytest
import torch

from tests.test_cpu_trainable_kernels_grad_after_inference_5314 import (
    BUILDERS,
    IN_FEATURES,
    OUT_FEATURES,
)


def _mode(layer):
    mode = getattr(layer, "linear_mode", None)
    return getattr(mode, "value", mode)


@pytest.mark.parametrize(
    "builder", [builder for _, builder in BUILDERS], ids=[name for name, _ in BUILDERS]
)
def test_two_eval_grad_forwards_do_not_latch_the_dense_path(builder):
    layer = builder()
    layer.eval()

    x_eval = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    with torch.inference_mode():
        layer(x_eval)
    assert _mode(layer) == "inference"

    for _ in range(2):
        x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16, requires_grad=True)
        out = layer(x)
        assert out.requires_grad, "the eval-mode grad forward detached the output"
        out.backward(torch.randn_like(out))
        assert x.grad is not None

    # The second eval-mode grad forward used to latch "train" here.
    assert _mode(layer) != "train"

    with torch.inference_mode():
        again = layer(x_eval)
    assert _mode(layer) == "inference", "the next inference forward did not re-transform"
    assert again.shape == (4, OUT_FEATURES)
    assert torch.isfinite(again).all()


@pytest.mark.parametrize(
    "builder", [builder for _, builder in BUILDERS], ids=[name for name, _ in BUILDERS]
)
def test_training_mode_still_latches_until_eval(builder):
    layer = builder()
    layer.train()

    x = torch.randn(4, IN_FEATURES, dtype=torch.bfloat16)
    with torch.no_grad():
        layer(x)
    assert _mode(layer) == "train"

    # ``eval()`` clears the training latch so the next forward can transform.
    layer.eval()
    assert _mode(layer) is None


def test_gguf_cpp_grad_forward_uses_the_exact_path():
    """The GGUF CPU sibling: the native ggml matmul must not drop the gradient."""

    from tests import test_gguf_cpp_matmul_accuracy_4316 as gguf_fixture

    gguf_fixture._require_cpu_bridge()
    cpp, exact, _ = gguf_fixture._build_pair()

    x = torch.randn(4, gguf_fixture.IN_FEATURES, requires_grad=True)
    out = cpp(x)
    assert out.requires_grad, "the native ggml matmul detached the grad-mode output"
    out.float().sum().backward()
    assert x.grad is not None

    # A grad-mode call takes the exact torch path, which matches GGUFTorchLinear
    # on the same bytes.
    with torch.no_grad():
        torch.testing.assert_close(out.detach().float(), exact(x).float(), rtol=2e-3, atol=2e-3)
