# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CUDA scalar relaxation compared with the existing eager GSQ equations."""

import pytest
import torch

from gptqmodel.quantization.gsq_cuda_relaxation import (
    cuda_relaxed_scalar_weights,
    cuda_rms_norm,
)
from gptqmodel.quantization.gsq_training import GSQScalarTrainingModule


@pytest.mark.parametrize("bits,group_size,logits_dtype", [
    (3, 32, torch.bfloat16), (3, 48, torch.bfloat16),
    (3, 128, torch.bfloat16), (4, 128, torch.bfloat16),
    (3, 128, torch.float32),
])
def test_cuda_gsq_relaxation_matches_eager_forward_and_gradients(bits, group_size, logits_dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    torch.manual_seed(102)
    rows, columns = 8, 256
    weight = (torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)*.02).clamp(-.04, .04)
    scales = torch.full((rows, (columns+group_size-1)//group_size), .02,
                        device="cuda", dtype=torch.float32)
    noise = torch.randn(5, rows, columns, device="cuda", dtype=torch.bfloat16)
    quant = GSQScalarTrainingModule(weight, scales, group_size, bits=bits, noise=noise,
                                    logits_dtype=logits_dtype)
    assert quant.initial.dtype == torch.float32
    uniform = torch.rand_like(quant.logits)
    upstream = torch.randn_like(weight)

    quant.zero_grad(set_to_none=True)
    eager = quant(uniform=uniform, temperature=.7, multiplier=125.)
    (eager*upstream).sum().backward()
    eager_logits = quant.logits.grad.detach().clone()
    eager_scales = quant.scales.grad.detach().clone()

    quant.zero_grad(set_to_none=True)
    fused = cuda_relaxed_scalar_weights(quant.logits, quant.scales, quant.initial,
                                        quant.valid, uniform, group_size, .7, 125.)
    if fused is None:
        pytest.skip("CUDA NVRTC relaxation unavailable")
    (fused*upstream).sum().backward()
    torch.testing.assert_close(fused, eager, rtol=0, atol=1e-4)
    torch.testing.assert_close(quant.logits.grad, eager_logits, rtol=.02, atol=.02)
    torch.testing.assert_close(quant.scales.grad, eager_scales, rtol=.02, atol=.05)


def test_cuda_gsq_relaxation_keeps_eager_fallback(monkeypatch):
    monkeypatch.setenv("GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION", "1")
    initial = torch.zeros(2, 32)
    logits = torch.zeros(5, 2, 32)
    scales = torch.ones(2, 1)
    valid = torch.ones_like(logits, dtype=torch.bool)
    uniform = torch.full_like(logits, .5)
    assert cuda_relaxed_scalar_weights(logits, scales, initial, valid, uniform, 32, 1., 1.) is None


def test_cuda_cached_llama_rms_norm_matches_eager_bf16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    torch.manual_seed(23)
    hidden = torch.randn(2, 64, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    actual = cuda_rms_norm(hidden, weight, 1e-6)
    if actual is None:
        pytest.skip("CUDA NVRTC RMSNorm unavailable")
    hidden_fp32 = hidden.float()
    expected = weight*(hidden_fp32*torch.rsqrt(hidden_fp32.square().mean(-1, keepdim=True)+1e-6)).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=.01, atol=.02)


def test_cuda_staged_llama_uses_fused_training_and_disk_targets(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.quantization import gsq_cuda_relaxation as fast_module

    used_fused = []
    original = fast_module.cuda_relaxed_scalar_weights

    def record_fused(*args, **kwargs):
        output = original(*args, **kwargs)
        if output is not None:
            used_fused.append(True)
        return output

    monkeypatch.setattr(fast_module, "cuda_relaxed_scalar_weights", record_fused)

    torch.manual_seed(41)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    model = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    train = [{"input_ids": list(range(1, 17))}, {"input_ids": list(range(17, 33))}]
    init = [{"input_ids": list(range(33, 49))}]
    validation = [{"input_ids": list(range(49, 65))}]
    run = quantize_llama_gsq_model(
        model, train, initialization_documents=init, validation_documents=validation,
        bits=3, group_size=32,
        gsq=GSQTrainingConfig(enabled=True, epochs=1, qk_steps=1,
                              batch_size=2, microbatch_size=1),
        offload_capture=True, capture_directory=tmp_path / "capture")
    assert run["state"] == "complete"
    assert used_fused
    assert all(isinstance(model.model.layers[0].get_submodule(name), TorchLinear) for name in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"))
    assert not list((tmp_path / "capture" / "layer-00").glob("gsq-*-*"))
