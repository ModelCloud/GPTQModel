# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""CUDA scalar relaxation compared with the existing eager GSQ equations."""

import copy

import pytest
import torch

from gptqmodel.quantization.gsq_cuda_relaxation import (
    cuda_relaxed_scalar_weights,
    cuda_rms_norm,
)
from gptqmodel.quantization.gsq_training import (
    GSQLion,
    GSQScalarTrainingModule,
    relaxed_scalar_weights,
    sampling_schedule,
    train_stage_update,
)


@pytest.mark.parametrize("bits,group_size,logits_dtype,temperature,multiplier", [
    (3, 32, torch.bfloat16, .7, 125.), (3, 48, torch.bfloat16, .7, 125.),
    (3, 128, torch.bfloat16, .7, 125.), (4, 128, torch.bfloat16, .7, 125.),
    (3, 128, torch.float32, .7, 125.), (3, 128, torch.float32, .05, 500.),
])
def test_cuda_gsq_relaxation_matches_eager_forward_and_gradients(
    bits, group_size, logits_dtype, temperature, multiplier,
):
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
    eager = quant(uniform=uniform, temperature=temperature, multiplier=multiplier)
    (eager*upstream).sum().backward()
    eager_logits = quant.logits.grad.detach().clone()
    eager_scales = quant.scales.grad.detach().clone()

    quant.zero_grad(set_to_none=True)
    fused = cuda_relaxed_scalar_weights(quant.logits, quant.scales, quant.initial,
                                        quant.valid, uniform, group_size, temperature, multiplier)
    if fused is None:
        pytest.skip("CUDA NVRTC relaxation unavailable")
    if logits_dtype == torch.float32:
        assert fused.grad_fn.saved_tensors[0].dtype == torch.float32
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


def test_cuda_fp32_qk_keeps_small_probability_gradient():
    """A nearly certain choice must retain its FP32 probability derivative."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    logits = torch.full((5, 1, 128), -1., device="cuda", requires_grad=True)
    with torch.no_grad():
        logits[1].zero_()
        logits[2].fill_(-.0007)
    scales = torch.ones((1, 1), device="cuda", requires_grad=True)
    initial = torch.zeros((1, 128), device="cuda")
    valid = torch.ones_like(logits, dtype=torch.bool)
    uniform = torch.full_like(logits, .5)
    candidates = torch.arange(-2, 3, dtype=torch.float32, device="cuda")[:, None, None].expand_as(logits)
    group_index = torch.zeros(128, dtype=torch.long, device="cuda")
    eager = relaxed_scalar_weights(
        logits, scales, candidates, group_index, uniform=uniform,
        temperature=.05, multiplier=500., initial=initial,
    )
    eager.sum().backward()
    eager_gradient = logits.grad.detach().clone()
    logits.grad = None
    scales.grad = None
    fused = cuda_relaxed_scalar_weights(logits, scales, initial, valid, uniform, 128, .05, 500.)
    if fused is None:
        pytest.skip("CUDA NVRTC relaxation unavailable")
    assert fused.grad_fn.saved_tensors[0].dtype == torch.float32
    fused.sum().backward()
    assert eager_gradient[1].abs().min() > 1.
    torch.testing.assert_close(fused, eager, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logits.grad, eager_gradient, rtol=.005, atol=.005)


@pytest.mark.parametrize("bits,logits_dtype", [
    (3, torch.bfloat16), (4, torch.bfloat16), (3, torch.float32),
])
def test_cuda_gsq_30_update_trajectory_matches_eager(bits, logits_dtype, monkeypatch):
    """Compare learning, hard choices, and held-out loss under matched draws."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    torch.manual_seed(2026 + bits)
    rows, columns, group_size = 128, 256, 64
    scales = torch.full((rows, columns // group_size), .02, device="cuda")
    original = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16) * .02
    levels = (original.float() / .02).round().clamp(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    weight = (levels * .02).bfloat16()
    noise = torch.randn(5, rows, columns, device="cuda", dtype=torch.bfloat16)
    initial = GSQScalarTrainingModule(weight, scales, group_size, bits=bits,
                                      noise=noise, logits_dtype=logits_dtype)
    eager, fused = copy.deepcopy(initial), copy.deepcopy(initial)
    inputs = torch.randn(128, columns, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(128, rows, device="cuda", dtype=torch.bfloat16) * .02

    def objective(_batch, weights):
        output = torch.nn.functional.linear(inputs, weights["weight"].bfloat16())
        return torch.nn.functional.mse_loss(output, target)

    optimizers = [GSQLion(module.optimizer_groups(
        assignment_lr=1e-4, scale_lr=5e-5, weight_decay=1.), betas=(.9, .95))
        for module in (eager, fused)]
    generators = [torch.Generator(device="cuda").manual_seed(91) for _ in range(2)]
    for step in range(30):
        temperature, multiplier = sampling_schedule(
            step, 30, temperature=(2., .05), multiplier=(100., 500.))
        losses = []
        for module, optimizer, generator, disabled in zip(
            (eager, fused), optimizers, generators, (True, False)
        ):
            if disabled:
                monkeypatch.setenv("GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION", "1")
            else:
                monkeypatch.delenv("GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION", raising=False)
            loss = train_stage_update(
                {"weight": module}, optimizer, [(None, 1)], objective,
                generator=generator, temperature=temperature, multiplier=multiplier,
            )
            losses.append(loss)
        torch.testing.assert_close(losses[0], losses[1], rtol=.01, atol=.001)

    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    eager_choices = eager.logits.masked_fill(~eager.valid, -torch.inf).argmax(0)
    fused_choices = fused.logits.masked_fill(~fused.valid, -torch.inf).argmax(0)
    assert (eager_choices != fused_choices).float().mean() < .001
    eager_hard = objective(None, {"weight": eager.hard_weight()})
    fused_hard = objective(None, {"weight": fused.hard_weight()})
    torch.testing.assert_close(eager_hard, fused_hard, rtol=.01, atol=.001)


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


def test_cuda_and_eager_staged_llama_agree_after_packed_training(tmp_path, monkeypatch):
    """Exercise the seven-projection W3 lifecycle with matched training seeds."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model
    from gptqmodel.quantization import GSQTrainingConfig

    torch.manual_seed(41)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=4, num_hidden_layers=1)
    config._attn_implementation = "sdpa"
    original = LlamaForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
    train = [{"input_ids": list(range(1, 17))}, {"input_ids": list(range(17, 33))}]
    initialization = [{"input_ids": list(range(33, 49))}]
    validation = [{"input_ids": list(range(49, 65))}]
    results = []
    for disabled, label in ((True, "eager"), (False, "cuda")):
        if disabled:
            monkeypatch.setenv("GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION", "1")
        else:
            monkeypatch.delenv("GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION", raising=False)
        model = copy.deepcopy(original)
        run = quantize_llama_gsq_model(
            model, train, initialization_documents=initialization,
            validation_documents=validation, bits=3, group_size=32,
            gsq=GSQTrainingConfig(enabled=True, epochs=3, qk_steps=3,
                                  batch_size=2, microbatch_size=1),
            offload_capture=True, capture_directory=tmp_path / label,
        )
        assert run["state"] == "complete"
        with torch.no_grad():
            logits = model(torch.tensor([list(range(1, 17))], device="cuda")).logits
        results.append((run, logits.detach().clone()))

    eager_run, eager_logits = results[0]
    fused_run, fused_logits = results[1]
    torch.testing.assert_close(fused_logits, eager_logits, rtol=.01, atol=.005)
    for stage in ("attention", "mlp"):
        eager_stage = eager_run["blocks"][0]["stages"][stage]
        fused_stage = fused_run["blocks"][0]["stages"][stage]
        for metric in ("hard_loss_after", "validation_hard_loss_after"):
            assert fused_stage[metric] == pytest.approx(eager_stage[metric], rel=.01, abs=1e-5)
