# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""G2PTQ configuration, block statistics, and end-to-end lifecycle tests."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.g2ptq_processor import G2PTQProcessor
from gptqmodel.models.quantization_run_plan import ProcessorStep, QuantizationRunPlan
from gptqmodel.quantization.config import (
    FORMAT,
    FOEMConfig,
    G2PTQConfig,
    GPTAQConfig,
    GPTQConfig,
    QuantizeConfig,
)
from gptqmodel.quantization.g2ptq import G2PTQ, g2ptq_compensate
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.backend import BACKEND


def _processor(qcfg=None, **attrs):
    """Build a G2PTQ processor without a tokenizer or calibration dataset."""

    from gptqmodel.quantization.config import CalibrationConfig

    config = qcfg if qcfg is not None else QuantizeConfig(
        bits=4, group_size=32, g2ptq=G2PTQConfig()
    )
    processor = G2PTQProcessor(
        tokenizer=None,
        qcfg=config,
        calibration=CalibrationConfig(),
    )
    for name, value in attrs.items():
        setattr(processor, name, value)
    return processor


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
def test_g2ptq_config_serializes_and_reloads():
    config = GPTQConfig(
        bits=4,
        group_size=32,
        g2ptq=G2PTQConfig(alpha_mse=0.25, alpha_kl=0.5, num_groups=2, kl_last_block=False),
    )

    payload = config.to_dict()

    assert payload["meta"]["g2ptq"] == {
        "alpha_mse": pytest.approx(0.25),
        "alpha_kl": pytest.approx(0.5),
        "num_groups": 2,
        "kl_last_block": False,
    }
    restored = QuantizeConfig(**json.loads(json.dumps(config.to_dict())))
    assert restored.g2ptq is not None
    assert restored.g2ptq.alpha_mse == pytest.approx(0.25)
    assert restored.g2ptq.num_groups == 2
    assert restored.g2ptq.kl_last_block is False


def test_g2ptq_config_accepts_booleans_and_dicts():
    enabled = QuantizeConfig(bits=4, g2ptq=True)
    assert enabled.g2ptq == G2PTQConfig()

    disabled = QuantizeConfig(bits=4, g2ptq=False)
    assert disabled.g2ptq is None

    from_dict = QuantizeConfig(bits=4, g2ptq={"alpha_mse": 0.5, "num_groups": 8})
    assert from_dict.g2ptq.alpha_mse == pytest.approx(0.5)
    assert from_dict.g2ptq.num_groups == 8

    # A serialized boolean must be lifted back to the object form.
    payload = json.loads(json.dumps(QuantizeConfig(bits=4, g2ptq=True).to_dict()))
    payload["meta"]["g2ptq"] = True
    assert QuantizeConfig(**payload).g2ptq == G2PTQConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha_mse": -1.0},
        {"alpha_kl": float("nan")},
        {"num_groups": 0},
        {"num_groups": 2.5},
        {"kl_last_block": "yes"},
    ],
)
def test_g2ptq_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        G2PTQConfig(**kwargs)


def test_g2ptq_rejects_other_gptq_family_solvers():
    with pytest.raises(ValueError, match="cannot be combined"):
        GPTQConfig(bits=4, g2ptq=G2PTQConfig(), gptaq=GPTAQConfig())
    with pytest.raises(ValueError, match="cannot be combined"):
        GPTQConfig(bits=4, g2ptq=G2PTQConfig(), foem=FOEMConfig())


def test_g2ptq_rejects_gsq_initializers():
    from gptqmodel.quantization.config import GSQConfig

    with pytest.raises(ValueError, match="plain GPTQ"):
        GPTQConfig(bits=4, g2ptq=G2PTQConfig(), gsq=GSQConfig(enabled=True))


def _build_plan(config, *, format_code=FORMAT.GPTQ, needs_lora=False):
    return QuantizationRunPlan.build(
        config,
        format_code=format_code,
        pack_backend=BACKEND.TORCH,
        loop_backend=BACKEND.TORCH,
        batch_size=1,
        needs_lora=needs_lora,
    )


def test_g2ptq_run_plan_selects_the_g2ptq_processor():
    plan = _build_plan(QuantizeConfig(bits=4, group_size=32, g2ptq=G2PTQConfig()))

    assert plan.processors == (ProcessorStep.G2PTQ,)
    assert plan.checkpoint_supported is True


def test_g2ptq_run_plan_requests_the_deterministic_blas_pin():
    from gptqmodel.models.base import _needs_deterministic_hessian

    plan = _build_plan(QuantizeConfig(bits=4, group_size=32, g2ptq=G2PTQConfig()))

    # The refresh accumulates fp32 Fisher Hessians, so a ROCm run must pin the
    # reproducible backend for the whole run instead of waiting on the GPTQ
    # per-module pin to flip it mid-run.
    assert _needs_deterministic_hessian(plan) is True


def test_g2ptq_run_plan_rejects_other_gptq_family_methods():
    from gptqmodel.quantization.config import QQQConfig

    # QQQConfig inherits GPTQConfig, so it is the reachable way to pair a
    # non-GPTQ method with the G2PTQ switch.
    with pytest.raises(NotImplementedError, match="does not support G2PTQ"):
        _build_plan(QQQConfig(bits=4, g2ptq=G2PTQConfig()))


# ----------------------------------------------------------------------
# block statistics
# ----------------------------------------------------------------------
def test_grouped_hessian_matches_equation_8_and_9():
    torch.manual_seed(0)
    tokens, columns, channels = 9, 5, 6
    activations = torch.randn(tokens, columns)
    saliency = torch.rand(tokens, channels)
    num_groups = 3

    hessian = G2PTQProcessor._accumulate_grouped_hessian(
        None, activations, saliency, num_groups=num_groups
    )

    per_group = channels // num_groups
    for group in range(num_groups):
        weights = saliency[:, group * per_group:(group + 1) * per_group].mean(dim=1)
        expected = torch.einsum("i,ij,ik->jk", weights, activations, activations)
        torch.testing.assert_close(hessian[group], expected)


def test_grouped_hessian_falls_back_to_one_group_for_uneven_channels():
    activations = torch.randn(4, 3)
    saliency = torch.rand(4, 5)

    hessian = G2PTQProcessor._accumulate_grouped_hessian(
        None, activations, saliency, num_groups=2
    )

    assert hessian.shape == (1, 3, 3)


def test_block_mse_objectives_match_the_paper():
    torch.manual_seed(0)
    tokens, hidden = 2048, 64
    quantized = torch.randn(tokens, hidden, requires_grad=True)
    target = torch.randn(tokens, hidden)

    loss_g, loss_h = G2PTQProcessor._block_objectives(
        quantized, target, use_kl=False
    )

    torch.testing.assert_close(
        loss_g, 0.5 * (quantized - target).square().sum() / hidden
    )
    grad_g = torch.autograd.grad(loss_g, quantized, retain_graph=True)[0]
    torch.testing.assert_close(grad_g, (quantized - target) / hidden)

    grad_h = torch.autograd.grad(loss_h, quantized)[0]
    # Theorem 1 injects eps ~ N(0, d I), so the squared output gradient is an
    # unbiased estimator of the identity with the 1/d normalization.
    assert grad_h.square().mean().item() == pytest.approx(1.0 / hidden, rel=0.05)


def test_block_kl_objectives_match_the_paper():
    torch.manual_seed(0)
    tokens, vocab = 7, 11
    quantized = torch.randn(tokens, vocab, requires_grad=True)
    target = torch.randn(tokens, vocab)

    loss_g, loss_h = G2PTQProcessor._block_objectives(quantized, target, use_kl=True)

    log_q = torch.log_softmax(quantized.float(), dim=-1)
    log_p = torch.log_softmax(target.float(), dim=-1)
    expected = torch.nn.functional.kl_div(
        log_q, log_p.exp(), reduction="batchmean"
    )
    torch.testing.assert_close(loss_g, expected.to(loss_g.dtype))

    grad_g = torch.autograd.grad(loss_g, quantized, retain_graph=True)[0]
    torch.testing.assert_close(
        grad_g, (log_q.exp() - log_p.exp()) / tokens, atol=1e-6, rtol=1e-5
    )
    assert torch.isfinite(loss_h)
    assert loss_h.item() > 0


def test_kl_objective_only_targets_the_output_head():
    processor = _processor(gptq_model=SimpleNamespace(lm_head="lm_head"))

    assert processor._uses_kl_objective(["model.layers.0.self_attn.q_proj"]) is False
    assert processor._uses_kl_objective(["lm_head"]) is True
    assert processor._uses_kl_objective(["model.lm_head"]) is True

    processor._g2ptq_is_lm_head = True
    assert processor._uses_kl_objective(["lm_head"]) is True

    processor.qcfg.g2ptq.kl_last_block = False
    assert processor._uses_kl_objective(["lm_head"]) is False


def test_clean_stream_advances_only_through_pristine_replay():
    processor = _processor()
    clean = [[torch.zeros(1, 4)]]
    processor.receive_input_cache(SimpleNamespace(layer_inputs=clean))

    noisy = [[torch.ones(1, 4)]]
    assert processor.clean_group_layer_inputs(layer_index=0, layer_inputs=noisy) is clean

    advanced = [[torch.full((1, 4), 2.0)]]
    processor.receive_clean_layer_inputs(layer_index=0, layer_inputs=advanced)
    assert processor.clean_group_layer_inputs(layer_index=1, layer_inputs=noisy) is advanced

    state = processor.continuation_state_dict()
    restored = _processor()
    restored.load_continuation_state_dict(state)
    assert restored.clean_group_layer_inputs(layer_index=1, layer_inputs=noisy) is advanced


# ----------------------------------------------------------------------
# differentiable replay
# ----------------------------------------------------------------------
def test_differentiable_block_repairs_inference_tensors():
    with torch.inference_mode():
        block = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))

    snapshot = G2PTQProcessor._differentiable_block(block, layer=None)

    assert snapshot is block
    assert not any(parameter.is_inference() for parameter in snapshot.parameters())
    assert all(parameter.requires_grad for parameter in snapshot.parameters())
    output = snapshot(torch.randn(3, 8))
    output.square().sum().backward()
    assert all(parameter.grad is not None for parameter in snapshot.parameters())


def test_differentiable_block_copies_the_live_layer():
    with torch.inference_mode():
        layer = nn.Sequential(nn.Linear(8, 4))

    snapshot = G2PTQProcessor._differentiable_block(layer, layer=layer)

    assert snapshot is not layer
    assert not any(parameter.is_inference() for parameter in snapshot.parameters())
    assert all(parameter.is_inference() for parameter in layer.parameters())


def test_differentiable_block_restores_hooked_linears():
    from gptqmodel.nn_modules.hooked_linear import HookedLinear

    block = nn.Sequential(nn.Linear(8, 8))
    hooked = HookedLinear.from_linear(block[0])
    block[0] = hooked

    snapshot = G2PTQProcessor._differentiable_block(block, layer=None)

    assert not isinstance(snapshot[0], HookedLinear)
    output = snapshot(torch.randn(2, 8))
    output.sum().backward()
    assert snapshot[0].weight.grad is not None


def test_differentiable_block_restores_a_hooked_root():
    from gptqmodel.nn_modules.hooked_linear import HookedLinear

    head = HookedLinear.from_linear(nn.Linear(8, 5))

    snapshot = G2PTQProcessor._differentiable_block(head, layer=head)

    # The output head is the projection itself: the repair has to rebuild the
    # root, because there is no child to replace.
    assert isinstance(snapshot, nn.Linear)
    assert not isinstance(snapshot, HookedLinear)
    assert not any(parameter.is_inference() for parameter in snapshot.parameters())
    snapshot(torch.randn(2, 8)).sum().backward()
    assert snapshot.weight.grad is not None
    assert head.weight.grad is None


def test_replay_kwargs_drop_keys_the_replay_target_cannot_accept():
    processor = _processor(gptq_model=SimpleNamespace())
    captured = {
        "position_embeddings": (torch.zeros(2, 2), torch.zeros(2, 2)),
        "past_key_values": object(),
        "use_cache": True,
    }

    head_kwargs = processor._replay_kwargs(
        block=nn.Linear(8, 5),
        layer_input=[torch.zeros(2, 8)],
        captured=captured,
        attention_mask=None,
        position_ids=None,
        device=torch.device("cpu"),
    )
    # A plain projection accepts none of the decoder-layer scaffolding, and a
    # KV cache is never replayed.
    assert head_kwargs == {}

    class _Block(nn.Module):
        def forward(self, hidden_states, position_embeddings=None, attention_mask=None, use_cache=False):
            del attention_mask
            return hidden_states + position_embeddings[0][:, :1, :1].sum() * 0.0 + (0.0 if not use_cache else 1.0)

    block_kwargs = processor._replay_kwargs(
        block=_Block(),
        layer_input=[torch.zeros(2, 8)],
        captured=captured,
        attention_mask=None,
        position_ids=None,
        device=torch.device("cpu"),
    )
    assert set(block_kwargs) == {"position_embeddings", "use_cache"}
    assert block_kwargs["use_cache"] is False


# ----------------------------------------------------------------------
# solver plumbing
# ----------------------------------------------------------------------
def test_quantize_without_statistics_delegates_to_gptq(monkeypatch):
    qcfg = QuantizeConfig(bits=4, group_size=-1, g2ptq=G2PTQConfig())
    task = G2PTQ(module=nn.Linear(4, 4, bias=False), qcfg=qcfg)
    calls = {}

    def fake(self, blocksize=128):
        calls["blocksize"] = blocksize
        return ("gptq",)

    monkeypatch.setattr(GPTQ, "quantize", fake)
    assert task.quantize(blocksize=64) == ("gptq",)
    assert calls["blocksize"] == 64


def test_block_statistics_reject_mismatched_shapes():
    qcfg = QuantizeConfig(bits=4, group_size=-1, g2ptq=G2PTQConfig())
    task = G2PTQ(module=nn.Linear(4, 3, bias=False), qcfg=qcfg)

    with pytest.raises(ValueError, match="not square"):
        task.set_block_statistics(
            group_hessians=torch.zeros(1, 3, 2),
            weight_gradient=torch.zeros(3, 3),
        )
    with pytest.raises(ValueError, match="gradient"):
        task.set_block_statistics(
            group_hessians=torch.zeros(1, 3, 3),
            weight_gradient=torch.zeros(4, 3),
        )


def test_solver_runs_with_installed_statistics():
    torch.manual_seed(0)
    module = nn.Linear(16, 8, bias=False)
    qcfg = QuantizeConfig(bits=4, group_size=8, g2ptq=G2PTQConfig(num_groups=2))
    task = G2PTQ(module=module, qcfg=qcfg)
    # The processor configures the grid before any capture; do the same here.
    task.quantizer.configure(perchannel=True)
    generator = torch.Generator().manual_seed(3)
    activations = torch.randn(32, 16, generator=generator)

    # Plain GPTQ capture supplies the dead-column reference and nsamples.
    task.add_batch(activations, None)
    task.set_block_statistics(
        group_hessians=torch.stack(
            [activations.t() @ activations, activations.t() @ activations]
        ),
        weight_gradient=torch.randn(8, 16, generator=generator),
        activation_square=activations.square().mean(dim=0),
        trust_alpha=0.1,
    )

    quantized, scale, zero, g_idx, duration, avg_loss, damp, nsamples = task.quantize()

    assert quantized.shape == module.weight.shape
    assert torch.isfinite(quantized.float()).all()
    assert duration >= 0
    assert nsamples == 32
    assert task.group_hessians is None and task.weight_gradient is None


def _tiny_llama_fixture(directory):
    """Write a one-layer Llama checkpoint with an untied head and calibration rows."""

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab, hidden = 256, 64
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=vocab,
            hidden_size=hidden,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).to(torch.bfloat16).eval()
    model.save_pretrained(directory)

    backend = Tokenizer(
        models.WordLevel(vocab={f"t{i}": i for i in range(vocab)}, unk_token="t0")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="t0",
        pad_token="t0",
        bos_token="t1",
        eos_token="t2",
    )
    tokenizer.save_pretrained(directory)

    generator = torch.Generator().manual_seed(7)
    rows = []
    for _ in range(4):
        ids = torch.randint(3, vocab, (16,), generator=generator)
        rows.append({"input_ids": ids.tolist(), "attention_mask": [1] * ids.numel()})
    return rows


@pytest.mark.parametrize(
    ("desc_act", "act_group_aware"),
    [(False, False), (True, False), (False, True)],
    ids=["natural", "desc_act", "act_group_aware"],
)
def test_g2ptq_end_to_end_quantizes_and_runs(desc_act, act_group_aware):
    if not torch.cuda.is_available():
        pytest.skip("G2PTQ end-to-end requires a CUDA device")

    import tempfile

    from gptqmodel import BACKEND as RUNTIME_BACKEND
    from gptqmodel import CalibrationConfig, GPTQModel

    directory = tempfile.mkdtemp(prefix="g2ptq-test-")
    rows = _tiny_llama_fixture(directory)

    solver_calls = []
    original = G2PTQ._g2ptq_quantize

    def counting(self, blocksize=128):
        solver_calls.append(blocksize)
        return original(self, blocksize=blocksize)

    G2PTQ._g2ptq_quantize = counting
    try:
        qcfg = QuantizeConfig(
            bits=4,
            group_size=16,
            device="cuda:0",
            desc_act=desc_act,
            act_group_aware=act_group_aware,
            g2ptq=G2PTQConfig(num_groups=2, alpha_mse=0.05),
        )
        qm = GPTQModel.load(directory, qcfg, dtype=torch.bfloat16)
        qm.quantize(
            CalibrationConfig(dataset=rows, batch_size=1, min_length=1),
            backend=RUNTIME_BACKEND.GPTQ_TORCH,
        )
    finally:
        G2PTQ._g2ptq_quantize = original

    assert solver_calls, "G2PTQ fell back to plain GPTQ for every module"

    qm.model.to("cuda:0")
    with torch.inference_mode():
        logits = qm.model(
            input_ids=torch.tensor([rows[0]["input_ids"]], device="cuda:0"),
            use_cache=False,
        ).logits
    assert logits.shape == (1, 16, 256)
    assert torch.isfinite(logits.float()).all()


def test_g2ptq_quantizes_the_output_head_with_the_kl_objective():
    if not torch.cuda.is_available():
        pytest.skip("G2PTQ end-to-end requires a CUDA device")

    import tempfile

    from gptqmodel import BACKEND as RUNTIME_BACKEND
    from gptqmodel import CalibrationConfig, GPTQModel

    directory = tempfile.mkdtemp(prefix="g2ptq-head-test-")
    rows = _tiny_llama_fixture(directory)

    solver_names = []
    original = G2PTQ._g2ptq_quantize

    def counting(self, blocksize=128):
        solver_names.append(getattr(self, "name", ""))
        return original(self, blocksize=blocksize)

    G2PTQ._g2ptq_quantize = counting
    try:
        qcfg = QuantizeConfig(
            bits=4,
            group_size=16,
            device="cuda:0",
            lm_head=True,
            g2ptq=G2PTQConfig(num_groups=2, alpha_mse=0.05, alpha_kl=0.1),
        )
        qm = GPTQModel.load(directory, qcfg, dtype=torch.bfloat16)
        qm.quantize(
            CalibrationConfig(dataset=rows, batch_size=1, min_length=1),
            backend=RUNTIME_BACKEND.GPTQ_TORCH,
        )
    finally:
        G2PTQ._g2ptq_quantize = original

    # The head is a hooked ``HookedLinear`` root, and its replay must build the
    # KL/cross-entropy pair instead of falling back to GPTQ.
    assert any("lm_head" in name for name in solver_names), solver_names

    qm.model.to("cuda:0")
    with torch.inference_mode():
        logits = qm.model(
            input_ids=torch.tensor([rows[0]["input_ids"]], device="cuda:0"),
            use_cache=False,
        ).logits
    assert logits.shape == (1, 16, 256)
    assert torch.isfinite(logits.float()).all()


def test_g2ptq_checkpoint_round_trips_through_save_and_reload(tmp_path):
    """Quantize, write a checkpoint, reload it, and compare inference output."""

    if not torch.cuda.is_available():
        pytest.skip("G2PTQ end-to-end requires a CUDA device")

    from gptqmodel import BACKEND as RUNTIME_BACKEND
    from gptqmodel import CalibrationConfig, GPTQModel

    source = tmp_path / "source"
    source.mkdir()
    rows = _tiny_llama_fixture(str(source))

    qcfg = QuantizeConfig(
        bits=4,
        group_size=16,
        device="cuda:0",
        lm_head=True,
        g2ptq=G2PTQConfig(num_groups=2, alpha_mse=0.05, alpha_kl=0.1),
    )
    qm = GPTQModel.load(str(source), qcfg, dtype=torch.bfloat16)
    qm.quantize(
        CalibrationConfig(dataset=rows, batch_size=1, min_length=1),
        backend=RUNTIME_BACKEND.GPTQ_TORCH,
    )

    qm.model.to("cuda:0")
    inputs = torch.tensor([rows[0]["input_ids"]], device="cuda:0")
    with torch.inference_mode():
        expected = qm.model(input_ids=inputs, use_cache=False).logits.float()

    packed = tmp_path / "packed"
    qm.save_quantized(str(packed))
    assert list(packed.glob("*.safetensors"))
    assert not list(packed.glob("*.bin"))

    restored = GPTQModel.load(
        str(packed), backend=RUNTIME_BACKEND.GPTQ_TORCH, device="cuda:0",
        dtype=torch.bfloat16,
    )
    restored.model.to("cuda:0")
    restored_g2ptq = restored.quantize_config.g2ptq
    assert restored_g2ptq is not None
    assert restored_g2ptq.alpha_mse == pytest.approx(0.05)
    assert restored_g2ptq.alpha_kl == pytest.approx(0.1)
    assert restored_g2ptq.num_groups == 2

    with torch.inference_mode():
        actual = restored.model(input_ids=inputs, use_cache=False).logits.float()

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-4)

    # The default backend may pick a different kernel than the one used for the
    # reference pass, so allow kernel-level bf16 drift but no gross corruption.
    default = GPTQModel.load(str(packed), device="cuda:0", dtype=torch.bfloat16)
    default.model.to("cuda:0")
    with torch.inference_mode():
        default_logits = default.model(input_ids=inputs, use_cache=False).logits.float()

    assert torch.allclose(default_logits, expected, rtol=0.0, atol=1e-2)


def test_g2ptq_compensate_rejects_bad_shapes():
    with pytest.raises(ValueError):
        g2ptq_compensate(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 3, 2),
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 1),
            lambda column: torch.zeros(2),
            blocksize=1,
        )
