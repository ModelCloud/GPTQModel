# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""G2PTQ configuration, block statistics, and end-to-end lifecycle tests."""

import json
from copy import deepcopy
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
    QuantizeEmbed,
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



def test_g2ptq_dynamic_disable_and_solver_conflicts():
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = GPTQConfig(
        g2ptq=True,
        dynamic={r"+:^plain$": {"g2ptq": False}, r"+:^conflict$": {"foem": FOEMConfig()}},
    )
    assert clone_gptq_config_for_module(config, "plain").g2ptq is None
    assert clone_gptq_config_for_module(config, "other").g2ptq is not None
    with pytest.raises(ValueError, match="cannot be combined"):
        clone_gptq_config_for_module(config, "conflict")

    no_run_setting = GPTQConfig(dynamic={r"+:^proj$": {"g2ptq": True}})
    with pytest.raises(ValueError, match="requires run-level"):
        clone_gptq_config_for_module(no_run_setting, "proj")


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
        log_q, log_p.exp(), reduction="sum"
    )
    torch.testing.assert_close(loss_g, expected.to(loss_g.dtype))

    grad_g = torch.autograd.grad(loss_g, quantized, retain_graph=True)[0]
    torch.testing.assert_close(
        grad_g, log_q.exp() - log_p.exp(), atol=1e-6, rtol=1e-5
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
    restored._g2ptq_seed = processor._g2ptq_seed + 1
    restored.load_continuation_state_dict(state)
    assert restored._g2ptq_seed == processor._g2ptq_seed
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
    assert set(block_kwargs) == {"position_embeddings", "attention_mask", "use_cache"}
    assert block_kwargs["attention_mask"] is None
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


def test_g2ptq_requantizes_both_embedding_endpoints(tmp_path):
    """Requantize the token-id input embedding and the linear output head.

    The grouped pristine replay used to forward the embedding endpoint with
    the decoder kwargs (``position_ids``), which ``AutoNF12Embedding`` rejects.
    The input embedding consumes token ids and has no float activation, so
    G2PTQ leaves it to GPTQ; the linear output head still uses the KL refresh.
    """

    if not torch.cuda.is_available():
        pytest.skip("G2PTQ end-to-end requires a CUDA device")

    from gptqmodel import BACKEND as RUNTIME_BACKEND
    from gptqmodel import CalibrationConfig, GPTQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    source = tmp_path / "source"
    source.mkdir()
    rows = _tiny_llama_fixture(str(source))

    qcfg = QuantizeConfig(
        bits=4,
        group_size=16,
        device="cuda:0",
        g2ptq=G2PTQConfig(num_groups=2, alpha_mse=0.05, alpha_kl=0.1),
    )
    qm = GPTQModel.load(str(source), qcfg, dtype=torch.bfloat16)
    calibration = CalibrationConfig(dataset=rows, batch_size=1, min_length=1)
    qm.quantize(calibration, backend=RUNTIME_BACKEND.GPTQ_TORCH)

    solver_names = []
    original = G2PTQ._g2ptq_quantize

    def counting(self, blocksize=128):
        solver_names.append(getattr(self, "name", ""))
        return original(self, blocksize=blocksize)

    G2PTQ._g2ptq_quantize = counting
    try:
        qm.requantize(
            calibration,
            embed_quant_mode=QuantizeEmbed.BOTH,
            backend=RUNTIME_BACKEND.GPTQ_TORCH,
        )
    finally:
        G2PTQ._g2ptq_quantize = original

    assert isinstance(qm.get_input_embeddings(), BaseQuantLinear)
    assert isinstance(qm.get_output_embeddings(), BaseQuantLinear)
    # The output head is a float block and keeps the G2PTQ KL refresh; the
    # input embedding consumes token ids and falls back to GPTQ.
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


@pytest.mark.parametrize(
    "device", [
        "cpu",
        pytest.param("cuda:0", marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="Accelerator required"
        )),
    ],
)
def test_fp16_statistics_widen_before_reductions_and_squaring(device):
    # The scalar loss exceeds FP16 capacity before division by the hidden size.
    predicted = torch.full((256, 8), 8, dtype=torch.float16, device=device, requires_grad=True)
    target = torch.zeros_like(predicted)
    loss_g, loss_h = G2PTQProcessor._block_objectives(predicted, target, use_kl=False)
    torch.testing.assert_close(loss_g.float(), torch.tensor(8192.0, device=device))
    assert torch.isfinite(loss_h)

    # These gradients are representable in FP16 but their squares are not.
    activation = torch.ones((2, 1), dtype=torch.float16, device=device)
    gradient = torch.full((2, 2), 2**-13, dtype=torch.float16, device=device)
    rows, saliency = G2PTQProcessor._module_rows(activation, gradient)
    hessian = G2PTQProcessor._accumulate_grouped_hessian(
        None, rows, saliency, num_groups=2
    )
    torch.testing.assert_close(
        hessian, torch.full((2, 1, 1), 2**-25, device=device), atol=0, rtol=0
    )


@pytest.mark.parametrize("static_groups", [False, True])
def test_grouped_fisher_does_not_erase_a_live_input_channel(static_groups):
    module = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1, 7, 1, 1], [1, 7, 1, 1]]))
    config = GPTQConfig(
        bits=4, group_size=4, static_groups=static_groups, act_group_aware=False,
        g2ptq=G2PTQConfig(num_groups=2, alpha_mse=0),
    )
    task = G2PTQ(module, qcfg=config)
    task.quantizer.configure(perchannel=True)
    task.add_batch(torch.eye(4), None)
    task.set_block_statistics(
        group_hessians=torch.stack([torch.diag(torch.tensor([1.0, 0, 1, 1])), torch.eye(4)]),
        weight_gradient=torch.zeros_like(module.weight),
    )
    weight, scale, zero, *_ = task.quantize()
    codes = (weight / scale + zero).round().to(torch.int)
    # With 4-bit symmetric quantization and max |w| = 7, zero point is 8,
    # scale is 1, and the live outlier is exactly representable at code 15.
    torch.testing.assert_close(codes, torch.tensor([[9, 15, 9, 9], [9, 15, 9, 9]], dtype=torch.int))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Accelerator required")
@pytest.mark.parametrize("alpha", [0.0, 0.01])
def test_grouped_inverse_oom_completes_the_solve_on_cpu(monkeypatch, alpha):
    config = GPTQConfig(bits=4, group_size=4, g2ptq=G2PTQConfig(num_groups=2, alpha_mse=alpha))
    weights = torch.tensor([[1.0, 2.0, -3.0, 4.0], [-2.0, 4.0, 1.0, -1.0]])

    def solve(device, fail_inverse=False):
        module = nn.Linear(4, 2, bias=False, device=device)
        with torch.no_grad():
            module.weight.copy_(weights)
        task = G2PTQ(module, qcfg=config)
        task.quantizer.configure(perchannel=True)
        task.add_batch(torch.eye(4, device=device), None)
        task.set_block_statistics(
            group_hessians=torch.eye(4, device=device).repeat(2, 1, 1),
            weight_gradient=torch.full_like(module.weight, 0.1),
            activation_square=torch.tensor([2.0, 3.0, 4.0, 1.0], device=device),
        )
        if fail_inverse:
            inverse = task.hessian_inverse

            def fail_on_accelerator(hessian):
                if hessian.device.type == "cuda":
                    raise torch.OutOfMemoryError("forced allocation failure")
                return inverse(hessian)

            monkeypatch.setattr(task, "hessian_inverse", fail_on_accelerator)
        return [tensor.cpu() for tensor in task.quantize()[:4]]

    for actual, expected in zip(solve("cuda:0", fail_inverse=True), solve("cpu")):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_refresh_uses_the_resolved_module_configuration():
    from gptqmodel.looper.gptq_processor import clone_gptq_config_for_module

    config = GPTQConfig(
        bits=4, group_size=4, g2ptq=G2PTQConfig(num_groups=4),
        dynamic={r"+:^proj$": {"g2ptq": {"num_groups": 2, "alpha_mse": 0.25}}},
    )
    processor = _processor(config)
    block = nn.Sequential()
    block.add_module("proj", nn.Linear(4, 4, bias=False))
    task = G2PTQ(block.proj, qcfg=clone_gptq_config_for_module(config, "proj"))
    processor.tasks["proj"] = task
    activations = torch.eye(4).unsqueeze(0)
    processor._g2ptq_layer_state[0] = {
        "layer": block,
        "inputs": [[activations]],
        "targets": [[torch.zeros_like(activations)]],
    }
    processor._refresh_block(layer_index=0, names=["proj"])

    assert task.group_hessians.shape == (2, 4, 4)
    assert task._g2ptq_trust_alpha == 0.25



def test_refresh_excludes_calibration_padding_from_statistics():
    processor = _processor(GPTQConfig(g2ptq=G2PTQConfig(num_groups=1)))
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    task = G2PTQ(layer, qcfg=processor.qcfg)
    inputs = torch.tensor([[[1.0, 2.0], [9.0, 11.0], [3.0, 4.0]]])
    mask = torch.tensor([[True, False, True]])
    processor.inputs_cache.attention_masks = [torch.ones_like(mask)]
    state = {
        "layer": layer,
        "inputs": [[inputs]],
        "targets": [[torch.zeros_like(inputs)]],
        "attention_masks": [mask],
    }

    stats = processor._collect_block_statistics(
        state, None, {"proj": task}, use_kl=False
    )["proj"]
    torch.testing.assert_close(stats["gradient"], torch.tensor([[5.0, 7.0], [7.0, 10.0]]))
    torch.testing.assert_close(stats["activation_square"], torch.tensor([10.0, 20.0]))


def test_pristine_head_replay_applies_the_same_preprocessing_as_live_replay(monkeypatch):
    from gptqmodel.looper.stage_layer import _capture_pristine_group_context

    clean = [[torch.tensor([[[1.0, 2.0]]])]]
    noisy = [[torch.tensor([[[2.0, 3.0]]])]]
    layer = nn.Identity()
    processor = _processor()
    processor.receive_input_cache(SimpleNamespace(layer_inputs=clean))
    received = []

    def prepare_head_inputs(batches):
        for batch in batches:
            batch[0] = batch[0] * 2 + 1
        return batches

    def replay_outputs(_looper, *, layer_inputs, **kwargs):
        received.extend(layer_inputs)
        return layer_inputs

    monkeypatch.setattr("gptqmodel.looper.stage_layer._replay_layer_outputs", replay_outputs)
    looper = SimpleNamespace(
        gptq_model=SimpleNamespace(lm_head_pre_quantize_generate_hook=prepare_head_inputs)
    )
    _capture_pristine_group_context(
        looper, processor=processor, module=layer, pristine_module=None,
        subset_plans=[object()], layer_inputs=prepare_head_inputs(noisy),
        layer_input_kwargs=[{}], position_ids=[], attention_masks=[],
        cur_layer_device=torch.device("cpu"), is_lm_head_module=True,
        shared_kv_cache_dict={}, layer_index=1, layer_descriptor="lm_head",
        full={}, log=None, region_timer=None,
    )
    torch.testing.assert_close(received[0][0], torch.tensor([[[3.0, 5.0]]]))
    torch.testing.assert_close(clean[0][0], torch.tensor([[[1.0, 2.0]]]))


def test_replayed_projection_accumulates_every_invocation():
    class ReusedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 1, bias=False)

        def forward(self, x):
            return self.proj(x) + self.proj(x * 2)

    seed = 19
    torch.manual_seed(seed)
    processor = _processor(GPTQConfig(g2ptq=G2PTQConfig(num_groups=1)))
    block = ReusedProjection()
    task = G2PTQ(block.proj, qcfg=processor.qcfg)
    x = torch.tensor([[[1.0], [3.0]]])
    state = {"layer": block, "inputs": [[x]], "targets": [[torch.zeros_like(x)]]}

    stats = processor._collect_block_statistics(
        state, None, {"proj": task}, use_kl=False
    )["proj"]
    # Avoid depending on float32 and float64 RNGs having the same sequence.
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed)).double()
    hessian = (5 * noise.square() * x.double().square()).sum().reshape(1, 1, 1)
    torch.testing.assert_close(stats["hessian"].double(), hessian, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(stats["activation_square"], torch.tensor([50.0]))


@pytest.mark.parametrize("blocksize", [3, 6, 8])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Accelerator required")),
])
def test_dynamic_grid_matches_columnwise_refit(blocksize, device):
    generator = torch.Generator().manual_seed(47)
    module = nn.Linear(12, 4, bias=False, device=device)
    with torch.no_grad():
        module.weight.copy_(torch.randn(4, 12, generator=generator))
    x = torch.randn(32, 12, generator=generator).to(device)
    gradient = torch.randn(4, 12, generator=generator).to(device)
    config = GPTQConfig(
        bits=2, group_size=4, act_group_aware=False,
        g2ptq=G2PTQConfig(num_groups=1, alpha_mse=0.1),
    )

    def solve(size):
        task = G2PTQ(module, qcfg=config)
        task.quantizer.configure(perchannel=True)
        task.add_batch(x, None)
        task.set_block_statistics(group_hessians=(x.T @ x).unsqueeze(0), weight_gradient=gradient)
        return task.quantize(blocksize=size)[:4]

    expected_weight, expected_scale, expected_zero, expected_groups = solve(1)
    weight, scale, zero, groups = solve(blocksize)
    torch.testing.assert_close(groups, expected_groups)
    torch.testing.assert_close(zero, expected_zero, atol=0, rtol=0)
    torch.testing.assert_close(scale, expected_scale, atol=1e-6, rtol=1e-6)
    expected_codes = (expected_weight / expected_scale[:, expected_groups.long()]).round()
    actual_codes = (weight / scale[:, groups.long()]).round()
    torch.testing.assert_close(actual_codes, expected_codes, atol=0, rtol=0)


@pytest.mark.parametrize("integer_parameter", [False, True])
def test_refresh_preserves_shared_weights_and_index_parameters(integer_parameter):
    class SharedProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(2, 2, bias=False)
            self.right = nn.Linear(2, 2, bias=False)
            self.right.weight = self.left.weight
            indices = torch.tensor([1, 0])
            if integer_parameter:
                self.indices = nn.Parameter(indices, requires_grad=False)
            else:
                self.register_buffer("indices", indices)

        def forward(self, x):
            return self.left(x) + self.right(x.index_select(-1, self.indices))

    with torch.inference_mode():
        block = SharedProjection()
        block.left.weight.copy_(torch.eye(2))
        pristine = deepcopy(block)
    x = torch.tensor([[[1.0, 2.0]]])
    processor = _processor(GPTQConfig(g2ptq=G2PTQConfig(num_groups=1)))
    task = G2PTQ(block.left, qcfg=processor.qcfg)
    processor.tasks["left"] = task
    processor.receive_pristine_layer_module(layer_index=0, layer_module=pristine)
    processor._g2ptq_layer_state[0] = {
        "layer": block, "inputs": [[x]], "targets": [[torch.zeros_like(x)]],
    }
    processor._refresh_block(layer_index=0, names=["left"])
    # Both uses contribute to the gradient of the same parameter.
    assert task.weight_gradient is not None
    torch.testing.assert_close(task.weight_gradient, torch.full((2, 2), 4.5))
    assert block.left.weight is block.right.weight and block.left.weight.is_inference()
