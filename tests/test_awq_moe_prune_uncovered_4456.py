# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4456: AWQ with ``fallback=None`` must prune uncovered experts.

With ``fallback=None`` the subset executor prunes modules that never received
calibration input (MoE experts no token was routed to): it removes them from the
subset, restores their dense source and records a dynamic exclusion. AWQ also
tracks every module of the layer in ``_AWQLayerState.modules`` /
``pending_modules`` and only quantizes the layer once nothing in scope is
pending. The prune path used to drop the module from ``processor.tasks`` only,
so the pruned expert stayed pending forever, ``_quantize_layer`` never ran, and
every module of the layer reached ``pack_module`` without zero points.
"""

import types

import pcre
import pytest
import torch
from torch import nn

import gptqmodel.looper.stage_subset as stage_subset_module
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_subset import CalibrationCoveragePolicy, SubsetPlan
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig


class _ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _ImmediatePool:
    def submit(self, _device, callback, *args):
        return _ImmediateFuture(callback(*args))


class _Logger:
    def error(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None

    def isEnabledFor(self, *_args, **_kwargs):
        return False


class _Looper:
    """Minimal looper: the modules are already dense, so preparation is a no-op."""

    def __init__(self, qcfg):
        self.gptq_model = types.SimpleNamespace(quantize_config=qcfg, quant_region_timer=None)

    def _prepare_named_module_for_quantization(self, *, processor, named_module, fallback_device):
        return fallback_device


def _layer(num_experts):
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.ModuleList()
    for _ in range(num_experts):
        expert = nn.Module()
        expert.gate_proj = nn.Linear(16, 8, bias=False)
        expert.up_proj = nn.Linear(16, 8, bias=False)
        layer.mlp.experts.append(expert)
    return layer


def _processor(qcfg, layer):
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=CalibrationConfig(),
        gptq_model=types.SimpleNamespace(rotary_embedding=None),
        model=layer,
        calculate_w_wq_diff=False,
    )
    processor._quant_batch_size = 0
    return processor


def _named_modules(layer):
    named = {}
    for index, expert in enumerate(layer.mlp.experts):
        for leaf in ("gate_proj", "up_proj"):
            name = f"mlp.experts.{index}.{leaf}"
            module = NamedModule(
                getattr(expert, leaf),
                name=name,
                full_name=f"model.layers.0.{name}",
                layer_index=0,
            )
            module.state["layer_module"] = layer
            named[name] = module
    return named


def _plan(named_modules):
    return SubsetPlan(
        modules=named_modules,
        subset_index=0,
        subset_total=1,
        execute_forward=False,
        replay_after_process=False,
        forward_mode="serial",
        batch_count=0,
        forward_row_counts=[],
        forward_total_rows=0,
        moe_groups={},
        forward_device_map={},
        calibration_coverage_policy=CalibrationCoveragePolicy(
            validate_input_coverage=True,
            fallback_enabled=False,
            prune_uncovered_modules=True,
            record_dynamic_exclusions=True,
        ),
        module_chunks=[named_modules],
        ordered_module_names=list(named_modules),
    )


def test_uncovered_expert_is_pruned_from_awq_layer_state(monkeypatch):
    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    layer = _layer(num_experts=3)
    processor = _processor(qcfg, layer)
    named = _named_modules(layer)
    uncovered = {"mlp.experts.1.gate_proj", "mlp.experts.1.up_proj"}

    for module in named.values():
        processor.preprocess(module, fallback=None)
    # Captured activations for every covered expert; none for expert 1.
    for name in named:
        if name not in uncovered:
            processor.tasks[name]["inputs"].append(torch.randn(1, 4, 16))

    quantized_layers = []

    def record_quantize(layer_index, state):
        quantized_layers.append((layer_index, set(state.modules), set(state.pending_modules)))
        state.quantized = True

    monkeypatch.setattr(processor, "_quantize_layer", record_quantize)
    monkeypatch.setattr(stage_subset_module, "DEVICE_THREAD_POOL", _ImmediatePool())
    monkeypatch.setattr(stage_subset_module, "torch_sync", lambda: None)
    monkeypatch.setattr(
        stage_subset_module,
        "_emit_moe_parallel_quant_subset_telemetry",
        lambda **_kwargs: None,
    )

    processed, _outputs, _used_data_parallel = stage_subset_module._run_single_subset_pass(
        looper=_Looper(qcfg),
        processor=processor,
        module=layer,
        plan=_plan(dict(named)),
        layer_inputs=[],
        layer_input_kwargs=[],
        position_ids=[],
        attention_masks=[],
        cur_layer_device=torch.device("cpu"),
        is_lm_head_module=False,
        layer_descriptor="model.layers.0",
        layer_title="Layer 0",
        layer_index=0,
        full=dict(named),
        fallback=None,
        shared_kv_cache_dict={},
        pb=None,
        logger=_Logger(),
        is_awq_processor=True,
        execute_forward=False,
    )

    covered = set(named) - uncovered
    assert set(processed) == covered
    # The layer quantizes once every covered module is processed, and the
    # pruned expert is neither tracked nor awaited.
    assert len(quantized_layers) == 1
    layer_index, modules, pending = quantized_layers[0]
    assert layer_index == 0
    assert modules == covered
    assert not (pending & uncovered)
    for name in uncovered:
        assert name not in processor.tasks
    # Same dynamic-exclusion contract as GPTQ's prune path.
    assert set(qcfg.dynamic) == {
        f"-:{pcre.escape('model.layers.0.' + name)}" for name in uncovered
    }


def test_pruned_module_does_not_block_layer_completion():
    """Pruning drops every AWQ record of the module, so the layer still completes."""

    qcfg = AWQConfig(device="cpu", group_size=16, fallback=None)
    layer = _layer(num_experts=2)
    processor = _processor(qcfg, layer)
    named = _named_modules(layer)

    for module in named.values():
        processor.preprocess(module, fallback=None)
    pruned = named["mlp.experts.0.gate_proj"]
    processor.prune_uncovered_module(pruned)

    state = processor._get_layer_state(0)
    assert pruned.name not in state.modules
    assert pruned.name not in state.pending_modules
    assert pruned.name not in processor.tasks

    # Processing every remaining module completes the layer.
    calls = []
    processor._quantize_layer = lambda layer_index, layer_state: calls.append(set(layer_state.modules))
    remaining = {name: module for name, module in named.items() if name != pruned.name}
    for module in remaining.values():
        processor.process(module, subset=remaining, subset_index=0, subset_total=1)
    assert calls and calls[-1] == set(remaining)


def _build_tiny_qwen3_moe(model_dir):
    """Save a two-layer, 16-expert Qwen3-MoE fixture with a local tokenizer."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeForCausalLM,
    )

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "experts that receive no calibration tokens stay dense",
        "a third row with different words",
        "short row",
    ]
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        texts, trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)

    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        # Every projection keeps 64 output features: AWQ clip search batches
        # output channels in multiples of 64.
        num_key_value_heads=4,
        head_dim=16,
        num_experts=16,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
    )
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(config).save_pretrained(model_dir)
    calibration = [dict(fast_tokenizer(text, return_tensors="pt")) for text in texts]
    return config, calibration


@pytest.mark.slow
def test_tiny_qwen3_moe_awq_fallback_none_keeps_uncovered_experts_dense(tmp_path):
    """The issue's repro: AWQ + ``fallback=None`` with experts that get no tokens."""

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    if not torch.cuda.is_available():
        pytest.skip("AWQ packing on the TORCH backend needs CUDA for this fixture")

    model_dir = tmp_path / "native"
    config, calibration = _build_tiny_qwen3_moe(model_dir)
    qcfg = AWQConfig(bits=4, group_size=32, fallback=None)

    model = GPTQModel.load(str(model_dir), quantize_config=qcfg, dtype=torch.float16)
    model.quantize(CalibrationConfig(dataset=calibration, batch_size=4, min_length=1))

    excluded = {
        pattern[2:].replace("\\", "")
        for pattern in (model.quantize_config.dynamic or {})
        if pattern.startswith("-:")
    }
    # 4 short rows with top-2 routing over 16 experts leave experts uncovered.
    assert excluded, "fixture must leave at least one expert without calibration tokens"
    assert all(".mlp.experts." in name for name in excluded)

    dense_experts, quantized = set(), set()
    for name, module in model.model.named_modules():
        if not name.startswith("model.layers."):
            continue
        if isinstance(module, BaseQuantLinear):
            quantized.add(name)
        elif isinstance(module, nn.Linear) and ".mlp.experts." in name:
            dense_experts.add(name)
    # Exactly the pruned experts stay dense; the rest of every layer, including
    # attention and the covered experts, is quantized.
    assert dense_experts == excluded
    expected_linears = config.num_hidden_layers * (4 + 3 * config.num_experts)
    assert len(quantized) + len(dense_experts) == expected_linears
    assert any(".self_attn.q_proj" in name for name in quantized)

    output_dir = tmp_path / "awq"
    model.save(output_dir)
    reloaded = GPTQModel.load(str(output_dir), backend=BACKEND.TORCH, device="cuda:0")
    input_ids = calibration[0]["input_ids"].to("cuda:0")
    with torch.no_grad():
        logits = reloaded.model(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()
