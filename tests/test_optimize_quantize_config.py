# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Configuration coverage for the production quantization CLI."""

import importlib
import os
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    LengthAwareMode,
    QuantizeEmbed,
    VramStrategy,
)
from optimize._common import build_quantize_config, parse_gpus


def _args(*, group_size: int, hessian_length_aware: bool) -> Namespace:
    return Namespace(
        bits=4,
        group_size=group_size,
        desc_act=False,
        act_group_aware=True,
        scale_search="activation",
        moe_routing_bypass=True,
        moe_batch_size=None,
        vram_strategy=None,
        dense_vram_strategy=VramStrategy.BALANCED.value,
        moe_vram_strategy=VramStrategy.BALANCED.value,
        calibration_data_device=None,
        adaptive_damping=False,
        adaptive_clipping=False,
        hessian_length_aware=hessian_length_aware,
    )


@pytest.mark.parametrize("group_size", [64, 128])
def test_build_quantize_config_matches_maca_moe_reproducer(group_size):
    qcfg = build_quantize_config(_args(group_size=group_size, hessian_length_aware=True))

    assert qcfg.bits == 4
    assert qcfg.group_size == group_size
    assert qcfg.desc_act is False
    assert qcfg.act_group_aware is True
    assert qcfg.scale_search == "activation"
    assert isinstance(qcfg.moe.routing, ExpertsRoutingBypass)
    assert qcfg.moe.execution.batch_size is None
    assert qcfg.moe.execution.parallel_input_capture is True
    assert qcfg.dense_vram_strategy is VramStrategy.BALANCED
    assert qcfg.moe_vram_strategy is VramStrategy.BALANCED
    assert qcfg.adaptive_damping.enabled is False
    assert qcfg.adaptive_clipping.enabled is False
    assert qcfg.hessian.length_aware.mode is LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT
    assert qcfg.hessian.length_aware.target_bucket_count == 6
    assert qcfg.hessian.length_aware.bucket_weight_exponent == 0.2


def test_hessian_length_aware_flag_changes_cli_configuration():
    disabled = build_quantize_config(_args(group_size=64, hessian_length_aware=False))
    enabled = build_quantize_config(_args(group_size=64, hessian_length_aware=True))

    assert disabled.hessian.length_aware.mode is LengthAwareMode.DISABLED
    assert enabled.hessian.length_aware.mode is LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT
    assert disabled.hessian.to_dict() != enabled.hessian.to_dict()


class _FakeQuantizeConfig:
    def __init__(self):
        self.dynamic = None

    def dynamic_get(self, layer_name, default=None):
        if self.dynamic is None:
            return default
        return self.dynamic.get(layer_name, default)


class _FakeRequantModel:
    def __init__(self, *, input_name="transformer.word_embeddings", output_name="output_projection", lm_head="lm_head"):
        self.input_name = input_name
        self.output_name = output_name
        self.lm_head = lm_head
        self.quantize_config = _FakeQuantizeConfig()
        self.requantize_kwargs = None
        self.saved_to = None

    def get_input_embeddings_name(self):
        return self.input_name

    def get_output_embeddings_name(self):
        return self.output_name

    def requantize(self, **kwargs):
        self.requantize_kwargs = kwargs

    def save(self, output):
        self.saved_to = output


def _requant_args(tmp_path, mode):
    return SimpleNamespace(
        model_path="quantized-model",
        output=str(tmp_path / mode),
        trust_remote_code=False,
        bits=4,
        group_size=64,
        no_act_group_aware=False,
        desc_act=False,
        scale_search="activation",
        calibration_parquet=None,
        dataset_path="dataset",
        dataset_name=None,
        dataset_split="train",
        dataset_size=1,
        calibration_concat_size=128,
        calibration_concat_separator="\n",
        calibration_sort="desc",
        batch_size=1,
        embed_quant_mode=mode,
    )


def _patch_requant_runtime(monkeypatch, requant_module, model):
    import gptqmodel
    import torch

    monkeypatch.setattr(gptqmodel.GPTQModel, "load", lambda *args, **kwargs: model)
    monkeypatch.setattr(requant_module, "load_calibration_data", lambda **kwargs: ["calibration"])
    monkeypatch.setattr(requant_module, "set_torch_threads", lambda: None)
    monkeypatch.setattr(requant_module.faulthandler, "enable", lambda **kwargs: None)
    monkeypatch.setattr(requant_module.faulthandler, "register", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_parse_gpus_accepts_distinct_nonnegative_physical_ids():
    assert parse_gpus(" 0, 1 ") == [0, 1]


@pytest.mark.parametrize(
    "gpu_arg",
    [None, "", " ", ",", "0,0", "-1", "gpu0", "0,", ",0", "0,,1"],
)
def test_parse_gpus_rejects_empty_duplicate_or_invalid_ids(gpu_arg):
    with pytest.raises(ValueError):
        parse_gpus(gpu_arg)


def test_requant_cli_import_defers_cuda_dependencies_until_after_gpu_selection(monkeypatch):
    """Importing the CLI must not initialize Torch before CUDA visibility is configured."""

    module_name = "optimize.requant_embed_lm_head"
    sys.modules.pop(module_name, None)
    try:
        with monkeypatch.context() as patcher:
            patcher.setitem(sys.modules, "torch", None)
            patcher.setitem(sys.modules, "gptqmodel", None)
            imported = importlib.import_module(module_name)
        assert "torch" not in imported.__dict__
        assert "GPTQModel" not in imported.__dict__
    finally:
        sys.modules.pop(module_name, None)


def test_requant_main_sets_cuda_visibility_before_runtime_initialization(monkeypatch):
    import optimize.requant_embed_lm_head as requant_module

    args = SimpleNamespace(gpus="0,1", idle_timeout=3.0)
    gpu_infos = {0: {"uuid": "first"}, 1: {"uuid": "second"}}
    events = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(requant_module, "_parse_args", lambda: args)
    monkeypatch.setattr(requant_module, "idle_gate", lambda physical_gpus, timeout: gpu_infos)

    def verify(physical_gpus, verified_infos):
        events.append(("verify", physical_gpus, verified_infos, os.environ.get("CUDA_VISIBLE_DEVICES")))

    def run_requant(received_args):
        events.append(("requant", received_args, os.environ.get("CUDA_VISIBLE_DEVICES")))

    monkeypatch.setattr(requant_module, "verify_visible_uuids", verify)
    monkeypatch.setattr(requant_module, "requant_embed_lm_head", run_requant)

    requant_module._main()

    assert events == [
        ("verify", [0, 1], gpu_infos, "0,1"),
        ("requant", args, "0,1"),
    ]


@pytest.mark.parametrize(
    ("mode", "model_kwargs", "expected_targets"),
    (
        pytest.param("input", {}, {"transformer.word_embeddings"}, id="input-only"),
        pytest.param("output", {}, {"output_projection"}, id="output-only"),
        pytest.param(
            "output",
            {"output_name": None, "lm_head": "fallback_output"},
            {"fallback_output"},
            id="output-fallback",
        ),
        pytest.param(
            "both",
            {},
            {"transformer.word_embeddings", "output_projection"},
            id="both",
        ),
        pytest.param(
            "both",
            {"output_name": "transformer.word_embeddings"},
            {"transformer.word_embeddings"},
            id="tied-input-output",
        ),
    ),
)
def test_requant_cli_updates_only_selected_runtime_embedding_paths(
    monkeypatch,
    tmp_path,
    mode,
    model_kwargs,
    expected_targets,
):
    """Single-endpoint requantization must not relabel untouched packed tensors."""

    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel(**model_kwargs)
    _patch_requant_runtime(monkeypatch, requant_module, model)
    args = _requant_args(tmp_path, mode)

    output = requant_module.requant_embed_lm_head(args)

    assert set(model.quantize_config.dynamic) == expected_targets
    assert model.requantize_kwargs["embed_quant_mode"] is QuantizeEmbed(mode)
    assert model.saved_to == args.output
    assert output == tmp_path / mode


def test_requant_cli_preserves_existing_dynamic_overrides(monkeypatch, tmp_path):
    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel()
    existing_override = {"bits": 8, "group_size": 128}
    model.quantize_config.dynamic = {"transformer.layers.0.proj": existing_override}
    _patch_requant_runtime(monkeypatch, requant_module, model)

    requant_module.requant_embed_lm_head(_requant_args(tmp_path, "input"))

    assert model.quantize_config.dynamic["transformer.layers.0.proj"] is existing_override
    assert "transformer.word_embeddings" in model.quantize_config.dynamic
    assert "output_projection" not in model.quantize_config.dynamic


@pytest.mark.parametrize(
    ("mode", "model_kwargs", "message"),
    (
        pytest.param("input", {"input_name": None}, "input-embedding", id="missing-input"),
        pytest.param(
            "output",
            {"output_name": None, "lm_head": None},
            "output-embedding",
            id="missing-output",
        ),
    ),
)
def test_requant_cli_rejects_unresolved_embedding_paths(monkeypatch, tmp_path, mode, model_kwargs, message):
    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel(**model_kwargs)
    _patch_requant_runtime(monkeypatch, requant_module, model)

    with pytest.raises(ValueError, match=message):
        requant_module.requant_embed_lm_head(_requant_args(tmp_path, mode))

    assert model.requantize_kwargs is None
    assert model.saved_to is None
