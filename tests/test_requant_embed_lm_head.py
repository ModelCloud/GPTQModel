# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the input/output embedding requantization command."""

import importlib
import os
import sys
from types import SimpleNamespace

import pytest

from gptqmodel.quantization.config import GPTQConfig, QuantizeEmbed
from optimize._common import parse_gpus


class _FakeQuantizeConfig:
    def __init__(self):
        self.dynamic = None

    def dynamic_get(self, layer_name, default=None):
        if self.dynamic is None:
            return default
        return self.dynamic.get(layer_name, default)


class _FakeRequantModel:
    def __init__(
        self,
        *,
        input_name="transformer.word_embeddings",
        output_name="output_projection",
        lm_head="lm_head",
    ):
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
    import torch

    import gptqmodel

    monkeypatch.setattr(gptqmodel.GPTQModel, "load", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        requant_module, "load_calibration_data", lambda **kwargs: ["calibration"]
    )
    monkeypatch.setattr(requant_module, "set_torch_threads", lambda: None)
    monkeypatch.setattr(requant_module.faulthandler, "enable", lambda **kwargs: None)
    monkeypatch.setattr(
        requant_module.faulthandler, "register", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_parse_gpus_accepts_distinct_nonnegative_physical_ids():
    assert parse_gpus(" 0, 1 ") == [0, 1]


@pytest.mark.parametrize(
    "gpu_arg", [None, "", " ", ",", "0,0", "-1", "gpu0", "0,", ",0", "0,,1"]
)
def test_parse_gpus_rejects_empty_duplicate_or_invalid_ids(gpu_arg):
    with pytest.raises(ValueError):
        parse_gpus(gpu_arg)


def test_import_defers_cuda_dependencies_until_after_gpu_selection(monkeypatch):
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


def test_main_sets_cuda_visibility_before_runtime_initialization(monkeypatch):
    import optimize.requant_embed_lm_head as requant_module

    args = SimpleNamespace(gpus="0,1", idle_timeout=3.0)
    gpu_infos = {0: {"uuid": "first"}, 1: {"uuid": "second"}}
    events = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(requant_module, "_parse_args", lambda: args)
    monkeypatch.setattr(
        requant_module, "idle_gate", lambda physical_gpus, timeout: gpu_infos
    )

    def verify(physical_gpus, verified_infos):
        events.append(
            (
                "verify",
                physical_gpus,
                verified_infos,
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        )

    def run_requant(received_args):
        events.append(
            ("requant", received_args, os.environ.get("CUDA_VISIBLE_DEVICES"))
        )

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
            "both", {}, {"transformer.word_embeddings", "output_projection"}, id="both"
        ),
        pytest.param(
            "both",
            {"output_name": "transformer.word_embeddings"},
            {"transformer.word_embeddings"},
            id="same-runtime-path",
        ),
    ),
)
def test_updates_only_selected_runtime_embedding_paths(
    monkeypatch, tmp_path, mode, model_kwargs, expected_targets
):
    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel(**model_kwargs)
    _patch_requant_runtime(monkeypatch, requant_module, model)
    args = _requant_args(tmp_path, mode)

    output = requant_module.requant_embed_lm_head(args)

    assert set(model.quantize_config.dynamic) == expected_targets
    assert model.requantize_kwargs["embed_quant_mode"] is QuantizeEmbed(mode)
    calibration = model.requantize_kwargs["calibration"]
    assert (calibration.dataset, calibration.concat_size, calibration.concat_separator, calibration.sort) == (
        ["calibration"], 128, "\n", "desc",
    )
    assert model.saved_to == args.output
    assert output == tmp_path / mode


def test_preserves_existing_dynamic_overrides(monkeypatch, tmp_path):
    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel()
    existing_override = {"bits": 8, "group_size": 128}
    model.quantize_config.dynamic = {"transformer.layers.0.proj": existing_override}
    original_dynamic = model.quantize_config.dynamic
    _patch_requant_runtime(monkeypatch, requant_module, model)

    requant_module.requant_embed_lm_head(_requant_args(tmp_path, "input"))

    assert (
        model.quantize_config.dynamic["transformer.layers.0.proj"] is existing_override
    )
    assert model.quantize_config.dynamic is not original_dynamic
    assert "transformer.word_embeddings" in model.quantize_config.dynamic
    assert "output_projection" not in model.quantize_config.dynamic


def test_selected_target_precedes_existing_broad_dynamic_pattern(monkeypatch, tmp_path):
    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel()
    model.quantize_config.dynamic = {".*": {"bits": 8, "group_size": 128}}
    _patch_requant_runtime(monkeypatch, requant_module, model)

    requant_module.requant_embed_lm_head(_requant_args(tmp_path, "input"))

    dynamic_items = list(model.quantize_config.dynamic.items())
    assert dynamic_items[0][0] == "transformer.word_embeddings"
    assert dynamic_items[0][1]["bits"] == 4
    assert dynamic_items[1] == (".*", {"bits": 8, "group_size": 128})


def test_replaces_dynamic_mapping_to_invalidate_cached_lookup(monkeypatch, tmp_path):
    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel()
    model.quantize_config = GPTQConfig(
        bits=4,
        group_size=128,
        dynamic={".*": {"bits": 8, "group_size": 128}},
    )
    assert model.quantize_config.dynamic_get("transformer.word_embeddings")["bits"] == 8
    _patch_requant_runtime(monkeypatch, requant_module, model)

    requant_module.requant_embed_lm_head(_requant_args(tmp_path, "input"))

    effective = model.quantize_config.dynamic_get("transformer.word_embeddings")
    assert effective["bits"] == 4
    assert effective["group_size"] == 64


@pytest.mark.parametrize(
    ("mode", "model_kwargs", "message"),
    (
        pytest.param(
            "input", {"input_name": None}, "input-embedding", id="missing-input"
        ),
        pytest.param(
            "output",
            {"output_name": None, "lm_head": None},
            "output-embedding",
            id="missing-output",
        ),
    ),
)
def test_rejects_unresolved_embedding_paths(
    monkeypatch, tmp_path, mode, model_kwargs, message
):
    import optimize.requant_embed_lm_head as requant_module

    model = _FakeRequantModel(**model_kwargs)
    _patch_requant_runtime(monkeypatch, requant_module, model)

    with pytest.raises(ValueError, match=message):
        requant_module.requant_embed_lm_head(_requant_args(tmp_path, mode))

    assert model.requantize_kwargs is None
    assert model.saved_to is None


_EMBED_CONFIG = {
    "bits": 4,
    "group_size": 64,
    "sym": True,
    "desc_act": False,
    "act_group_aware": True,
}


def test_embed_config_round_trips_through_a_real_gptq_config():
    """The tool's override must survive save -> load, not only the stub (#5206)."""

    config = GPTQConfig(
        bits=4,
        group_size=64,
        dynamic={
            "model.embed_tokens": dict(_EMBED_CONFIG),
            "lm_head": dict(_EMBED_CONFIG),
        },
    )

    payload = config.to_dict()
    reloaded = GPTQConfig.from_quant_config(payload)

    assert reloaded.dynamic == {
        "model.embed_tokens": _EMBED_CONFIG,
        "lm_head": _EMBED_CONFIG,
    }
    assert "scale_search" not in payload["dynamic"]["model.embed_tokens"]


def test_legacy_scale_search_dynamic_key_still_loads_and_is_stripped_on_save():
    """Checkpoints written before the fix must load and be rewritten cleanly."""

    payload = GPTQConfig(
        bits=4,
        group_size=64,
        dynamic={"model.embed_tokens": dict(_EMBED_CONFIG)},
    ).to_dict()
    payload["dynamic"]["model.embed_tokens"]["scale_search"] = "activation"

    reloaded = GPTQConfig.from_quant_config(payload)

    assert reloaded.dynamic["model.embed_tokens"]["scale_search"] == "activation"
    resaved = reloaded.to_dict()
    assert "scale_search" not in resaved["dynamic"]["model.embed_tokens"]


def test_to_dict_refuses_an_unknown_dynamic_key():
    """A dynamic map the loader would reject must never reach a checkpoint."""

    config = GPTQConfig(bits=4, group_size=64)
    # Assignment is a plain dataclass field, so validation has to happen on save.
    config.dynamic = {"model.embed_tokens": {"scale_seach": "activation"}}

    with pytest.raises(TypeError, match=r"unknown `dynamic\['model\.embed_tokens'\]`"):
        config.to_dict()
