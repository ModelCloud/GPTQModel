# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Looped decoders must calibrate on every loop, not just loop 0.

Issue #4093 item 6: Ouro (``total_ut_steps=4``) and Nanbeige 4.2
(``num_loops=2``) reuse the same physical layers several times per forward.
The default capture stops at the first layer, so each processor's Hessian only
saw loop 0's input (cosine -0.03..0.42 for Ouro, 0.54/0.68 for Nanbeige).
``LoopedLayerCaptureMixin`` runs one full forward and replays every invocation,
the mechanism #4055 introduced for HRM-Text.
"""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    create_causal_mask,
)

from gptqmodel.models.definitions.hrm_text import HrmTextQModel
from gptqmodel.models.definitions.nanbeige import NanbeigeQModel
from gptqmodel.models.definitions.ouro import OuroQModel
from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.sensenova_loop import SenseNovaLoopQModel
from gptqmodel.models.looped_capture import LoopedLayerCaptureMixin
from gptqmodel.utils.model import get_layers_with_prefixes


OURO_PATH = Path("/monster/data/model/Ouro-1.4B")
NANBEIGE_PATH = Path("/monster/data/model/Nanbeige4.2-3B")


class _LoopLayer(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.act = nn.Tanh()

    def forward(self, hidden_states):
        return hidden_states + self.act(self.proj(hidden_states))


class _SyntheticLoopedModel(nn.Module):
    """Every physical layer runs ``loops`` times per forward, like Ouro/Nanbeige."""

    def __init__(self, layers: int = 2, dim: int = 8, loops: int = 4, vocab: int = 32):
        super().__init__()
        self.loops = loops
        self.model = nn.Module()
        self.model.embed = nn.Embedding(vocab, dim)
        self.model.layers = nn.ModuleList([_LoopLayer(dim) for _ in range(layers)])
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        hidden = self.model.embed(input_ids)
        for _ in range(self.loops):
            for layer in self.model.layers:
                hidden = layer(hidden)
        return self.lm_head(hidden)


class _SyntheticLoopedQModel(HrmTextQModel):
    @classmethod
    def extract_layers_node(cls):
        return ["model.layers"]

    @classmethod
    def get_base_modules(cls, model):
        return ["model.embed"]

    @classmethod
    def get_modules_with_direct_meta_tensors(cls, model):
        return []


def _wrapper(model, qmodel_cls):
    instance = object.__new__(qmodel_cls)
    nn.Module.__init__(instance)
    instance._turtle_lock = threading.RLock()
    instance.model = model
    instance.turtle_model = None
    instance.quantize_config = SimpleNamespace(device=torch.device("cpu"))
    return instance


def _capture(instance, model, qmodel_cls, examples, loops):
    layers, layer_names = get_layers_with_prefixes(model, qmodel_cls.extract_layers_node())
    cache = instance.build_quantization_input_cache(
        layers=layers,
        calibration_data=examples,
        use_cache=False,
        layer_names=layer_names,
    )
    return layers, cache


def test_looped_capture_records_every_loop():
    torch.manual_seed(0)
    model = _SyntheticLoopedModel(loops=4)
    instance = _wrapper(model, _SyntheticLoopedQModel)
    examples = [
        {
            "input_ids": torch.randint(0, 32, (1, 6)),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }
        for _ in range(2)
    ]

    layers, cache = _capture(instance, model, _SyntheticLoopedQModel, examples, 4)

    per_layer = instance._looped_per_layer
    assert [len(per_layer[id(layer)]["layer_inputs"]) for layer in layers] == [8, 8]
    assert cache.captured_batch_count == 8
    # The replay stage bounds itself by this count, otherwise it truncates to
    # loop 0 (2 calibration batches instead of 2 * 4 invocations).
    assert cache.captured_batch_count == len(examples) * 4

    first = per_layer[id(layers[0])]["layer_inputs"]
    inputs = [batch[0] for batch in first[:4]]
    for later in inputs[1:]:
        cosine = torch.nn.functional.cosine_similarity(inputs[0].flatten(), later.flatten(), dim=0)
        assert cosine.item() < 0.99, "loop 1..3 replayed loop 0's input"


def test_looped_capture_serves_each_layer_its_own_invocations():
    torch.manual_seed(0)
    model = _SyntheticLoopedModel(loops=2)
    instance = _wrapper(model, _SyntheticLoopedQModel)
    examples = [
        {
            "input_ids": torch.randint(0, 32, (1, 6)),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }
    ]

    layers, cache = _capture(instance, model, _SyntheticLoopedQModel, examples, 2)

    for layer in layers:
        instance.pre_quantize(layer)
        assert cache.layer_inputs is instance._looped_per_layer[id(layer)]["layer_inputs"]
        assert len(cache.layer_inputs) == 2
    instance.after_quantize()
    assert instance._looped_per_layer is None


@pytest.mark.parametrize(
    ("qmodel_cls", "expected_loops"),
    [(OuroQModel, 4), (NanbeigeQModel, 2), (HrmTextQModel, None)],
)
def test_looped_families_use_the_shared_capture(qmodel_cls, expected_loops):
    """A family that regresses to the single-chain capture loses this mixin."""

    assert issubclass(qmodel_cls, LoopedLayerCaptureMixin)
    if expected_loops is None:
        # HRM-Text is the reference implementation; #4055's counts are covered
        # by tests/test_hrm_text_support.py.
        return
    assert expected_loops > 1


def _real_capture(qmodel_cls, source: Path, tmp_path: Path, overrides, loops):
    from tests.tiny_remote_models import build_tiny_remote_model

    path = build_tiny_remote_model(source, tmp_path / "tiny", overrides)
    if qmodel_cls is OuroQModel:
        OuroQModel.before_model_load(OuroQModel, str(path), False)
    model = qmodel_cls.loader.from_pretrained(str(path)).eval()
    instance = _wrapper(model, qmodel_cls)
    example = {
        "input_ids": torch.randint(4, 128, (1, 16)),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
    }
    layers, cache = _capture(instance, model, qmodel_cls, [example], loops)

    per_layer = instance._looped_per_layer
    counts = [len(per_layer[id(layer)]["layer_inputs"]) for layer in layers]
    assert counts == [loops] * len(layers)
    assert cache.captured_batch_count == loops

    inputs = [batch[0] for batch in per_layer[id(layers[0])]["layer_inputs"]]
    cosines = [
        torch.nn.functional.cosine_similarity(inputs[0].flatten(), later.flatten(), dim=0).item()
        for later in inputs[1:]
    ]
    assert all(cosine < 0.99 for cosine in cosines), cosines
    return counts, cosines


@pytest.mark.skipif(not OURO_PATH.is_dir(), reason=f"local Ouro fixture is unavailable: {OURO_PATH}")
def test_ouro_captures_all_four_steps(tmp_path):
    counts, cosines = _real_capture(
        OuroQModel,
        OURO_PATH,
        tmp_path,
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "total_ut_steps": 4,
            "layer_types": ["full_attention", "full_attention"],
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        4,
    )
    assert counts == [4, 4]
    print({"ouro_invocations": counts, "per_loop_cosine": cosines})


@pytest.mark.skipif(
    not NANBEIGE_PATH.is_dir(),
    reason=f"local Nanbeige fixture is unavailable: {NANBEIGE_PATH}",
)
def test_nanbeige_captures_both_loops(tmp_path):
    counts, cosines = _real_capture(
        NanbeigeQModel,
        NANBEIGE_PATH,
        tmp_path,
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "num_loops": 2,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-5,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        2,
    )
    assert counts == [2, 2]
    print({"nanbeige_invocations": counts, "per_loop_cosine": cosines})


class _LoopedTextModel(nn.Module):
    """Qwen3.5-shaped text decoder whose block plan matches SenseNova-Loop.

    `SenseNovaLoopTextModel.forward` runs
    `0..start-1, (start..end-1) * eval_count, end..num_layers-1` over the same
    physical `ModuleList`, so a layer in the looped range is invoked
    `eval_count` times per sample and a post-loop layer sees the state after
    `eval_count` iterations. The block signature mirrors the published remote
    `MHCDecoderLayer.forward`, including the positional-only-in-fact
    `position_embeddings` tuple that the real forward passes each layer.
    """

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_idx=index) for index in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)

    def execution_plan(self):
        loop = self.config.loop_settings
        start, end, count = loop["start"], loop["end"], loop["eval_count"]
        plan = list(range(start))
        plan += list(range(start, end)) * count
        plan += list(range(end, self.config.num_hidden_layers))
        return plan

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        del use_cache, kwargs
        hidden_states = self.embed_tokens(input_ids)
        position_ids = (
            torch.arange(hidden_states.shape[1], device=hidden_states.device)
            .view(1, 1, -1)
            .expand(4, hidden_states.shape[0], -1)
        )
        text_position_ids = position_ids[0]
        # Qwen3.5's multi-axis RoPE consumes the three vision axes; the text
        # decoder forwards `position_ids[1:]` exactly as the published model does.
        position_embeddings = self.rotary_emb(hidden_states, position_ids[1:])
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        for index in self.execution_plan():
            hidden_states = self.layers[index](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return self.norm(hidden_states)


def _sensenova_like_config(layers: int = 8, hidden: int = 32) -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        max_position_embeddings=64,
        # Full attention only: Qwen3.5's gated-delta layers call CUDA-only
        # `causal_conv1d` kernels, and the capture path under test is
        # independent of the attention type.
        layer_types=["full_attention"] * layers,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    # The remote `SenseNovaLoopTextModel` reads its loop window from the
    # checkpoint config; transformers 5 rejects undeclared attributes, so the
    # tiny fixture installs the published window scaled to 8 physical layers.
    config.__dict__["loop_settings"] = {"start": 2, "end": 6, "eval_count": 2}
    return config


def _count_invocations(model, examples):
    counts = dict.fromkeys(range(len(model.layers)), 0)

    def make_hook(index):
        def hook(_module, _args, _kwargs):
            counts[index] += 1

        return hook

    handles = [
        layer.register_forward_pre_hook(make_hook(index), with_kwargs=True) for index, layer in enumerate(model.layers)
    ]
    try:
        with torch.no_grad():
            for example in examples:
                model(**example)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def _sensenova_capture(examples):
    config = _sensenova_like_config()
    model = _LoopedTextModel(config).eval()
    instance = _wrapper(model, SenseNovaLoopQModel)
    layers, _ = get_layers_with_prefixes(model, ["layers"])

    # Post-loop layer inputs: the deployed model only ever feeds them the
    # hidden state after `eval_count` loop visits, so those are the inputs
    # whose distribution the calibration must reproduce.
    first_post_loop = config.loop_settings["end"]
    observed = []

    def record(_module, args, kwargs):
        hidden_states = args[0] if args else kwargs["hidden_states"]
        observed.append(hidden_states.detach().clone())

    handle = layers[first_post_loop].register_forward_pre_hook(record, with_kwargs=True)
    try:
        with torch.no_grad():
            for example in examples:
                model(**example)
    finally:
        handle.remove()

    cache = instance.build_quantization_input_cache(
        layers=layers,
        calibration_data=examples,
        use_cache=False,
        layer_names=[f"layers.{index}" for index in range(len(layers))],
    )
    return config, model, instance, layers, cache, observed


def test_sensenova_loop_definition_opts_into_looped_capture():
    """#5081: a plain chain capture loses half of every looped layer's inputs."""

    assert issubclass(SenseNovaLoopQModel, LoopedLayerCaptureMixin)


def test_sensenova_loop_capture_reproduces_the_execution_plan():
    torch.manual_seed(0)
    config = _sensenova_like_config()
    examples = [
        {
            "input_ids": torch.randint(4, 128, (1, 6)),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }
        for _ in range(2)
    ]
    _, model, instance, layers, cache, observed = _sensenova_capture(examples)

    # Independent oracle: the model's own block plan, not the config arithmetic,
    # decides how many times each physical layer runs for one sample.
    per_sample = Counter(model.execution_plan())
    expected = {index: len(examples) * per_sample[index] for index in per_sample}
    assert len(per_sample) == config.num_hidden_layers

    captured = [len(instance._looped_per_layer[id(layer)]["layer_inputs"]) for layer in layers]
    assert captured == [expected[index] for index in range(len(layers))]
    assert cache.captured_batch_count == max(captured)

    # The layer's own replay feed must be the true post-loop distribution, not
    # the one-loop chain the default capture would have produced.
    first_post_loop = config.loop_settings["end"]
    replayed = instance._looped_per_layer[id(layers[first_post_loop])]["layer_inputs"]
    assert len(replayed) == len(observed)
    for captured_batch, real_input in zip(replayed, observed):
        torch.testing.assert_close(captured_batch[0], real_input)

    # The plain Qwen3.5 chain capture only walks each physical layer once, so it
    # can never serve the looped invocations this test requires.
    assert not hasattr(Qwen3_5QModel, "build_quantization_input_cache")
    assert _count_invocations(model, examples) == expected
