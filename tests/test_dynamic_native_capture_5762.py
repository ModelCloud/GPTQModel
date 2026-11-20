# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""A dynamic GPTAQ/FOEM rule must still get the native capture pass (#5762).

The run plan used to decide the ``NATIVE`` step from the global options only.
A ``dynamic`` rule that enables GPTAQ or FOEM for one projection then built an
active correction task in ``GPTQProcessor.preprocess`` and raised
``KeyError: 'native_inp'`` because nothing had captured the inputs. The capture
layout must also be per sample: with ``batch_size > 1`` a batch-level entry does
not match the per-sample replay.
"""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.models.quantization_run_plan import ProcessorStep, QuantizationRunPlan
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization.config import (
    FORMAT,
    FOEMConfig,
    GPTAQConfig,
    GPTQConfig,
    QuantizeConfig,
)


VOCAB = {"<pad>": 0, "<s>": 1, "eos": 2, **{f"w{i}": i for i in range(3, 128)}}
CALIBRATION = [
    {"input_ids": torch.randint(4, 128, (32,)).tolist()}
    for _ in range(4)
]


def _tiny_checkpoint(directory):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(directory)
    tokenizer = Tokenizer(WordLevel(vocab=VOCAB, unk_token="<pad>"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="eos",
    ).save_pretrained(directory)


@pytest.mark.parametrize(
    ("method", "alpha", "batch_size"),
    [
        ("gptaq", 0.25, 1),
        ("gptaq", 0.25, 2),
        ("foem", 0.25, 2),
        ("foem", 0.0, 2),
    ],
)
def test_dynamic_rule_quantizes_with_native_capture(tmp_path, method, alpha, batch_size):
    source = tmp_path / "source"
    source.mkdir()
    _tiny_checkpoint(source)

    rule = {r"+:model\.layers\.0\.self_attn\.q_proj$": {method: {"alpha": alpha, "device": "cpu"}}}
    config = GPTQConfig(
        bits=4,
        group_size=32,
        device="cpu",
        offload_to_disk=False,
        dynamic=rule,
    )
    model = GPTQModel.load(str(source), quantize_config=config, dtype=torch.float16, backend=BACKEND.TORCH)

    model.quantize(
        CalibrationConfig(dataset=CALIBRATION, batch_size=batch_size),
        backend=BACKEND.TORCH,
    )

    q_proj = model.model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, BaseQuantLinear)
    assert torch.isfinite(q_proj.qweight.float()).all()


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (QuantizeConfig(bits=4, gptaq=GPTAQConfig()), True),
        (QuantizeConfig(bits=4, foem=FOEMConfig(alpha=0.25)), True),
        (QuantizeConfig(bits=4, foem=FOEMConfig(alpha=0.0)), False),
        (QuantizeConfig(bits=4, dynamic={r".*q_proj$": {"gptaq": {"alpha": 0.0}}}), True),
        (QuantizeConfig(bits=4, dynamic={r".*q_proj$": {"foem": {"alpha": 0.25}}}), True),
        (QuantizeConfig(bits=4, dynamic={r".*q_proj$": {"foem": {"beta": 0.2}}}), False),
        (QuantizeConfig(bits=4, dynamic={r".*q_proj$": {"bits": 8}}), False),
    ],
)
def test_native_correction_requested_sees_dynamic_rules(config, expected):
    from gptqmodel.quantization.config import native_correction_requested

    assert native_correction_requested(config) is expected


def _plan(config):
    return QuantizationRunPlan.build(
        config,
        format_code=FORMAT.GPTQ,
        pack_backend=BACKEND.TORCH,
        loop_backend=BACKEND.TORCH,
        batch_size=1,
        needs_lora=False,
    )


@pytest.mark.parametrize(
    ("dynamic", "expected"),
    [
        ({r"+:model\\.layers\\.0\\.self_attn\\.q_proj$": {"gptaq": {"alpha": 0.25}}}, True),
        ({r"+:model\\.layers\\.0\\.self_attn\\.q_proj$": {"foem": {"alpha": 0.25}}}, True),
        ({r"+:model\\.layers\\.0\\.self_attn\\.q_proj$": {"foem": {"alpha": 0.0}}}, False),
        ({r"+:model\\.layers\\.0\\.self_attn\\.q_proj$": {"bits": 8}}, False),
    ],
)
def test_run_plan_adds_native_step_for_dynamic_correction(dynamic, expected):
    plan = _plan(GPTQConfig(bits=4, dynamic=dynamic))
    assert (ProcessorStep.NATIVE in plan.processors) is expected


@pytest.mark.parametrize(
    ("dynamic", "expected"),
    [
        ({r".*q_proj$": {"gptaq": {"alpha": 0.25}}}, True),
        ({r".*q_proj$": {"foem": {"alpha": 0.25}}}, True),
        ({r".*q_proj$": {"foem": {"beta": 0.2}}}, False),
    ],
)
def test_native_capture_layout_follows_dynamic_rules(dynamic, expected):
    """A dynamic correction task consumes per-sample entries; capture must match."""

    processor = NativeProcessor(
        tokenizer=None,
        qcfg=QuantizeConfig(bits=4, dynamic=dynamic),
        calibration=CalibrationConfig(),
    )

    assert processor._native_per_sample is expected


def _mixed_padding_dataset():
    """Rows padded to S=32 where only the first ``n`` tokens are real."""

    rows = []
    for length in (32, 20, 28, 12):
        ids = torch.randint(4, 128, (32,)).tolist()
        mask = [1] * length + [0] * (32 - length)
        rows.append({"input_ids": ids, "attention_mask": mask})
    return rows


@pytest.mark.parametrize(("method", "alpha"), [("gptaq", 0.25), ("foem", 0.25)])
def test_dynamic_rule_quantizes_with_mixed_padding(tmp_path, method, alpha):
    """A padded batch still replays only the real tokens of each sample."""

    source = tmp_path / "source"
    source.mkdir()
    _tiny_checkpoint(source)

    rule = {r"+:model\.layers\.0\.self_attn\.q_proj$": {method: {"alpha": alpha, "device": "cpu"}}}
    config = GPTQConfig(
        bits=4,
        group_size=32,
        device="cpu",
        offload_to_disk=False,
        dynamic=rule,
    )
    model = GPTQModel.load(str(source), quantize_config=config, dtype=torch.float16, backend=BACKEND.TORCH)

    model.quantize(
        CalibrationConfig(dataset=_mixed_padding_dataset(), batch_size=2),
        backend=BACKEND.TORCH,
    )

    q_proj = model.model.model.layers[0].self_attn.q_proj
    assert isinstance(q_proj, BaseQuantLinear)
    assert torch.isfinite(q_proj.qweight.float()).all()
