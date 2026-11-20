# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""A per-module AWQ fallback threshold must select the fallback grid (#5820).

``GPTQModel.quantize()`` accepts a ``dynamic`` fallback threshold for one
projection, but AWQ compared coverage against the run-wide ``Fallback`` in
``_starved_group_members`` and gated ``_quantize_layer`` on the same object.
The dynamic override was merged only by ``_fallback_strategy_for_module``,
after ``apply_quant`` already knew the module would fall back, so a dynamic
STDCLIP threshold of 4096 produced exactly the disabled-threshold pack: 255 of
256 packed words differed from the run-wide STDCLIP control, and the maximum
scale error against the independent FP64 STDCLIP grid was 0.0055802 instead of
0.0000109645.
"""

import os

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


# The portable AWQ_TORCH path is the one under test; keep the fixture on CPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import AWQConfig, Fallback


TARGET = "model.layers.0.mlp.down_proj"
GROUP_SIZE = 16
BITS = 4
CALIBRATION_IDS = list(range(3, 35))


def _stdclip_scale_oracle(weight: torch.Tensor) -> torch.Tensor:
    """Independent FP64 AWQ STDCLIP scales for the raw ``[out, in]`` weight.

    STDCLIP bounds a group at ``|mean| + 3 * std`` (population std) and divides
    by the symmetric INT4 maximum (7), which is what the fallback grid uses when
    no AWQ scale has been folded into the module.
    """

    groups = weight.reshape(-1, GROUP_SIZE).double()
    bound = groups.mean(1, keepdim=True).abs() + 3 * groups.std(1, keepdim=True, unbiased=False)
    expected = (bound / (2 ** (BITS - 1) - 1)).reshape(weight.shape[0], -1).T
    return expected


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory):
    path = tmp_path_factory.mktemp("awq-dynamic-fallback-5820")

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    source = LlamaForCausalLM(config).half().eval()
    source.save_pretrained(path)

    tokenizer = Tokenizer(models.WordLevel({f"w{i}": i for i in range(128)}, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="w0", pad_token="w0").save_pretrained(path)

    weight = source.get_submodule(TARGET).weight.detach().clone()
    return str(path), weight


def _quantize(path: str, fallback: Fallback, dynamic=None):
    config = AWQConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=True,
        device="cpu",
        offload_to_disk=False,
        fallback=fallback,
        dynamic=dynamic,
    )
    model = GPTQModel.load(path, config, dtype=torch.float16, backend=BACKEND.AWQ_TORCH)
    if dynamic is not None:
        # The public config must resolve the override before quantization runs.
        assert model.quantize_config.resolve_module_config(TARGET).get("fallback") == dynamic[TARGET]["fallback"]
    calibration = CalibrationConfig(
        dataset=[
            {
                "input_ids": CALIBRATION_IDS,
                "attention_mask": [1] * len(CALIBRATION_IDS),
            }
        ],
        min_length=1,
    )
    model.quantize(calibration, backend=BACKEND.AWQ_TORCH)
    return model.model.get_submodule(TARGET)


@pytest.fixture(scope="module")
def packs(tiny_llama):
    path, weight = tiny_llama
    disabled = Fallback(threshold=0)
    above = {TARGET: {"fallback": {"strategy": "stdclip", "threshold": 4096}}}
    below = {TARGET: {"fallback": {"strategy": "stdclip", "threshold": 1}}}
    return {
        "weight": weight,
        "disabled": _quantize(path, disabled),
        "global": _quantize(path, Fallback(strategy="stdclip", threshold=4096)),
        "dynamic_above": _quantize(path, disabled, above),
        "dynamic_below": _quantize(path, disabled, below),
    }


@pytest.mark.cpu
@pytest.mark.slow
def test_dynamic_threshold_selects_the_fallback_grid(packs):
    disabled = packs["disabled"]
    global_stdclip = packs["global"]
    dynamic = packs["dynamic_above"]
    expected_scale = _stdclip_scale_oracle(packs["weight"])

    # The dynamic threshold must select the same grid as the run-wide one.
    assert torch.equal(dynamic.qweight, global_stdclip.qweight)
    assert torch.equal(dynamic.qzeros, global_stdclip.qzeros)
    assert torch.equal(dynamic.scales, global_stdclip.scales)

    # And it must not be the disabled control any more: it differs exactly as
    # much as the run-wide STDCLIP control does (255 of 256 packed words here).
    assert not torch.equal(dynamic.qweight, disabled.qweight)
    assert int((dynamic.qweight != disabled.qweight).sum()) == int((global_stdclip.qweight != disabled.qweight).sum())

    # Both fallback runs land on the independent STDCLIP grid; the AWQ control
    # is far away. fp16 storage of the fp64 scales accounts for the residual.
    dynamic_error = (dynamic.scales.double() - expected_scale).abs().amax().item()
    global_error = (global_stdclip.scales.double() - expected_scale).abs().amax().item()
    disabled_error = (disabled.scales.double() - expected_scale).abs().amax().item()
    assert global_error < 5e-5
    assert dynamic_error == pytest.approx(global_error, abs=1e-12)
    assert disabled_error > 1e-3


@pytest.mark.cpu
@pytest.mark.slow
def test_dynamic_threshold_below_observed_coverage_keeps_awq(packs):
    """A threshold under the captured token count must not trigger fallback."""

    disabled = packs["disabled"]
    below = packs["dynamic_below"]

    assert torch.equal(below.qweight, disabled.qweight)
    assert torch.equal(below.qzeros, disabled.qzeros)
    assert torch.equal(below.scales, disabled.scales)
