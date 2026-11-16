# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Re-save round trip for a ``gptq_v2`` W4AFP8 checkpoint (issue #4323).

The export shell that a W4A re-save builds must be told which qzero layout the
raw native tensors use *before* they are assigned. Otherwise the shell labels
the source words with its own (target) format and the writer shifts every zero
point by one, so the re-saved file no longer loads.
"""

import glob
import os

import pytest
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization.config import QuantizeConfig
from tests.models.test_w4a_tiny_llama_lifecycle import _tokenizer
from tests.w4a_hardware_marks import FP8_HARDWARE


GROUP_SIZE = 128
CALIBRATION = [
    "A short calibration sentence about a fox. A short calibration sentence about a fox.",
    "Another sentence about the moon. Another sentence about the moon.",
]


def _checkpoint_state(directory):
    state = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        state.update(load_file(path))
    return state


def _unpack_rows(packed):
    """Expand eight low-nibble-first codes per int32 word: (rows, cols) -> (rows*8, cols)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return ((packed.to(torch.int64).unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(
        packed.shape[0] * 8, packed.shape[1]
    )


def _unpack_zero_words(packed):
    """Expand packed qzero words: (groups, cols/8) -> (groups, cols)."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    return ((packed.to(torch.int64).unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF).reshape(
        packed.shape[0], -1
    )


def _logical_zeros(packed, on_disk_format):
    """Independent decode of the stored zero-point words.

    A v1 checkpoint stores ``zero - 1``; W4AFP8 always wants the logical value.
    """
    zeros = _unpack_zero_words(packed)
    if on_disk_format == "gptq":
        zeros = zeros + 1
    return zeros


def _dequant_fp64(state, prefix, on_disk_format):
    """Independent FP64 dequant of one module's stored GPTQ codes."""
    qweight = state[prefix + "qweight"]
    scales = state[prefix + "scales"].double()
    in_features = qweight.shape[0] * 8
    out_features = scales.shape[1]
    codes = _unpack_rows(qweight)[:in_features].double()
    zeros = _logical_zeros(state[prefix + "qzeros"], on_disk_format)[:, :out_features].double()
    groups = torch.arange(in_features) // GROUP_SIZE
    return (codes - zeros[groups]) * scales[groups]


def _qzeros(state):
    return {key: value for key, value in state.items() if key.endswith("qzeros")}


def _tiny_llama(source):
    torch.manual_seed(41)
    tokenizer = _tokenizer()
    config = LlamaConfig(
        vocab_size=32, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, bos_token_id=2, eos_token_id=3, pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer.save_pretrained(source)


def _output(model):
    ids = torch.tensor([[4, 6, 7, 8, 9, 10, 11, 12]], device="cuda")
    with torch.inference_mode():
        return model.model(input_ids=ids).logits.float().cpu()


def _round_trip(tmp_path, source_format, target_format):
    source = tmp_path / "source"
    source.mkdir()
    _tiny_llama(source)
    qcfg = QuantizeConfig(
        bits=4, group_size=GROUP_SIZE, sym=True, desc_act=False,
        activation={"mode": "w4afp8"}, format=source_format, offload_to_disk=False,
    )
    model = GPTQModel.load(str(source), qcfg, device="cuda", dtype=torch.bfloat16)
    model.quantize(CALIBRATION, batch_size=1, backend=BACKEND.GPTQ_W4AFP8)
    first = tmp_path / "first"
    model.save(str(first))
    del model
    torch.cuda.empty_cache()

    baseline = _output(
        GPTQModel.load(str(first), backend=BACKEND.AUTO, device="cuda", dtype=torch.bfloat16)
    )
    first_state = _checkpoint_state(first)
    first_zeros = _qzeros(first_state)

    loaded = GPTQModel.load(str(first), backend=BACKEND.AUTO, device="cuda", dtype=torch.bfloat16)
    loaded.quantize_config.format = target_format
    second = tmp_path / "second"
    loaded.save(str(second))

    second_state = _checkpoint_state(second)
    second_zeros = _qzeros(second_state)
    assert first_zeros.keys() == second_zeros.keys()

    if source_format == target_format:
        # (a) A same-format re-save must not touch a single word.
        for key, value in first_zeros.items():
            torch.testing.assert_close(second_zeros[key], value, rtol=0, atol=0)
    else:
        # A format change rewrites the words, but not the logical zero points.
        for key, value in first_zeros.items():
            assert not torch.equal(second_zeros[key], value)
            torch.testing.assert_close(
                _logical_zeros(second_zeros[key], target_format),
                _logical_zeros(value, source_format),
                rtol=0, atol=0,
            )

    # Independent FP64 oracle over the stored codes: the weights are unchanged.
    for key in first_zeros:
        prefix = key[: -len("qzeros")]
        torch.testing.assert_close(
            _dequant_fp64(second_state, prefix, target_format),
            _dequant_fp64(first_state, prefix, source_format),
            rtol=0, atol=0,
        )

    # (b) The re-saved file loads and produces the same output.
    reloaded = GPTQModel.load(str(second), backend=BACKEND.AUTO, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(_output(reloaded), baseline, rtol=0, atol=0)


@FP8_HARDWARE
def test_w4afp8_gptq_v2_resave_is_bit_identical(tmp_path):
    """Issue #4323: re-saving a gptq_v2 W4AFP8 checkpoint must be a no-op."""
    _round_trip(tmp_path, "gptq_v2", "gptq_v2")


@pytest.mark.parametrize(("source_format", "target_format"), [
    ("gptq_v2", "gptq"),
    ("gptq", "gptq_v2"),
])
@FP8_HARDWARE
def test_w4afp8_resave_format_change_preserves_logical_zeros(tmp_path, source_format, target_format):
    """The W4A export shell must declare the source layout before the load."""
    _round_trip(tmp_path, source_format, target_format)
