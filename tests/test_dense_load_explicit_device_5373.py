# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""An explicit ``device`` must place a dense load, not Transformers' ``auto`` (#5373).

``GPTQModel.load(dense, device="cpu")`` used the resolved device for dtype
selection and then forwarded ``device_map="auto"`` to Transformers whenever the
caller omitted ``device_map``. On a CUDA host the auto policy fills the first
visible GPU, so the returned model lived on ``cuda:0`` although the caller asked
for CPU. The dense path now turns an explicit device into a root mapping.
"""

from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import GPTQModel
from gptqmodel.models._const import DEVICE
from gptqmodel.models.loader import _dense_placement_device_map
from tests.tiny_remote_models import build_wordlevel_tokenizer


pytestmark = [pytest.mark.cpu]


def _tiny_dense(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(directory)
    build_wordlevel_tokenizer().save_pretrained(directory)
    return directory


def test_dense_placement_keeps_the_auto_policy_when_nothing_is_requested():
    assert _dense_placement_device_map(None, None, None) == "auto"


def test_dense_placement_turns_an_explicit_device_into_a_root_map():
    assert _dense_placement_device_map("cpu", None, torch.device("cpu")) == {"": "cpu"}
    assert _dense_placement_device_map(
        torch.device("cpu"), None, torch.device("cpu")
    ) == {"": "cpu"}
    assert _dense_placement_device_map(
        "cuda:0", None, torch.device("cuda", 0)
    ) == {"": "cuda:0"}
    # The DEVICE enum resolution drops the ordinal, so the family target is the
    # accelerator's default device.
    assert _dense_placement_device_map(DEVICE.CPU, None, torch.device("cpu")) == {"": "cpu"}


def test_dense_placement_lets_an_explicit_device_map_win():
    explicit = {"": "cuda:0"}

    assert _dense_placement_device_map("cpu", explicit, torch.device("cpu")) is explicit
    assert _dense_placement_device_map("cpu", "balanced", torch.device("cpu")) == "balanced"


def test_dense_placement_keeps_auto_for_a_device_pool():
    pool = (torch.device("cuda", 0), torch.device("cuda", 1))

    assert _dense_placement_device_map(["cuda:0", "cuda:1"], None, pool) == "auto"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="reproducing the auto placement needs a visible CUDA accelerator",
)
def test_dense_load_honors_device_cpu_on_a_cuda_host(tmp_path):
    checkpoint = _tiny_dense(tmp_path / "dense")

    model = GPTQModel.load(str(checkpoint), device="cpu")
    devices = {str(parameter.device) for parameter in model.model.parameters()}

    assert devices == {"cpu"}, f"device='cpu' must not load onto the visible GPU: {devices}"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="reproducing the auto placement needs a visible CUDA accelerator",
)
def test_dense_load_honors_an_explicit_cuda_ordinal(tmp_path):
    checkpoint = _tiny_dense(tmp_path / "dense")

    model = GPTQModel.load(str(checkpoint), device="cuda:0")
    devices = {str(parameter.device) for parameter in model.model.parameters()}

    assert devices == {"cuda:0"}, devices
