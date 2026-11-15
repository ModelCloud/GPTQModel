# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Concurrent module-tree branches must be calibrated on their own inputs.

The regression shape matches #4116 item 1: two text decoder layers plus a
sibling ``vision_tower.layers`` stack.  Before the fix the looper concatenated
the stacks, captured only the first text layer, and replayed the vision stack on
the text decoder's last hidden state.  Equal-width branches were silently
mis-quantized; a different vision width raised a shape error mid-quantize.
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from gptqmodel import BACKEND, QuantizeConfig
from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


pytestmark = [pytest.mark.cpu, pytest.mark.slow]

_TEXT_HIDDEN = 896
_VISION_HIDDEN = 512
_VOCAB = 128
_PIXEL_TOKENS = 8
_PIXEL_FEATURES = 16
_BATCHES = 8


class TinyParallelCalibrationConfig(PretrainedConfig):
    model_type = "tiny_parallel_calibration_4116"

    def __init__(
        self,
        text_hidden: int = _TEXT_HIDDEN,
        vision_hidden: int = _VISION_HIDDEN,
        vocab_size: int = _VOCAB,
        num_text_layers: int = 2,
        num_vision_layers: int = 2,
        **kwargs,
    ):
        self.text_hidden = text_hidden
        self.vision_hidden = vision_hidden
        self.vocab_size = vocab_size
        self.num_text_layers = num_text_layers
        self.num_vision_layers = num_vision_layers
        self.use_cache = False
        self.tie_word_embeddings = False
        super().__init__(**kwargs)


class _Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.o_proj(
            self.q_proj(hidden_states)
            + self.k_proj(hidden_states)
            + self.v_proj(hidden_states)
        )


class _TextMlp(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.up_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.down_proj = nn.Linear(hidden * 2, hidden, bias=False)

    def forward(self, hidden_states):
        return self.down_proj(
            torch.nn.functional.gelu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class _VisionMlp(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states):
        return self.fc2(torch.nn.functional.gelu(self.fc1(hidden_states)))


class _TextBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.mlp = _TextMlp(hidden)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        return hidden_states + self.mlp(hidden_states + self.self_attn(hidden_states))


class _VisionBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.mlp = _VisionMlp(hidden)

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.mlp(hidden_states + self.self_attn(hidden_states))


class TinyParallelCalibrationModel(PreTrainedModel):
    config_class = TinyParallelCalibrationConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = []

    def __init__(self, config: TinyParallelCalibrationConfig):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.text_hidden)
        self.model.layers = nn.ModuleList(
            [_TextBlock(config.text_hidden) for _ in range(config.num_text_layers)]
        )
        self.model.norm = nn.LayerNorm(config.text_hidden)
        self.vision_tower = nn.Module()
        self.vision_tower.patch_embed = nn.Linear(
            _PIXEL_FEATURES, config.vision_hidden, bias=False
        )
        self.vision_tower.layers = nn.ModuleList(
            [_VisionBlock(config.vision_hidden) for _ in range(config.num_vision_layers)]
        )
        self.lm_head = nn.Linear(config.text_hidden, config.vocab_size, bias=False)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        text = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            text = layer(text, attention_mask=attention_mask)
        text = self.model.norm(text)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.vision_tower.patch_embed.weight.dtype)
            vision = self.vision_tower.patch_embed(pixel_values)
            for layer in self.vision_tower.layers:
                vision = layer(vision)
        return self.lm_head(text)


class ParallelCalibrationQModel(BaseQModel):
    """Unknown model type: the tree comes from ``module_tree="auto"``."""

    loader = AutoModelForCausalLM
    layer_modules_strict = False
    module_tree = None

    def prepare_dataset(self, calibration_dataset, **kwargs):
        # Keep multimodal keys. The base normalizer only keeps input_ids and
        # attention_mask, which would hide the vision branch entirely.
        return calibration_dataset


AutoConfig.register(
    TinyParallelCalibrationConfig.model_type,
    TinyParallelCalibrationConfig,
    exist_ok=True,
)
AutoModelForCausalLM.register(
    TinyParallelCalibrationConfig,
    TinyParallelCalibrationModel,
    exist_ok=True,
)


def _build_fixture(tmp_path, *, text_hidden: int, vision_hidden: int):
    config = TinyParallelCalibrationConfig(
        text_hidden=text_hidden,
        vision_hidden=vision_hidden,
    )
    TinyParallelCalibrationModel(config).save_pretrained(tmp_path)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        ["tiny parallel calibration sample"] * 4,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(tmp_path)
    return fast


def _calibration(tokenizer, *, include_pixel_values: bool):
    rows = []
    for index in range(_BATCHES):
        encoded = tokenizer(
            f"tiny parallel calibration sample {index}", return_tensors="pt"
        )
        row = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if include_pixel_values:
            row["pixel_values"] = torch.randn(
                1, _PIXEL_TOKENS, _PIXEL_FEATURES, dtype=torch.float32
            )
        rows.append(row)
    return rows


def _quantize(tmp_path, *, text_hidden: int, vision_hidden: int, include_pixel_values: bool):
    tokenizer = _build_fixture(
        tmp_path, text_hidden=text_hidden, vision_hidden=vision_hidden
    )
    model = ParallelCalibrationQModel.from_pretrained(
        str(tmp_path),
        quantize_config=QuantizeConfig(
            bits=4, group_size=32, device="cpu", desc_act=False
        ),
        backend=BACKEND.TORCH,
        module_tree="auto",
    )
    assert model.extract_layers_node() == ["model.layers", "vision_tower.layers"]

    vision_inputs = []
    text_outputs = []
    text_last = model.model.model.layers[-1]
    vision_first = model.model.vision_tower.layers[0]

    def _record_text(module, args, output):
        text_outputs.append(output.detach().clone())

    def _record_vision(module, args):
        vision_inputs.append(args[0].detach().clone())

    text_handle = text_last.register_forward_hook(_record_text)
    vision_handle = vision_first.register_forward_pre_hook(_record_vision)
    try:
        model.quantize(
            _calibration(tokenizer, include_pixel_values=include_pixel_values),
            batch_size=1,
            backend=BACKEND.TORCH,
            calibration_data_min_length=1,
        )
    finally:
        text_handle.remove()
        vision_handle.remove()

    return model, vision_inputs, text_outputs


def _count_quantized(model, prefix: str) -> int:
    return sum(
        1
        for name, module in model.model.named_modules()
        if name.startswith(prefix) and isinstance(module, TorchLinear)
    )


@pytest.mark.parametrize("vision_hidden", [_TEXT_HIDDEN, _VISION_HIDDEN])
def test_parallel_branch_calibration_uses_each_branch_inputs(tmp_path, vision_hidden):
    model, vision_inputs, text_outputs = _quantize(
        tmp_path,
        text_hidden=_TEXT_HIDDEN,
        vision_hidden=vision_hidden,
        include_pixel_values=True,
    )

    assert vision_inputs
    assert text_outputs
    # In the buggy path the vision stack is replayed on the text decoder's last
    # output, so at least one recorded vision input is exactly a text output.
    matches = 0
    for vision_input in vision_inputs:
        for text_output in text_outputs:
            if vision_input.shape == text_output.shape and torch.equal(
                vision_input, text_output
            ):
                matches += 1
                break
    assert matches == 0, (
        f"{matches}/{len(vision_inputs)} vision inputs are text decoder outputs"
    )

    assert _count_quantized(model, "vision_tower.layers.") == 12
    assert _count_quantized(model, "model.layers.") == 14


def test_parallel_branch_calibration_fails_when_branch_is_unreached(tmp_path):
    with pytest.raises(ValueError, match="never reached the first layer") as excinfo:
        _quantize(
            tmp_path,
            text_hidden=_TEXT_HIDDEN,
            vision_hidden=_TEXT_HIDDEN,
            include_pixel_values=False,
        )

    message = str(excinfo.value)
    assert "vision_tower.layers.0" in message
    assert "Refusing to calibrate a branch on another branch's activations" in message
