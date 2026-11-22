# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ token-row replay must not hand a recurrent mixer an attention grid (#4671).

With the default ``quantize(batch_size=1)``, ``transformers`` 5 builds every
per-layer-type mask from an unpadded batch and therefore returns ``None`` for
both ``create_recurrent_attention_mask`` and the causal builders. AWQ replay
treated that ``None`` as "the module relied on its internal causal path" and
fell back to the additive ``[1, 1, T, T]`` grid. A recurrent/convolution mixer
applies its mask as ``hidden * attention_mask[:, :, None]``, so the grid raised
on the first scale-search replay for every Mamba/conv hybrid
(Granite-4.0-H, LFM2, Nemotron-H, OLMo-Hybrid).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gptqmodel import CalibrationConfig
from gptqmodel.looper.awq_processor import (
    AWQProcessor,
    _is_recurrent_attention_layer_type,
)


def _bare_processor(*, masks, position_ids=None) -> AWQProcessor:
    processor = object.__new__(AWQProcessor)
    processor.inputs_cache = SimpleNamespace(
        attention_masks=list(masks),
        position_ids=list(position_ids or []),
    )
    return processor


@pytest.mark.parametrize("layer_type", ["linear_attention", "conv"])
def test_recurrent_layer_types_are_recognised(layer_type):
    assert _is_recurrent_attention_layer_type(layer_type)


@pytest.mark.parametrize(
    "layer_type", ["full_attention", "sliding_attention", "chunked_attention", None, ""]
)
def test_attention_layer_types_are_not_recurrent(layer_type):
    assert not _is_recurrent_attention_layer_type(layer_type)


def test_recurrent_replay_uses_a_planar_keep_mask():
    """The packed replay must give a recurrent mixer a ``[batch, seq]`` mask."""

    processor = _bare_processor(masks=[None])
    tensors = [torch.zeros(1, 5, 8)]

    kwargs = processor._token_row_replay_kwargs(
        tensors,
        [0],
        [(0, row) for row in range(5)],
        torch.float32,
        mask_overrides={0: None},
        recurrent=True,
    )

    mask = kwargs["attention_mask"]
    assert mask.dim() == 2, f"recurrent replay produced a {mask.dim()}-D mask"
    assert tuple(mask.shape) == (1, 5)
    assert bool(mask.all())
    # The mixer consumes it as `hidden * mask[:, :, None]`.
    assert tuple(mask[:, :, None].shape) == (1, 5, 1)
    assert kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 4]]


def test_attention_replay_keeps_the_block_diagonal_grid():
    """An attention layer still needs the explicit causal grid."""

    processor = _bare_processor(masks=[None])

    kwargs = processor._token_row_replay_kwargs(
        [torch.zeros(1, 4, 8)],
        [0],
        [(0, row) for row in range(4)],
        torch.float32,
        mask_overrides={0: None},
        recurrent=False,
    )

    mask = kwargs["attention_mask"]
    assert tuple(mask.shape) == (1, 1, 4, 4)
    assert torch.isfinite(mask).all()


def _tiny_granite_hybrid(tmp: Path):
    from tokenizers import Tokenizer, models
    from transformers import GraniteMoeHybridConfig, PreTrainedTokenizerFast
    from transformers.models.granitemoehybrid.modeling_granitemoehybrid import (
        GraniteMoeHybridForCausalLM,
    )

    torch.manual_seed(0)
    config = GraniteMoeHybridConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        # One recurrent mixer and one attention block: the hybrid shape that
        # crashed.
        layer_types=["mamba", "attention"],
        num_local_experts=4,
        num_experts_per_tok=2,
        shared_intermediate_size=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    model = GraniteMoeHybridForCausalLM(config).eval()
    model.save_pretrained(tmp)

    vocab = {f"tok{index}": index for index in range(128)}
    vocab["[PAD]"] = 0
    vocab["[UNK]"] = 1
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")),
        pad_token="[PAD]",
        unk_token="[UNK]",
    ).save_pretrained(tmp)
    return model


def test_awq_batch_size_one_quantizes_a_mamba_hybrid(tmp_path):
    """End to end: the default batch size must not crash on a hybrid model."""

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import AWQConfig

    model = _tiny_granite_hybrid(tmp_path)
    layer_types = list(model.config.layer_types)
    assert "linear_attention" in layer_types and "full_attention" in layer_types

    calibration = [
        {"input_ids": list(range(3, 27)), "attention_mask": [1] * 24},
        {"input_ids": list(range(5, 25)), "attention_mask": [1] * 20},
        {"input_ids": list(range(7, 23)), "attention_mask": [1] * 16},
    ]

    quantized = GPTQModel.load(
        str(tmp_path),
        AWQConfig(bits=4, group_size=32, desc_act=False, device="cpu"),
        dtype=torch.bfloat16,
    )
    quantized.quantize(
        CalibrationConfig(
            dataset=calibration,
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.AWQ_TORCH,
    )

    # The recurrent mixer must have been packed, not silently skipped.
    names = {name for name, _module in quantized.model.named_modules()}
    assert any(name.endswith("mamba.in_proj") for name in names)
