# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Paro replay must use the owning module-tree branch's masks (#4226).

The fixture in ``tests/module_tree/test_parallel_branch_calibration.py``
already proves that the *kwargs* replayed for each layer come from that layer's
branch (#4150/#4153).  This module covers the remaining replay context:
``position_ids`` and ``attention_masks`` were read from the top-level
``InputCache``, which always describes the first (text) branch.
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import threading

import pytest
import torch

import tests.module_tree.test_parallel_branch_calibration as branch_fixture
from gptqmodel import CalibrationConfig
from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.paro_processor import ParoProcessor
from gptqmodel.quantization.config import ParoConfig


pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _quantize_with_mask_recorders(tmp_path, monkeypatch):
    vision_masks = []

    def vision_forward(self, hidden_states, attention_mask=None, cu_seqlens=None, **kwargs):
        del cu_seqlens, kwargs
        vision_masks.append(
            None if attention_mask is None else tuple(attention_mask.shape)
        )
        return hidden_states + self.mlp(hidden_states + self.self_attn(hidden_states))

    # The stock vision block has no ``attention_mask`` parameter, so the replay
    # kwargs builder would filter the leaked mask out. Accept it explicitly to
    # observe what the replay actually passes, as in the issue's repro.
    monkeypatch.setattr(branch_fixture._VisionBlock, "forward", vision_forward)

    tokenizer = branch_fixture._build_fixture(
        tmp_path, text_hidden=64, vision_hidden=32
    )
    qcfg = ParoConfig(
        bits=4,
        group_size=32,
        device="cpu",
        opt_scope="layer",
        opt_rotation_epochs=1,
        opt_finetune_epochs=1,
        opt_train_samples=4,
        opt_validation_samples=1,
        opt_batch_size=1,
    )
    model = branch_fixture.ParallelCalibrationQModel.from_pretrained(
        str(tmp_path), quantize_config=qcfg, module_tree="auto"
    )

    text_masks = []

    def record_text_mask(module, args, kwargs):
        del module, args
        mask = kwargs.get("attention_mask")
        text_masks.append(None if mask is None else tuple(mask.shape))

    handles = [
        layer.register_forward_pre_hook(record_text_mask, with_kwargs=True)
        for layer in model.model.model.layers
    ]
    try:
        model.quantize(
            CalibrationConfig(
                dataset=branch_fixture._calibration(tokenizer, include_pixel_values=True),
                batch_size=1,
                min_length=1,
            ),
        )
    finally:
        for handle in handles:
            handle.remove()

    return vision_masks, text_masks


def test_vision_replay_never_receives_the_text_attention_mask(tmp_path, monkeypatch):
    vision_masks, text_masks = _quantize_with_mask_recorders(tmp_path, monkeypatch)

    assert vision_masks, "vision branch layers were never replayed"
    leaked = [shape for shape in vision_masks if shape is not None]
    assert leaked == [], (
        f"{len(leaked)}/{len(vision_masks)} vision-layer calls received a "
        f"non-None attention_mask, e.g. {leaked[:3]}"
    )
    # The text branch keeps its own mask; the fix must not drop masks globally.
    assert any(shape is not None for shape in text_masks)


def _input_cache(layer_inputs, *, masks, positions, kwargs=None):
    return InputCache(
        layer_inputs=layer_inputs,
        layer_input_kwargs=kwargs or [{} for _ in layer_inputs],
        position_ids=positions,
        attention_masks=masks,
    )


def test_processor_records_the_owning_branch_replay_context():
    text_inputs = [[torch.randn(1, 4)]]
    vision_inputs = [[torch.randn(1, 4)]]
    text_cache = _input_cache(
        text_inputs,
        masks=[torch.ones(1, 5, dtype=torch.long)],
        positions=[torch.arange(5).unsqueeze(0)],
    )
    vision_cache = _input_cache(
        vision_inputs,
        masks=[None],
        positions=[None],
        kwargs=[{"cu_seqlens": torch.tensor([[0, 4]])}],
    )
    top_level = _input_cache(
        text_inputs,
        masks=[torch.ones(1, 5, dtype=torch.long)],
        positions=[torch.arange(5).unsqueeze(0)],
    )
    top_level.branch_first_layer_inputs = {0: text_cache, 2: vision_cache}

    processor = object.__new__(ParoProcessor)
    processor.inputs_cache = top_level
    processor._layer_states = {}
    processor._layer_states_lock = threading.Lock()

    processor.receive_layer_forward_context(
        layer_index=3,
        layer_inputs=vision_inputs,
        layer_input_kwargs=[{"cu_seqlens": torch.tensor([[0, 4]])}],
        layer_outputs=[[torch.randn(1, 4)]],
    )
    processor.receive_layer_forward_context(
        layer_index=0,
        layer_inputs=text_inputs,
        layer_input_kwargs=[{}],
        layer_outputs=[[torch.randn(1, 4)]],
    )

    # Layer 3 belongs to the vision branch (first index 2): no mask, no
    # position ids, even though the top-level cache has the text branch's.
    assert processor._layer_states[3].attention_masks == [None]
    assert processor._layer_states[3].position_ids == [None]
    # Layer 0 belongs to the text branch and keeps its own mask/positions.
    assert torch.equal(
        processor._layer_states[0].attention_masks[0],
        torch.ones(1, 5, dtype=torch.long),
    )
    assert torch.equal(
        processor._layer_states[0].position_ids[0], torch.arange(5).unsqueeze(0)
    )
