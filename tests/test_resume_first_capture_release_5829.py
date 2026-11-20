# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5829: a resumed checkpoint releases the discarded capture.

``ModuleLooper._loop_impl`` kept the freshly captured first-layer replay
frontier in a frame local after handing it to the processor. Checkpoint resume
replaces the processor's cache with the durable restored one, so the local
became the only owner of the fresh hidden states and kept them alive for the
rest of the run. The local is now dropped at the handoff.
"""

from __future__ import annotations

import weakref
from unittest.mock import patch

import torch
import transformers

from gptqmodel import CalibrationConfig, CheckpointConfig, GPTQModel
from gptqmodel.looper import module_looper
from gptqmodel.looper.checkpoint import CheckpointExtension, CheckpointStopped
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
from gptqmodel.quantization.config import GPTQConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


ROW_COUNT = 64
ROW_LENGTH = 128
HIDDEN_SIZE = 64
FRONTIER_BYTES = ROW_COUNT * ROW_LENGTH * HIDDEN_SIZE * 2  # fp16 hidden states


def _tiny_model(model_dir):
    torch.manual_seed(836)
    transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=128,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=192,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    ).to(torch.float16).save_pretrained(model_dir)
    build_wordlevel_tokenizer().save_pretrained(model_dir)
    return model_dir


def _calibration():
    generator = torch.Generator().manual_seed(5829)
    rows = torch.randint(4, 128, (ROW_COUNT, ROW_LENGTH), generator=generator)
    return CalibrationConfig(
        dataset=[{"input_ids": row, "attention_mask": torch.ones_like(row)} for row in rows],
        min_length=1,
    )


def _quantize(source, workdir, label, calibration, checkpoint=None):
    model = GPTQModel.load(
        str(source),
        quantize_config=GPTQConfig(
            bits=4,
            group_size=32,
            device="cpu",
            offload_to_disk=True,
            offload_to_disk_path=str(workdir / label),
        ),
        dtype=torch.float16,
    )
    model.quantize(calibration, checkpoint=checkpoint)
    return model


def _interrupt_after_first_layer(workdir, source, calibration, checkpoint):
    boundary = CheckpointExtension.on_boundary

    def stop_after_first(extension, completed):
        if completed.step.index == 0:
            extension.request_stop()
        return boundary(extension, completed)

    with patch.object(CheckpointExtension, "on_boundary", stop_after_first):
        try:
            _quantize(source, workdir, "interrupted", calibration, checkpoint=checkpoint)
        except CheckpointStopped:
            return
    raise AssertionError("checkpoint stop did not trigger")


def _resume_with_release_probe(workdir, source, calibration, checkpoint):
    captured = []
    observations = []
    capture_inputs = StageInputsCapture.cache_inputs
    run_layer_stage = module_looper.run_layer_stage

    def capture(self, *args, **kwargs):
        cache = capture_inputs(self, *args, **kwargs)
        captured.extend(
            (weakref.ref(tensor), tensor.untyped_storage().nbytes())
            for batch in cache.layer_inputs
            for tensor in batch
        )
        return cache

    def live_bytes():
        return sum(size for ref, size in captured if ref() is not None)

    def measure(*args, **kwargs):
        # The resumed run selects its cursor here, after restore replaced the
        # processor's cache.
        assert kwargs["cursor"] == 1
        observations.append(live_bytes())
        result = run_layer_stage(*args, **kwargs)
        observations.append(live_bytes())
        return result

    with (
        patch.object(StageInputsCapture, "cache_inputs", capture),
        patch.object(module_looper, "run_layer_stage", measure),
    ):
        model = _quantize(source, workdir, "resumed", calibration, checkpoint=checkpoint)
    return model, captured, observations


def _state_dict_bytes(model):
    return {
        name: value.contiguous().reshape(-1).view(torch.uint8).clone()
        for name, value in model.model.state_dict().items()
    }


def test_resume_releases_discarded_first_capture(tmp_path):
    source = _tiny_model(tmp_path / "tiny")
    calibration = _calibration()
    checkpoint = CheckpointConfig(tmp_path / "checkpoint")

    _interrupt_after_first_layer(tmp_path, source, calibration, checkpoint)
    _model, captured, observations = _resume_with_release_probe(
        tmp_path, source, calibration, checkpoint
    )

    assert sum(size for _, size in captured) == FRONTIER_BYTES
    assert observations == [0, 0], (
        "the discarded first-layer capture must be released once restore "
        f"replaces it, got live bytes {observations}"
    )


def test_resume_matches_uninterrupted_control(tmp_path):
    source = _tiny_model(tmp_path / "tiny")
    calibration = _calibration()
    checkpoint = CheckpointConfig(tmp_path / "checkpoint")

    _interrupt_after_first_layer(tmp_path, source, calibration, checkpoint)
    resumed, _captured, _observations = _resume_with_release_probe(
        tmp_path, source, calibration, checkpoint
    )
    control = _quantize(source, tmp_path, "control", calibration)

    resumed_tensors = _state_dict_bytes(resumed)
    control_tensors = _state_dict_bytes(control)
    assert resumed_tensors.keys() == control_tensors.keys()
    for name in resumed_tensors:
        assert torch.equal(resumed_tensors[name], control_tensors[name]), (
            f"resume changed packed tensor {name}"
        )

    generator = torch.Generator().manual_seed(24)
    input_ids = torch.randint(4, 128, (1, 24), generator=generator)
    with torch.no_grad():
        assert torch.equal(resumed.model(input_ids=input_ids).logits,
                           control.model(input_ids=input_ids).logits)
