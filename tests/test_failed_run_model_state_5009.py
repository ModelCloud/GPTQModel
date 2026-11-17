# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cancelled ``quantize()`` must leave the in-session model usable.

Two exits skipped the repair the success path performs:

* ``_run_single_subset_pass`` guarded the forward itself but not the tail after
  it (``receive_layer_forward_context``, ``receive_layer_inputs`` and the
  ``forward_end`` subset callback). A ``StopMainLoop`` there unwound before the
  success-path ``_release_capture_hooks()``, so the in-flight layer kept its
  capture hooks and ``forward_hook_last`` and every later forward raised
  ``StopForward`` (#5009).
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.quantization import FORMAT, AWQConfig, CalibrationConfig, GPTQConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


VOCAB_SIZE = 128
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 128
NUM_LAYERS = 2
GROUP_SIZE = 32


def _tiny_model_dir(tmp_path) -> str:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    build_wordlevel_tokenizer().save_pretrained(tmp_path)
    return str(tmp_path)


def _calibration(rows: int = 8, length: int = 16):
    generator = torch.Generator().manual_seed(0)
    samples = []
    for _ in range(rows):
        input_ids = torch.randint(5, VOCAB_SIZE - 4, (1, length), generator=generator)
        samples.append(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        )
    return samples


def _load(model_dir: str, method: str, offload: bool) -> GPTQModel:
    if method == "gptq":
        quantize_config = GPTQConfig(
            bits=4,
            group_size=GROUP_SIZE,
            device="cpu",
            offload_to_disk=offload,
        )
    else:
        quantize_config = AWQConfig(
            bits=4,
            group_size=GROUP_SIZE,
            format=FORMAT.GEMM,
            device="cpu",
            offload_to_disk=offload,
        )
    return GPTQModel.load(model_dir, quantize_config=quantize_config, dtype=torch.float32)


def _run_expecting_failure(model: GPTQModel) -> BaseException | None:
    try:
        model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=2, min_length=1))
    except BaseException as exc:  # noqa: BLE001 - the stop signal is the point
        return exc
    return None


def _lingering_capture_hooks(model: torch.nn.Module):
    return [
        (name, type(module).__name__)
        for name, module in model.named_modules()
        if getattr(module, "forward_hook", None) is not None
    ]


def _stale_forward_hook_last(model: torch.nn.Module):
    return [
        name for name, module in model.named_modules() if getattr(module, "forward_hook_last", False)
    ]


def _input_ids(model: GPTQModel):
    return model.tokenizer("hello world", return_tensors="pt").input_ids


@pytest.mark.parametrize("mode", ["return_stop", "raise"])
def test_forward_end_stop_releases_capture_hooks(tmp_path, mode):
    """#5009: the post-forward tail must release the capture hooks too."""

    model = _load(_tiny_model_dir(tmp_path), method="gptq", offload=False)
    stopped_at: list[int] = []

    def callback(**kwargs):
        if kwargs["stage"] != "forward_end":
            return None
        stopped_at.append(kwargs["layer_idx"])
        if mode == "return_stop":
            return StopMainLoop
        raise RuntimeError("subset callback boom")

    model.subset_callback = callback

    error = _run_expecting_failure(model)
    assert stopped_at, "the run must reach the forward_end callback"
    if mode == "return_stop":
        assert isinstance(error, StopMainLoop)
    else:
        assert isinstance(error, RuntimeError)

    assert _lingering_capture_hooks(model.model) == []
    assert _stale_forward_hook_last(model.model) == []

    with torch.no_grad():
        outputs = model.model(_input_ids(model))
    assert outputs.logits.shape[-1] == VOCAB_SIZE
