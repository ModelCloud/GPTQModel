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
* ``ModuleLooper.loop()`` / ``WeightOnlyLooper.loop()`` called
  ``ensure_version_counters`` only on the success path, so the norms, biases and
  buffers materialized under ``inference_mode()`` stayed inference tensors and a
  plain grad-mode forward raised "Inference tensors cannot be saved for
  backward" (#5012).
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.quantization import FORMAT, AWQConfig, CalibrationConfig, GPTQConfig, RTNConfig
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
    elif method == "awq":
        quantize_config = AWQConfig(
            bits=4,
            group_size=GROUP_SIZE,
            format=FORMAT.GEMM,
            device="cpu",
            offload_to_disk=offload,
        )
    else:
        quantize_config = RTNConfig(
            bits=4,
            group_size=GROUP_SIZE,
            device="cpu",
            offload_to_disk=offload,
        )
    # fp16: the CPU AWQ torch kernel supports fp16/bf16 only, and an
    # unsupported dtype fails at pack time (asynchronously), which would
    # make the stop point nondeterministic.
    return GPTQModel.load(model_dir, quantize_config=quantize_config, dtype=torch.float16)


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


def _inference_tensors(model: torch.nn.Module):
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    return [
        name
        for name, tensor in tensors
        if tensor.device.type != "meta" and tensor.is_inference()
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


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("offload", [True, False])
def test_failed_run_repairs_inference_tensors(tmp_path, method, offload):
    """#5012: a cancelled run must return ordinary (non-inference) tensors."""

    model = _load(_tiny_model_dir(tmp_path), method=method, offload=offload)
    stopped_at: list[int] = []

    def callback(**kwargs):
        if kwargs["stage"] == "quant_complete" and kwargs["layer_idx"] == NUM_LAYERS - 1:
            stopped_at.append(kwargs["layer_idx"])
            return StopMainLoop
        return None

    model.subset_callback = callback

    error = _run_expecting_failure(model)
    assert stopped_at, "the run must reach the quant_complete callback"
    assert isinstance(error, StopMainLoop)

    assert _inference_tensors(model.model) == []

    # A plain forward in default grad mode is what a loss computation does; it
    # fails on an inference tensor that has to be saved for backward.
    outputs = model.model(_input_ids(model))
    assert outputs.logits.shape[-1] == VOCAB_SIZE


@pytest.mark.parametrize("offload", [True, False])
def test_failed_weight_only_run_repairs_inference_tensors(tmp_path, offload, monkeypatch):
    """#5012: the weight-only looper's failure branch needs the same repair."""

    from gptqmodel.looper.weight_only_processor import WeightOnlyProcessor

    model = _load(_tiny_model_dir(tmp_path), method="rtn", offload=offload)
    original = WeightOnlyProcessor.quantize_module

    def boom(self, module, **kwargs):
        if getattr(module, "full_name", "").endswith("layers.1.self_attn.q_proj"):
            raise RuntimeError("injected quantization failure")
        return original(self, module, **kwargs)

    monkeypatch.setattr(WeightOnlyProcessor, "quantize_module", boom)

    with pytest.raises(RuntimeError, match="injected quantization failure"):
        model.quantize(None)

    assert _inference_tensors(model.model) == []

    # The weight-only looper does not yet move the in-flight layer back to its
    # home device on failure (#5011), so on an accelerator host the model can be
    # mixed-device here for a reason unrelated to this contract. The repair
    # above is still asserted; the forward is only checked when the layer was
    # restored (always true on a CPU run).
    if len({tensor.device.type for tensor in model.model.parameters()}) == 1:
        outputs = model.model(_input_ids(model))
        assert outputs.logits.shape[-1] == VOCAB_SIZE
