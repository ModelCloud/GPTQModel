# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""EXL3 cancellations must leave a runnable model (#5083).

``EXL3Processor._stage_runtime_weight`` swaps each packed projection's dense
weight for a META placeholder and rebuilds it only for the next layer replay.
A stop at ``quant_complete`` (the documented ``StopMainLoop`` point), an
exception or Ctrl-C in that window therefore left the in-flight layer's
projections on META, and every later forward raised
``RuntimeError: Tensor on device cpu is not on the expected device meta!``.
``offload_to_disk=True`` escaped only because its layer shells are restored
from the checkpoint.

The exit recovery now publishes each deferred projection's packed module before
the layer is rehomed and before the remaining-meta verdict. These tests drive a
real EXL3 run on a two-layer model, stop it in the deferring windows, and assert
the contract in ``docs/cancelled-run-contract.md``: no META parameter, no
capture hook, no inference tensor, and an ordinary grad-mode forward.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.looper.module_looper import ModuleLooper, StopMainLoop
from gptqmodel.quantization import AutoDtypeCast
from gptqmodel.quantization.config import EXL3Config
from tests.tiny_remote_models import build_wordlevel_tokenizer


VOCAB_SIZE = 128
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 256
NUM_LAYERS = 2
INJECTED_FAULT = "injected exl3 cancellation fault"


def _exl3_gpu_reason() -> str | None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return "EXL3 quantization needs an NVIDIA CUDA device"
    return None


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.slow,
    pytest.mark.skipif(_exl3_gpu_reason() is not None, reason=_exl3_gpu_reason() or ""),
]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> str:
    destination = tmp_path_factory.mktemp("exl3-cancelled-5083") / "model-2l"
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(5083)
    LlamaForCausalLM(config).save_pretrained(destination)
    build_wordlevel_tokenizer().save_pretrained(destination)
    return str(destination)


def _calibration(rows: int = 8, length: int = 32):
    generator = torch.Generator().manual_seed(5083)
    return [
        {
            "input_ids": torch.randint(
                5, VOCAB_SIZE - 4, (1, length), generator=generator
            ),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        for _ in range(rows)
    ]


def _load(model_dir: str, *, offload: bool) -> GPTQModel:
    return GPTQModel.load(
        model_dir,
        quantize_config=EXL3Config(
            bits=4,
            device="cuda:0",
            offload_to_disk=offload,
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        dtype=torch.float16,
    )


def _quantize(model: GPTQModel) -> None:
    model.quantize(
        CalibrationConfig(dataset=_calibration(), batch_size=2, min_length=1)
    )


def _capture_recovery(monkeypatch):
    created = []
    original = ModuleLooper._make_run_exit_recovery

    def capture(self):
        recovery = original(self)
        created.append(recovery)
        return recovery

    monkeypatch.setattr(ModuleLooper, "_make_run_exit_recovery", capture)
    return created


def _stop_at_subset(model: GPTQModel, fired, *, layer: int, member: str, error: bool):
    def callback(**kwargs):
        if kwargs["stage"] != "quant_complete" or kwargs["layer_idx"] != layer:
            return None
        names = [str(name) for name in kwargs.get("module_names") or ()]
        if not any(member in name for name in names):
            return None
        fired.append((kwargs["stage"], layer, member))
        if error:
            raise RuntimeError(INJECTED_FAULT)
        return StopMainLoop

    model.subset_callback = callback


def _meta_parameters(model) -> list[str]:
    return [name for name, param in model.named_parameters() if param.is_meta]


def _lingering_hooks(model) -> list[str]:
    return [
        name
        for name, submodule in model.named_modules()
        if getattr(submodule, "forward_hook", None) is not None
        or getattr(submodule, "forward_hook_last", False)
    ]


def _inference_tensors(model) -> list[str]:
    return [
        name
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if tensor.device.type != "meta" and tensor.is_inference()
    ]


def _assert_runnable(model: GPTQModel) -> None:
    model.model.eval()
    input_ids = model.tokenizer("the quick brown fox", return_tensors="pt").input_ids
    outputs = model.model(input_ids, labels=input_ids)
    assert outputs.logits.shape[-1] == VOCAB_SIZE
    assert torch.isfinite(outputs.logits).all()


def _assert_deferred_modules_are_packed(model: GPTQModel) -> None:
    """The published modules must be packed kernels, not retained dense copies."""

    layer = model.model.model.layers[1]
    packed = [
        module
        for module in layer.modules()
        if getattr(module, "QUANT_TYPE", None) == "exl3"
    ]
    assert packed, "the in-flight layer kept no packed EXL3 projection"
    for module in packed:
        assert isinstance(module.trellis, torch.Tensor)
        assert module.trellis.device.type != "meta"
        assert "weight" not in module._parameters


@pytest.mark.parametrize("offload", [False, True], ids=["no-offload", "offload"])
@pytest.mark.parametrize("error", [False, True], ids=["stop", "raise"])
def test_quant_complete_stop_publishes_the_in_flight_layer(
    tiny_model_dir, offload, error, monkeypatch
):
    model = _load(tiny_model_dir, offload=offload)
    recoveries = _capture_recovery(monkeypatch)
    fired: list = []
    _stop_at_subset(
        model, fired, layer=1, member="q_proj", error=error
    )

    if error:
        with pytest.raises(RuntimeError, match=INJECTED_FAULT):
            _quantize(model)
    else:
        with pytest.raises(StopMainLoop):
            _quantize(model)

    assert fired, "the injected cancellation did not fire"
    recovery = recoveries[-1]
    assert recovery.completed is False
    assert recovery.failures == []
    assert _meta_parameters(model.model) == []
    assert _lingering_hooks(model.model) == []
    assert _inference_tensors(model.model) == []
    _assert_deferred_modules_are_packed(model)
    _assert_runnable(model)


@pytest.mark.parametrize("offload", [False, True], ids=["no-offload", "offload"])
def test_quant_complete_stop_in_the_last_layer_publishes_its_projections(
    tiny_model_dir, offload, monkeypatch
):
    """The last layer has no propagation replay to rebuild its own weights.

    ``prepare_layer_post_quantize`` replaces the final layer's remaining META
    placeholders only after every subset of that layer has been packed, so a
    stop at the last layer's ``quant_complete`` had no later replay to publish
    anything (#5083).
    """

    model = _load(tiny_model_dir, offload=offload)
    recoveries = _capture_recovery(monkeypatch)
    fired: list = []
    _stop_at_subset(
        model, fired, layer=NUM_LAYERS - 1, member="gate_proj", error=False
    )

    with pytest.raises(StopMainLoop):
        _quantize(model)

    assert fired, "the injected cancellation did not fire"
    recovery = recoveries[-1]
    assert recovery.failures == []
    assert _meta_parameters(model.model) == []
    assert _lingering_hooks(model.model) == []
    assert _inference_tensors(model.model) == []
    _assert_runnable(model)


def test_completed_run_still_publishes_every_projection(tiny_model_dir):
    """Control: the recovery hook must not disturb a run that replayed fully."""

    model = _load(tiny_model_dir, offload=False)
    _quantize(model)

    assert _meta_parameters(model.model) == []
    assert _lingering_hooks(model.model) == []
    _assert_runnable(model)
