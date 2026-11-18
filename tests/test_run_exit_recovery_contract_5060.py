# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""One run-exit recovery contract for both quantization loopers (#5060).

The calibrated (``ModuleLooper``) and weight-only (``WeightOnlyLooper``) paths
used to order their failure repairs in separate exception handlers. This suite
drives tiny real runs through the same scenarios and asserts the caller
contract in ``docs/cancelled-run-contract.md``:

* a fault during the next layer and a fault during the final restore both
  re-raise the original error;
* the in-session model still runs an ordinary grad-enabled forward;
* completed, in-flight and unreached layers are usable, with no capture hooks
  or inference tensors left behind;
* live disk shells keep their backing until release;
* a secondary repair failure is recorded but never replaces the primary error;
* a successful run publishes only after the restore.

The file is new rather than extending ``tests/test_cancelled_quantize_*``
because those files pin one looper each; this fixture is the shared
conformance matrix the issue asks for.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.looper.module_looper import ModuleLooper, StopMainLoop
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.quantization import CalibrationConfig, GPTQConfig, RTNConfig
from gptqmodel.quantization.config import AutoDtypeCast
from gptqmodel.utils.offload import has_disk_offloaded_modules, is_meta_module
from tests.tiny_remote_models import build_wordlevel_tokenizer


VOCAB_SIZE = 128
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
NUM_LAYERS = 3
GROUP_SIZE = 32
MID_RUN_FAULT = "injected next-layer failure"
FINAL_RESTORE_FAULT = "injected final-restore failure"
SECONDARY_FAULT = "injected secondary repair failure"


def _tiny_model_dir(tmp_path_factory) -> str:
    from transformers import LlamaConfig, LlamaForCausalLM

    destination = tmp_path_factory.mktemp("run-exit-5060") / "model-3l"
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
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(destination)
    build_wordlevel_tokenizer().save_pretrained(destination)
    return str(destination)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> str:
    return _tiny_model_dir(tmp_path_factory)


def _calibration(rows: int = 8, length: int = 16):
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "input_ids": torch.randint(
                5, VOCAB_SIZE - 4, (1, length), generator=generator
            ),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        for _ in range(rows)
    ]


def _load(model_dir: str, looper: str, *, offload: bool) -> GPTQModel:
    if looper == "gptq":
        quantize_config = GPTQConfig(
            bits=4,
            group_size=GROUP_SIZE,
            device="cpu",
            offload_to_disk=offload,
            auto_dtype_cast=AutoDtypeCast.OFF,
        )
    else:
        quantize_config = RTNConfig(
            bits=4,
            group_size=GROUP_SIZE,
            device="cpu",
            offload_to_disk=offload,
            auto_dtype_cast=AutoDtypeCast.OFF,
        )
    return GPTQModel.load(
        model_dir, quantize_config=quantize_config, dtype=torch.float16
    )


def _quantize(model: GPTQModel) -> None:
    model.quantize(
        CalibrationConfig(dataset=_calibration(), batch_size=2, min_length=1)
    )


def _capture_recovery(monkeypatch, looper_cls) -> List:
    """Record the run-scoped recovery owner so its state can be asserted."""

    created: List = []
    original = looper_cls._make_run_exit_recovery

    def capture(self):
        recovery = original(self)
        created.append(recovery)
        return recovery

    monkeypatch.setattr(looper_cls, "_make_run_exit_recovery", capture)
    return created


def _inject_mid_run_fault(model: GPTQModel, looper: str, monkeypatch, faults: List[str]):
    if looper == "gptq":
        def callback(**kwargs):
            if kwargs["stage"] == "quant_start" and kwargs["layer_idx"] == 1:
                faults.append(f"layer:{kwargs['layer_idx']}")
                raise RuntimeError(MID_RUN_FAULT)
            return None

        model.subset_callback = callback
        return

    from gptqmodel.looper.weight_only_processor import WeightOnlyProcessor

    original = WeightOnlyProcessor.quantize_module

    def boom(self, module, *args, **kwargs):
        if ".layers.1." in getattr(module, "full_name", ""):
            faults.append(getattr(module, "full_name", "?"))
            raise RuntimeError(MID_RUN_FAULT)
        return original(self, module, *args, **kwargs)

    monkeypatch.setattr(WeightOnlyProcessor, "quantize_module", boom)


def _inject_final_restore_fault(looper: str, monkeypatch, faults: List[str]) -> None:
    """Fail once in the success tail; the abort pass must then succeed."""

    if looper == "gptq":
        cls, name = ModuleLooper, "_materialize_remaining_meta_modules"

        def wrapper(self):
            if not faults:
                faults.append(name)
                raise RuntimeError(FINAL_RESTORE_FAULT)
            return original(self)

    else:
        cls, name = WeightOnlyLooper, "_materialize_meta_base_modules"

        def wrapper(self, *, strict):
            if not faults:
                faults.append(name)
                raise RuntimeError(FINAL_RESTORE_FAULT)
            return original(self, strict=strict)

    original = getattr(cls, name)
    monkeypatch.setattr(cls, name, wrapper)


def _inject_secondary_repair_fault(looper: str, monkeypatch) -> None:
    cls = ModuleLooper if looper == "gptq" else WeightOnlyLooper
    original = cls._restore_offloaded_base_modules

    def wrapper(self, *, strict):
        if not strict:
            raise RuntimeError(SECONDARY_FAULT)
        return original(self, strict=strict)

    monkeypatch.setattr(cls, "_restore_offloaded_base_modules", wrapper)


def _non_cpu_tensors(model: torch.nn.Module) -> List[str]:
    return [
        name
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if tensor.device.type not in {"cpu", "meta"}
    ]


def _inference_tensors(model: torch.nn.Module) -> List[str]:
    return [
        name
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]
        if tensor.device.type != "meta" and tensor.is_inference()
    ]


def _lingering_hooks(model: torch.nn.Module) -> List[str]:
    return [
        name
        for name, module in model.named_modules()
        if getattr(module, "forward_hook", None) is not None
        or getattr(module, "forward_hook_last", False)
    ]


def _assert_grad_forward(model: GPTQModel) -> None:
    """An ordinary default-grad forward must work after recovery."""

    model.model.eval()
    input_ids = model.tokenizer("hello world", return_tensors="pt").input_ids
    outputs = model.model(input_ids)
    assert outputs.logits.shape[-1] == VOCAB_SIZE
    assert torch.isfinite(outputs.logits).all()


def _assert_layers_usable(model: GPTQModel) -> None:
    layers = model.model.model.layers
    assert len(layers) == NUM_LAYERS
    # The unreached trailing layer must be materialized, not a meta shell.
    assert not is_meta_module(layers[NUM_LAYERS - 1]), "unreached layer is still meta"
    # The in-flight layer must be back on the model's home device.
    assert _non_cpu_tensors(layers[1]) == []


def _assert_completed_layer_usable(model: GPTQModel) -> None:
    layer = model.model.model.layers[0]
    if any(getattr(module, "QUANT_TYPE", None) is not None for module in layer.modules()):
        return
    assert any(
        is_meta_module(module) and getattr(module, "_hf_hook", None) is not None
        for module in layer.modules()
    ), "completed layer is neither quantized nor a live disk shell"


def _snapshot_completed_layer(model: GPTQModel) -> torch.Tensor:
    for layer_index in (0, NUM_LAYERS - 1):
        layer = model.model.model.layers[layer_index]
        for module in layer.modules():
            weight = getattr(module, "qweight", None)
            if isinstance(weight, torch.Tensor) and weight.device.type == "cpu":
                return weight.detach().clone()
        norm = getattr(layer, "input_layernorm", None)
        if norm is not None and norm.weight.device.type == "cpu":
            return norm.weight.detach().clone()
    raise AssertionError("no CPU tensor to snapshot after recovery")


def _assert_disk_shells_retain_backing(model: GPTQModel) -> None:
    if not has_disk_offloaded_modules(model.model):
        return
    run_root = getattr(model, "_private_offload_run_root", None)
    assert run_root is not None and Path(run_root).is_dir(), (
        "live disk shells lost their per-run offload root"
    )
    offloaded = [
        name
        for name, module in model.model.named_modules()
        if is_meta_module(module) and getattr(module, "_hf_hook", None) is not None
    ]
    assert offloaded
    for name in offloaded:
        assert (Path(run_root) / name / "module.safetensors").is_file(), (
            f"offload bundle for `{name}` was removed while its shell is live"
        )


@pytest.fixture(params=["gptq", "rtn"])
def looper(request) -> str:
    return request.param


@pytest.fixture(params=[True, False], ids=["offload", "no-offload"])
def offload(request) -> bool:
    return request.param


def test_mid_run_fault_recovers_and_preserves_the_original_error(
    tiny_model_dir, looper, offload, monkeypatch
):
    model = _load(tiny_model_dir, looper, offload=offload)
    recoveries = _capture_recovery(
        monkeypatch, ModuleLooper if looper == "gptq" else WeightOnlyLooper
    )
    faults: List[str] = []
    _inject_mid_run_fault(model, looper, monkeypatch, faults)

    with pytest.raises(RuntimeError, match=MID_RUN_FAULT):
        _quantize(model)

    assert faults, "the injected next-layer fault did not fire"
    recovery = recoveries[-1]
    assert recovery.failures == []
    assert recovery.completed is False
    assert _lingering_hooks(model.model) == []
    assert _inference_tensors(model.model) == []
    assert _non_cpu_tensors(model.model) == []
    _assert_layers_usable(model)
    _assert_completed_layer_usable(model)
    _assert_disk_shells_retain_backing(model)
    _assert_grad_forward(model)

    # The fence ran before teardown: a completed layer must not change after
    # the run has returned.
    snapshot = _snapshot_completed_layer(model)
    time.sleep(0.2)
    current = _snapshot_completed_layer(model)
    assert torch.equal(snapshot, current), "a late worker mutated a completed layer"


def test_final_restore_fault_recovers_and_preserves_the_original_error(
    tiny_model_dir, looper, offload, monkeypatch
):
    model = _load(tiny_model_dir, looper, offload=offload)
    recoveries = _capture_recovery(
        monkeypatch, ModuleLooper if looper == "gptq" else WeightOnlyLooper
    )
    faults: List[str] = []
    _inject_final_restore_fault(looper, monkeypatch, faults)

    with pytest.raises(RuntimeError, match=FINAL_RESTORE_FAULT):
        _quantize(model)

    assert faults, "the injected final-restore fault did not fire"
    recovery = recoveries[-1]
    assert recovery.completed is False
    assert _lingering_hooks(model.model) == []
    assert _inference_tensors(model.model) == []
    assert _non_cpu_tensors(model.model) == []
    _assert_layers_usable(model)
    _assert_grad_forward(model)


def test_secondary_repair_failure_is_recorded_not_primary(
    tiny_model_dir, looper, monkeypatch
):
    model = _load(tiny_model_dir, looper, offload=False)
    recoveries = _capture_recovery(
        monkeypatch, ModuleLooper if looper == "gptq" else WeightOnlyLooper
    )
    faults: List[str] = []
    _inject_mid_run_fault(model, looper, monkeypatch, faults)
    _inject_secondary_repair_fault(looper, monkeypatch)

    with pytest.raises(RuntimeError, match=MID_RUN_FAULT):
        _quantize(model)

    assert faults
    recovery = recoveries[-1]
    labels = [label for label, _exc in recovery.failures]
    assert "restore offloaded base modules" in labels, recovery.failures
    assert recovery.recovered is False
    # The secondary failure is contained; the original error is the one raised.


def test_post_forward_stop_gptq_recovers(tiny_model_dir, offload, monkeypatch):
    model = _load(tiny_model_dir, "gptq", offload=offload)
    recoveries = _capture_recovery(monkeypatch, ModuleLooper)
    stopped_at: List[int] = []

    def callback(**kwargs):
        if kwargs["stage"] == "forward_end" and kwargs["layer_idx"] == 0:
            stopped_at.append(kwargs["layer_idx"])
            return StopMainLoop
        return None

    model.subset_callback = callback

    with pytest.raises(StopMainLoop):
        _quantize(model)

    assert stopped_at, "the post-forward stop did not fire"
    recovery = recoveries[-1]
    assert recovery.failures == []
    assert _lingering_hooks(model.model) == []
    assert _inference_tensors(model.model) == []
    assert _non_cpu_tensors(model.model) == []
    _assert_layers_usable(model)
    _assert_grad_forward(model)


def test_successful_run_control(tiny_model_dir, looper, offload, monkeypatch):
    model = _load(tiny_model_dir, looper, offload=offload)
    recoveries = _capture_recovery(
        monkeypatch, ModuleLooper if looper == "gptq" else WeightOnlyLooper
    )

    _quantize(model)

    recovery = recoveries[-1]
    assert recovery.completed is True
    assert recovery.recovered is True
    assert recovery.failures == []
    assert _lingering_hooks(model.model) == []
    assert _inference_tensors(model.model) == []
    assert _non_cpu_tensors(model.model) == []
    assert not has_disk_offloaded_modules(model.model), (
        "a successful run must restore the disk-offloaded layers"
    )
    _assert_grad_forward(model)
