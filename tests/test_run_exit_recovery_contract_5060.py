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
* a successful run publishes only after the restore;
* a fault inside packing leaves the float projection installed, so the model
  stays runnable instead of holding a kernel without packed buffers (#5089).

The file is new rather than extending ``tests/test_cancelled_quantize_*``
because those files pin one looper each; this fixture is the shared
conformance matrix the issue asks for.

The two materialization-interrupt nodes at the end cover #5088: a Ctrl-C
between allocating a lazy parameter's storage and copying the checkpoint
values into it must not publish ``torch.empty`` weights.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path
from typing import List

import pytest
import torch
from safetensors.torch import load_file

from gptqmodel import GPTQModel
from gptqmodel.looper.module_looper import ModuleLooper, StopMainLoop
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization import CalibrationConfig, GPTQConfig, RTNConfig
from gptqmodel.quantization.config import AutoDtypeCast
from gptqmodel.utils.offload import has_disk_offloaded_modules, is_meta_module
from gptqmodel.utils.structure import LazyTurtle
from tests.tiny_remote_models import build_wordlevel_tokenizer


VOCAB_SIZE = 128
HIDDEN_SIZE = 32
INTERMEDIATE_SIZE = 64
NUM_LAYERS = 3
GROUP_SIZE = 32
MID_RUN_FAULT = "injected next-layer failure"
PACK_FAULT = "injected packing failure"
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


PACKED_BUFFERS = ("qweight", "qzeros", "scales", "g_idx")


def _unpacked_quant_modules(model: GPTQModel) -> List[str]:
    """Installed quant kernels that never received their packed buffers."""

    broken: List[str] = []
    for name, module in model.model.named_modules():
        if not hasattr(module, "QUANT_TYPE"):
            continue
        missing = [buffer for buffer in PACKED_BUFFERS if not hasattr(module, buffer)]
        if missing:
            broken.append(f"{name} ({type(module).__name__}): {missing}")
    return broken


def _inject_pack_fault(looper: str, monkeypatch, *, pack_call: int = 1) -> List[str]:
    """Raise once inside a ``pack_module`` call, as a Ctrl-C would.

    Before #5089 finalization installed the empty kernel first and packed it
    afterwards, so the fault left a kernel without ``qweight``/``qzeros``/
    ``scales``/``g_idx`` in the model and every later forward raised
    ``AttributeError``. The injection point is identical either way, so the
    test fails for the intended reason on the parent commit.

    The default targets the first module of the run, which is the position the
    report's deterministic repro interrupts at (inside layer 0's first pack).
    """

    module_name = (
        "gptqmodel.looper.gptq_processor" if looper == "gptq"
        else "gptqmodel.looper.weight_only_processor"
    )
    module = importlib.import_module(module_name)
    original = module.pack_module
    faults: List[str] = []
    calls = {"n": 0}

    def boom(name, *args, **kwargs):
        calls["n"] += 1
        if not faults and calls["n"] == pack_call:
            faults.append(name)
            raise KeyboardInterrupt(PACK_FAULT)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(module, "pack_module", boom)
    return faults


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


def _checkpoint_tensors(model_dir: str) -> dict:
    shard = next(Path(model_dir).glob("*.safetensors"))
    return load_file(str(shard))


def _assert_param_matches_checkpoint(param: torch.Tensor, checkpoint: torch.Tensor) -> None:
    expected = checkpoint.to(dtype=param.dtype, device="cpu")
    assert torch.equal(param.detach().cpu(), expected), (
        "recovered parameter does not match the checkpoint tensor"
    )


def _interrupt_first_materialization_copy(monkeypatch):
    """Arm a KeyboardInterrupt at the first copy inside checkpoint materialization.

    The defect is not the interrupt itself: it is publishing freshly allocated
    storage on the module before the checkpoint values are copied into it. The
    hook therefore fires only inside ``_copy_checkpoint_tensors_*`` so the test
    lands in the materialization window rather than on an unrelated copy.
    """

    state = {"armed": False, "materializing": False, "fired": False}
    original_copy = torch.Tensor.copy_
    original_materialize = LazyTurtle._copy_checkpoint_tensors_into_submodule

    def copy_(self, *args, **kwargs):
        if state["materializing"]:
            state["materializing"] = False
            state["fired"] = True
            raise KeyboardInterrupt()
        return original_copy(self, *args, **kwargs)

    def materialize(self, **kwargs):
        if state["armed"]:
            state["armed"] = False
            state["materializing"] = True
        try:
            return original_materialize(self, **kwargs)
        finally:
            state["materializing"] = False

    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    monkeypatch.setattr(
        LazyTurtle, "_copy_checkpoint_tensors_into_submodule", materialize
    )
    return state


def _assert_no_meta_or_nonfinite_weights(model: GPTQModel) -> None:
    meta = [name for name, param in model.model.named_parameters() if param.is_meta]
    assert meta == [], f"parameters still on meta after recovery: {meta}"
    nonfinite = [
        name
        for name, param in model.model.named_parameters()
        if param.is_floating_point() and not torch.isfinite(param).all()
    ]
    assert nonfinite == [], f"non-finite parameters after recovery: {nonfinite}"


def _assert_checkpoint_params_match(model: GPTQModel, checkpoint: dict) -> None:
    mismatched = [
        name
        for name, param in model.model.named_parameters()
        if name in checkpoint
        and not torch.equal(
            param.detach().cpu(),
            checkpoint[name].to(dtype=param.dtype, device="cpu"),
        )
    ]
    assert mismatched == [], (
        f"recovered parameters do not match the checkpoint: {mismatched}"
    )


def test_tail_materialization_interrupt_does_not_publish_empty_weights(
    tiny_model_dir, monkeypatch
):
    """A Ctrl-C in the post-loop tail must not leave torch.empty base weights (#5088).

    The weight-only loop finishes every layer before the success tail fills the
    base modules. Interrupting the first checkpoint copy there used to install
    uninitialized storage that run-exit recovery treated as loaded because it
    only tests ``is_meta``.
    """

    model = _load(tiny_model_dir, "rtn", offload=True)
    state = _interrupt_first_materialization_copy(monkeypatch)
    original_impl = WeightOnlyLooper._loop_impl

    def arm_after_body(self, **kwargs):
        result = original_impl(self, **kwargs)
        state["armed"] = True
        return result

    monkeypatch.setattr(WeightOnlyLooper, "_loop_impl", arm_after_body)

    with pytest.raises(KeyboardInterrupt):
        _quantize(model)

    assert state["fired"], "the injected materialization interrupt did not fire"
    _assert_no_meta_or_nonfinite_weights(model)

    # Every layer was quantized before the tail, so recovery only had to fill
    # the base modules; they must hold the checkpoint values, not torch.empty.
    checkpoint = _checkpoint_tensors(tiny_model_dir)
    _assert_param_matches_checkpoint(
        model.model.model.embed_tokens.weight, checkpoint["model.embed_tokens.weight"]
    )
    _assert_param_matches_checkpoint(
        model.model.model.norm.weight, checkpoint["model.norm.weight"]
    )
    _assert_grad_forward(model)


def test_capture_materialization_interrupt_does_not_publish_empty_weights(
    tiny_model_dir, monkeypatch
):
    """A Ctrl-C while materializing the first capture layer must not leave garbage (#5088).

    GPTQ materializes the first traversed layer before its early-stop hook.
    Interrupting the first checkpoint copy there left a partially materialized
    dense layer whose ``torch.empty`` weights the recovery considered loaded.
    The run stops before any layer is quantized, so every weight must still
    match the checkpoint afterwards.
    """

    model = _load(tiny_model_dir, "gptq", offload=True)
    state = _interrupt_first_materialization_copy(monkeypatch)
    original_prepare = BaseQModel.prepare_input_capture_layer

    def arm_before_capture(self, layer, **kwargs):
        state["armed"] = True
        return original_prepare(self, layer, **kwargs)

    monkeypatch.setattr(BaseQModel, "prepare_input_capture_layer", arm_before_capture)

    with pytest.raises(KeyboardInterrupt):
        _quantize(model)

    assert state["fired"], "the injected materialization interrupt did not fire"
    _assert_no_meta_or_nonfinite_weights(model)
    checkpoint = _checkpoint_tensors(tiny_model_dir)
    # No layer was quantized before the capture interrupt, so the recovered
    # model must be the untouched base model: every checkpoint-owned weight
    # (including the interrupted ``q_proj``) still matches the source file.
    _assert_checkpoint_params_match(model, checkpoint)
    _assert_grad_forward(model)


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


def test_pack_fault_leaves_no_half_built_kernel(
    tiny_model_dir, looper, offload, monkeypatch
):
    """A Ctrl-C inside packing must not publish an unusable quant module (#5089).

    The cancelled-run contract promises the in-session model stays runnable.
    Finalization covers most of a run's wall time in packing, and any kernel
    installed before its pack call returned owned no packed buffers, so the
    next forward raised ``AttributeError: ... has no attribute 'qzeros'``.
    """

    model = _load(tiny_model_dir, looper, offload=offload)
    recoveries = _capture_recovery(
        monkeypatch, ModuleLooper if looper == "gptq" else WeightOnlyLooper
    )
    faults = _inject_pack_fault(looper, monkeypatch)

    with pytest.raises(KeyboardInterrupt, match=PACK_FAULT):
        _quantize(model)

    assert faults, "the injected packing fault did not fire"
    assert recoveries[-1].completed is False
    assert _unpacked_quant_modules(model) == [], (
        "a cancelled run left a quant kernel installed without its packed buffers"
    )
    assert _lingering_hooks(model.model) == []
    assert _non_cpu_tensors(model.model) == []
    _assert_layers_usable(model)

    # The interrupted projection kept its float weight, so the ordinary forward
    # (not just an inference-mode probe) still runs.
    _assert_grad_forward(model)


