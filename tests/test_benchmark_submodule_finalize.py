import os

# force pytest runs to target GPU 7 (becomes cuda:0 after masking)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper import gptq_processor as gptq_processor_module
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.config import QuantizeConfig


def _dummy_prepare_dataset(*, calibration_dataset, calibration_dataset_concat_size, calibration_dataset_sort, batch_size):
    return calibration_dataset


class _DummyProgressBar:
    def title(self, _):
        return self

    def draw(self):
        return None


class _DummyTimer:
    def __init__(self):
        self.events = []

    def record(self, name, duration, source=None):
        self.events.append({"name": name, "duration": duration, "source": source})


class _TinyLinearModel(torch.nn.Module):
    def __init__(self, features: int = 8):
        super().__init__()
        self.linear = torch.nn.Linear(features, features, bias=False)


@pytest.mark.cuda
def test_submodule_finalize_timing():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPTQ finalize benchmark")

    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices visible after masking; cannot benchmark finalize")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    base_model = _TinyLinearModel().to(device=device, dtype=torch.float16)
    named_module = NamedModule(
        base_model.linear,
        name="linear",
        full_name="linear",
        layer_index=0,
    )
    named_module.target_device = device
    named_module.module.target_device = device

    qcfg = QuantizeConfig(
        group_size=8,
        desc_act=False,
        sym=True,
        mock_quantization=True,
        pack_impl="original",
        offload_to_disk=False,
    )

    processor = GPTQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=None,
        prepare_dataset_func=_dummy_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        require_fwd=False,
        calculate_w_wq_diff=False,
    )
    processor.pb = _DummyProgressBar()

    processor.preprocess(named_module, fail_safe=False)
    processor.process(named_module)

    # move weights to CPU to satisfy create_quant_module invariants
    base_model.to("cpu")
    named_module.module.to("cpu")
    named_module.target_device = torch.device("cpu")
    named_module.module.target_device = torch.device("cpu")

    timer = _DummyTimer()
    quant_model = SimpleNamespace(
        model=base_model,
        quantize_config=qcfg,
        qlinear_kernel=TorchQuantLinear,
        lm_head="lm_head",
        quant_region_timer=timer,
        quantized=False,
    )

    events = []
    last_checkpoint = None

    def record_gap(label: str, t_now: float):
        nonlocal last_checkpoint
        if last_checkpoint is None:
            last_checkpoint = t_now
            return
        gap = t_now - last_checkpoint
        if gap > 0:
            events.append((label, gap))
        last_checkpoint = t_now

    original_create = gptq_processor_module.create_quant_module
    original_pack = gptq_processor_module.pack_module
    original_result_pop = GPTQProcessor.result_pop
    original_unregister = NamedModule.unregister_parameter

    def wrapped_create_quant_module(*args, **kwargs):
        nonlocal last_checkpoint
        start = time.perf_counter()
        record_gap("state_cleanup", start)
        try:
            return original_create(*args, **kwargs)
        finally:
            end = time.perf_counter()
            events.append(("create_quant_module", end - start))
            last_checkpoint = end

    def wrapped_pack_module(*args, **kwargs):
        nonlocal last_checkpoint
        start = time.perf_counter()
        record_gap("between_create_and_pack", start)
        packer = None
        try:
            packer = original_pack(*args, **kwargs)
            return packer
        finally:
            end = time.perf_counter()
            label = packer or "None"
            events.append((f"pack_module[{label}]", end - start))
            last_checkpoint = end

    def wrapped_result_pop(self, *args, **kwargs):
        nonlocal last_checkpoint
        start = time.perf_counter()
        record_gap("between_pack_and_result_pop", start)
        try:
            return original_result_pop(self, *args, **kwargs)
        finally:
            end = time.perf_counter()
            events.append(("result_pop", end - start))
            last_checkpoint = end

    def wrapped_unregister_parameter(self, *args, **kwargs):
        nonlocal last_checkpoint
        start = time.perf_counter()
        record_gap("between_result_pop_and_unregister", start)
        try:
            return original_unregister(self, *args, **kwargs)
        finally:
            end = time.perf_counter()
            events.append(("unregister_parameter", end - start))
            last_checkpoint = end

    start_time = time.perf_counter()
    last_checkpoint = start_time

    with (
        mock.patch("gptqmodel.looper.gptq_processor.create_quant_module", new=wrapped_create_quant_module),
        mock.patch("gptqmodel.looper.gptq_processor.pack_module", new=wrapped_pack_module),
        mock.patch.object(GPTQProcessor, "result_pop", new=wrapped_result_pop),
        mock.patch.object(NamedModule, "unregister_parameter", new=wrapped_unregister_parameter),
    ):
        processor.submodule_finalize(named_module, quant_model)

    end_time = time.perf_counter()
    record_gap("post_unregister_tail", end_time)
    total_elapsed = end_time - start_time
    events.append(("total_elapsed", total_elapsed))

    # sanity checks to ensure finalize replaced module and cleared state
    assert "q_scales" not in named_module.state
    assert "q_zeros" not in named_module.state
    assert "q_g_idx" not in named_module.state
    assert isinstance(quant_model.model.linear, TorchQuantLinear)

    create_time = sum(duration for label, duration in events if label == "create_quant_module")
    pack_time = sum(duration for label, duration in events if label.startswith("pack_module"))
    cleanup_labels = {
        "state_cleanup",
        "between_create_and_pack",
        "between_pack_and_result_pop",
        "between_result_pop_and_unregister",
        "post_unregister_tail",
    }
    cleanup_time = sum(duration for label, duration in events if label in cleanup_labels)
    other_time = total_elapsed - (create_time + pack_time + cleanup_time)

    print("\nsubmodule_finalize timing breakdown (ms):")
    for label, duration in events:
        print(f"  {label:<32} {duration * 1000:.3f}")

    print(f"\nSummary:")
    print(f"  total_elapsed_ms              = {total_elapsed * 1000:.3f}")
    print(f"  create_quant_module_ms        = {create_time * 1000:.3f}")
    print(f"  pack_module_ms                = {pack_time * 1000:.3f}")
    print(f"  cleanup_gaps_ms               = {cleanup_time * 1000:.3f}")
    print(f"  other_ms                      = {other_time * 1000:.3f}")

    if timer.events:
        print("\nquant_region_timer records:")
        for entry in timer.events:
            name = entry["name"]
            duration_ms = entry["duration"] * 1000
            source = entry.get("source")
            print(f"  {name:<32} {duration_ms:.3f} (source={source})")
