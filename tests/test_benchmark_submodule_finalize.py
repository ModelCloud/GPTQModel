import os


# force pytest runs to target GPU 7 (becomes cuda:0 after masking)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from gptqmodel.looper import gptq_processor as gptq_processor_module
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.threadx import DeviceThreadPool


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


class _LockTracker:
    def __init__(self):
        self._state_lock = threading.Lock()
        self._entries = []
        self._active = 0
        self.max_active = 0

    def enter(self):
        start = time.perf_counter()
        with self._state_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        return start

    def exit(self, start_time: float):
        end = time.perf_counter()
        with self._state_lock:
            self._active -= 1
            self._entries.append((start_time, end - start_time))

    def total_duration(self) -> float:
        with self._state_lock:
            return sum(duration for _, duration in self._entries)

    def window_duration(self) -> float:
        with self._state_lock:
            if not self._entries:
                return 0.0
            start_min = min(start for start, _ in self._entries)
            end_max = max(start + duration for start, duration in self._entries)
            return end_max - start_min

    def entry_count(self) -> int:
        with self._state_lock:
            return len(self._entries)


class _CountingLock:
    def __init__(self, base_lock, tracker: _LockTracker):
        self._base_lock = base_lock
        self._tracker = tracker
        self._start_time = None

    def __enter__(self):
        self._base_lock.acquire()
        self._start_time = self._tracker.enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._tracker.exit(self._start_time)
        self._start_time = None
        self._base_lock.release()
        return False


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

    print("\nSummary:")
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


def _prepare_modules(processor, qcfg, device, module_count):
    modules = []
    for idx in range(module_count):
        base_model = _TinyLinearModel().to(device=device, dtype=torch.float16)
        named_module = NamedModule(
            base_model.linear,
            name=f"linear_{idx}",
            full_name="linear",
            layer_index=idx,
        )
        named_module.target_device = device
        named_module.module.target_device = device

        processor.preprocess(named_module, fail_safe=False)
        processor.process(named_module)

        base_model.to("cpu")
        named_module.module.to("cpu")
        named_module.target_device = torch.device("cpu")
        named_module.module.target_device = torch.device("cpu")

        quant_model = SimpleNamespace(
            model=base_model,
            quantize_config=qcfg,
            qlinear_kernel=TorchQuantLinear,
            lm_head="lm_head",
            quant_region_timer=_DummyTimer(),
            quantized=False,
        )
        modules.append((named_module, quant_model))
    return modules


def _finalize_worker(processor, module, quant_model):
    start = time.perf_counter()
    processor.submodule_finalize(module, quant_model)
    end = time.perf_counter()
    thread_name = threading.current_thread().name
    return thread_name, start, end


@pytest.mark.cuda
@pytest.mark.parametrize("cpu_workers", [8, 32])
def test_submodule_finalize_threadpool_serialization(cpu_workers):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GPTQ finalize pool benchmark")

    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices visible after masking; cannot benchmark finalize")

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

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

    module_count = min(cpu_workers * 2, 32)
    modules = _prepare_modules(processor, qcfg, device, module_count)

    lock_tracker = _LockTracker()
    original_pack_module = gptq_processor_module.pack_module

    def instrumented_pack_module(*args, **kwargs):
        args_list = list(args)

        if "lock" in kwargs and kwargs["lock"] is not None:
            kwargs = dict(kwargs)
            kwargs["lock"] = _CountingLock(kwargs["lock"], lock_tracker)
        elif len(args_list) > 7:
            args_list[7] = _CountingLock(args_list[7], lock_tracker)
        else:
            raise AssertionError("Expected lock argument in pack_module")

        return original_pack_module(*tuple(args_list), **kwargs)

    pool = DeviceThreadPool(include_cuda=False, include_cpu=True, workers={"cpu": cpu_workers}, inference_mode=True)

    try:
        with mock.patch.object(gptq_processor_module, "pack_module", new=instrumented_pack_module):
            futures = [
                pool.submit(torch.device("cpu"), _finalize_worker, processor, module, quant_model)
                for module, quant_model in modules
            ]
            results = [future.result() for future in futures]
    finally:
        pool.shutdown()

    thread_names = {name for name, _, _ in results}
    lock_total = lock_tracker.total_duration()
    lock_window = lock_tracker.window_duration()

    assert lock_tracker.entry_count() == len(modules)
    assert lock_tracker.max_active == 1, f"Expected serialized pack_module, saw max_active={lock_tracker.max_active}"

    if lock_total > 0 and lock_window > 0:
        ratio = lock_total / lock_window
        assert ratio <= 1.05, f"Expected serialized execution; total/window ratio={ratio:.3f}"

    assert len(thread_names) >= min(cpu_workers, len(modules), 2), "Thread pool failed to utilize multiple workers"
