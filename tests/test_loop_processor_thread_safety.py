# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Thread-safety and API coverage tests for LoopProcessor shared state."""

import threading
from concurrent.futures import ThreadPoolExecutor, wait
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.loop_processor import (
    LoopProcessor,
    _SafeDict,
    _ThreadSafeInputCache,
)
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.definitions.laguna import LagunaQModel
from gptqmodel.quantization.config import (
    ExpertsRoutingBypass,
    MoEConfig,
    MoERoutingConfig,
    QuantizeConfig,
)


def _make_processor() -> LoopProcessor:
    """Build a minimal LoopProcessor with all thread-safety primitives wired."""

    p: LoopProcessor = LoopProcessor.__new__(LoopProcessor)
    p.lock = threading.Lock()
    p._results_lock = threading.Lock()
    p._pb_lock = threading.Lock()
    p._fwd_time_lock = threading.Lock()
    p._device_smi_lock = threading.RLock()
    p._cache_lock = threading.RLock()
    p._global_sample_count_lock = threading.Lock()
    p.qcfg = QuantizeConfig(bits=4, group_size=8, sym=True, desc_act=False)
    p.is_weight_only = False
    p._global_reference_nsamples = None
    p._global_max_padded_nsamples = 0
    p.total_calibration_tokens = 0
    p._results = {}
    p.tasks = _SafeDict()
    p.inputs_cache = _ThreadSafeInputCache(InputCache([], [], [], [], None))
    p.fwd_time = None
    p.pb = None
    p._device_smi_handles = {}
    p._device_metric_failures = set()
    p._cpu_device_smi = None
    p._batch_tls = threading.local()
    p.gptq_model = None
    return p


def test_safe_dict_concurrent_read_write_iteration():
    """_SafeDict handles mixed concurrent operations without corruption."""

    d = _SafeDict()
    barrier = threading.Barrier(8)
    errors = []

    def worker(i: int):
        try:
            barrier.wait(timeout=5)
            for _ in range(100):
                d[f"key_{i}"] = i
                _ = d.get(f"key_{i}")
                _ = d[f"key_{i}"]
                _ = list(d.values())
                _ = list(d.items())
                _ = list(d.keys())
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(d) == 8
    for i in range(8):
        assert d[f"key_{i}"] == i


def test_safe_dict_pop_and_clear_are_thread_safe():
    """pop/popitem/clear are safe against concurrent readers."""

    d = _SafeDict({f"k{i}": i for i in range(10)})
    pops = []

    def worker():
        for _ in range(5):
            try:
                key, value = d.popitem()
                pops.append((key, value))
            except KeyError:
                pass

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(d) == 0 or len(d) + len(pops) == 10
    assert len(pops) == 10


def test_thread_safe_input_cache_proxy_isolates_set_cache():
    """set_cache replaces the wrapped cache without losing attribute access."""

    cache = _ThreadSafeInputCache(InputCache([], [], [], [], None))
    assert cache.layer_inputs == []
    cache.layer_inputs = [[torch.tensor([1.0])]]
    assert len(cache.layer_inputs) == 1

    new_cache = InputCache([torch.tensor([2.0])], None, None, None, None)
    cache.set_cache(new_cache)
    assert cache.layer_inputs == [torch.tensor([2.0])]


def test_thread_safe_input_cache_concurrent_attribute_access():
    """Concurrent reads/writes to the same cache attribute do not crash."""

    cache = _ThreadSafeInputCache(InputCache([], [], [], [], None))
    futures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(50):
            futures.append(
                executor.submit(
                    lambda i=i: cache.__setattr__(
                        "layer_inputs", [[torch.tensor([float(i)])]]
                    )
                )
            )
            futures.append(executor.submit(lambda: len(cache.layer_inputs)))
    wait(futures)
    # The final value is one of the writes; no exception is the real assertion.
    assert isinstance(cache.layer_inputs, list)


def test_results_are_thread_safe_and_return_snapshot():
    """result_save/get/pop/results are safe and results returns a copy."""

    p = _make_processor()

    def worker(i: int):
        p.result_save(f"mod_{i}", {"loss": float(i)})
        p.result_get(f"mod_{i}")
        if i % 2 == 0:
            p.result_pop(f"mod_{i}")

    with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(100):
            executor.submit(worker, i)

    snapshot = p.results()
    assert isinstance(snapshot, dict)
    # Popped only the even keys.
    assert len(snapshot) == 50
    # Mutation of snapshot must not affect internal state.
    snapshot["extra"] = 1
    assert "extra" not in p.results()


def test_fwd_time_is_thread_safe():
    """set_fwd_time and formatted_fwd_time do not race."""

    p = _make_processor()

    def worker(i: int):
        p.set_fwd_time(float(i))
        p.formatted_fwd_time()

    with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(200):
            executor.submit(worker, i)

    assert p.fwd_time is not None
    text = p.formatted_fwd_time()
    assert text.endswith(".000")


def test_draw_progress_is_thread_safe():
    """draw_progress uses the progress-bar lock and chains calls safely."""

    p = _make_processor()
    p.pb = MagicMock()
    p.pb.title.return_value = p.pb
    p.pb.subtitle.return_value = p.pb

    def worker(i: int):
        p.draw_progress(f"title {i}", f"subtitle {i}")

    with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(100):
            executor.submit(worker, i)

    assert p.pb.draw.call_count == 100


def test_device_memory_report_is_thread_safe():
    """device_memory_report and snapshot helpers are safe to call concurrently."""

    class FakeMetrics:
        memory_used = 2 * 1024**3

    class FakeHandle:
        def metrics(self, fast: bool = True):
            return FakeMetrics()

    p = _make_processor()
    p._device_smi_handles = {"cuda:0": FakeHandle(), "cuda:1": FakeHandle()}

    def worker(_):
        return p.device_memory_report()

    with ThreadPoolExecutor(max_workers=8) as executor:
        reports = list(executor.map(worker, range(20)))

    assert all("cuda" in r and "2G" in r for r in reports)


def test_input_cache_methods_are_thread_safe():
    """receive_input_cache, receive_layer_inputs, and clear_cache_data are safe."""

    p = _make_processor()

    def writer(i: int):
        p.receive_input_cache(
            InputCache([torch.tensor([float(i)])], None, None, None, None)
        )
        p.receive_layer_inputs([[torch.tensor([float(i)])]])

    def reader():
        return len(p.inputs_cache.layer_inputs)

    futures = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for i in range(50):
            futures.append(executor.submit(writer, i))
            futures.append(executor.submit(reader))
    wait(futures)

    p.clear_cache_data()
    assert len(p.inputs_cache.layer_inputs) == 0
    assert len(p.tasks) == 0


def test_current_batch_index_is_thread_local():
    """_set_current_batch_index/current_batch_index are isolated per thread."""

    p = _make_processor()
    results = {}

    def worker(tid: int):
        p._set_current_batch_index(tid)
        results[tid] = p.current_batch_index()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i in range(8):
        assert results[i] == i


def test_assert_calibration_sample_count_constant_for_dense_and_moe_bypass():
    """Dense modules and MoE bypass routing require a constant sample count."""

    p = _make_processor()
    p.gptq_model = object.__new__(LagunaQModel)

    p._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 512)
    p._assert_calibration_sample_count("model.layers.0.self_attn.k_proj", 512)
    assert p._global_reference_nsamples == 512

    p.qcfg.moe = MoEConfig(routing=ExpertsRoutingBypass())
    p._assert_calibration_sample_count("model.layers.0.mlp.experts.0.down_proj", 512)

    with pytest.raises(AssertionError):
        p._assert_calibration_sample_count("model.layers.0.self_attn.v_proj", 256)


def test_assert_calibration_sample_count_allows_routed_moe_variance():
    """Native-routed MoE experts may see fewer samples than the reference and up to the padded bound."""

    p = _make_processor()
    p.gptq_model = object.__new__(LagunaQModel)
    p.qcfg.moe = MoEConfig(routing=MoERoutingConfig())
    p._global_max_padded_nsamples = 2048

    routed_name = "model.layers.0.mlp.router"
    routed = NamedModule(nn.Linear(4, 4), name=routed_name, full_name=routed_name, layer_index=0)
    routed.state["module_tree_flags"] = frozenset({"moe"})
    p.tasks[routed_name] = MagicMock(_named_module=routed)

    p._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 512)
    p._assert_calibration_sample_count(routed_name, 256)
    p._assert_calibration_sample_count(routed_name, 800)
    p._assert_calibration_sample_count(routed_name, 2048)
    p._assert_calibration_sample_count("model.layers.0.mlp.experts.0.gate_proj", 0)

    with pytest.raises(AssertionError):
        p._assert_calibration_sample_count(routed_name, 2049)


def test_module_tree_helpers_public_and_safe():
    """All generic module-tree helpers can be called without error."""

    p = _make_processor()
    p.gptq_model = object.__new__(LagunaQModel)
    p.qcfg.moe = MoEConfig(routing=MoERoutingConfig())

    assert p._is_bypass_moe_routing() is False
    assert p._is_bypass_moe_routing() is False

    p.qcfg.moe = MoEConfig(routing=ExpertsRoutingBypass())
    assert p._is_bypass_moe_routing() is True

    # Routed expert isolation is derived from explicit module-tree flags.
    routed_name = "model.layers.0.mlp.experts.12.down_proj"
    routed = NamedModule(nn.Linear(4, 4), name=routed_name, full_name=routed_name, layer_index=0)
    routed.state["module_tree_flags"] = frozenset({"down", "routed"})
    routed.state["module_tree_expert_group"] = "model.layers.0.mlp.specialists.12"
    p.tasks[routed_name] = MagicMock(_named_module=routed)
    assert p._module_expert_isolation_key(
        routed_name
    ) == "model.layers.0.mlp.specialists.12"
    assert (
        p._module_expert_isolation_key("model.layers.0.mlp.shared_expert.down_proj")
        is None
    )

    # Expert down-proj detection does not guess from conventional path names.
    assert p._module_is_expert_down_proj(routed_name) is True
    assert (
        p._module_is_expert_down_proj("model.layers.0.mlp.experts.0.gate_proj") is False
    )

    # MoE membership uses metadata attached to the task.
    assert p._module_is_moe_related(routed_name) is True
    assert p._module_is_moe_related("model.layers.0.mlp.gate") is False
    assert p._module_is_moe_related("model.layers.0.self_attn.q_proj") is False


def test_public_hook_methods_are_callable():
    """The base processor hook methods can be invoked with minimal arguments."""

    p = _make_processor()
    module = NamedModule(nn.Linear(4, 4), name="m", full_name="m", layer_index=0)

    assert p.preprocess(module) is None
    assert p.is_skipped(module) is None
    assert p.process(module) is None
    assert p.prepare_subset({"m": module}) is None
    assert p.cleanup_subset({"m": module}) is None
    assert p.set_calibration_dataset([]) is None
    assert p.log_plotly() is None
