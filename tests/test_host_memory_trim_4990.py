# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Host-heap trimming at the quantization layer boundary (#4990).

AWQ leaves several GiB of freed host memory in the glibc heap; ``malloc_trim``
returns it to the OS. These tests cover the shared helper (cached glibc
resolution, silent no-op off glibc), a real measured reclaim, and the
per-layer call in the shared layer loop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
import torch

import gptqmodel.looper.stage_layer as stage_layer_module
from gptqmodel import QuantizeEmbed
from gptqmodel.looper.loop_processor import ForwardConfig
from gptqmodel.looper.module_looper import FinalizeProgressInfo
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.stage_layer import run_layer_stage
from gptqmodel.looper.stage_subset import SubsetStageResult
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils import host_memory


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _clear_resolution_cache() -> None:
    host_memory._resolved_malloc_trim.cache_clear()


def test_malloc_trim_resolves_the_symbol_once(monkeypatch):
    pad_values = []

    class _FakeTrim:
        def __init__(self):
            self.argtypes = None
            self.restype = None

        def __call__(self, pad):
            pad_values.append(pad)
            return 1

    class _FakeLibc:
        def __init__(self):
            self.malloc_trim = _FakeTrim()

    libc_names = []

    def fake_cdll(name):
        libc_names.append(name)
        return _FakeLibc()

    _clear_resolution_cache()
    monkeypatch.setattr(host_memory.ctypes, "CDLL", fake_cdll)
    try:
        assert host_memory.malloc_trim() == 1
        assert host_memory.malloc_trim() == 1
    finally:
        _clear_resolution_cache()

    assert libc_names == ["libc.so.6"]
    assert pad_values == [0, 0]


def test_malloc_trim_without_the_symbol_is_a_silent_noop(monkeypatch):
    libc_names = []

    class _NoTrimLibc:
        pass

    def fake_cdll(name):
        libc_names.append(name)
        return _NoTrimLibc()

    _clear_resolution_cache()
    monkeypatch.setattr(host_memory.ctypes, "CDLL", fake_cdll)
    try:
        assert host_memory.malloc_trim() is None
    finally:
        _clear_resolution_cache()

    # `libc.so.6` first, then the already-loaded process image.
    assert libc_names == ["libc.so.6", None]


def test_malloc_trim_is_a_noop_off_linux(monkeypatch):
    cdll_calls = []

    _clear_resolution_cache()
    monkeypatch.setattr(host_memory.sys, "platform", "darwin")
    monkeypatch.setattr(
        host_memory.ctypes, "CDLL", lambda name: cdll_calls.append(name) or object()
    )
    try:
        assert host_memory.malloc_trim() is None
    finally:
        _clear_resolution_cache()

    assert cdll_calls == []


_TRIM_MEASUREMENT_SCRIPT = r'''
import ctypes
import json
from pathlib import Path

from gptqmodel.utils.host_memory import malloc_trim


def rss_anon_bytes():
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("RssAnon is missing from /proc/self/status")


libc = ctypes.CDLL("libc.so.6")
libc.malloc.argtypes = [ctypes.c_size_t]
libc.malloc.restype = ctypes.c_void_p
libc.free.argtypes = [ctypes.c_void_p]
libc.free.restype = None

base = rss_anon_bytes()
blocks = [libc.malloc(64 * 1024) for _ in range(4096)]
for block in blocks:
    ctypes.memset(block, 1, 64 * 1024)
peak = rss_anon_bytes()
for block in blocks:
    libc.free(block)
freed = rss_anon_bytes()
result = malloc_trim()
after = rss_anon_bytes()
print(json.dumps({
    "base_mib": base / 2**20,
    "peak_mib": peak / 2**20,
    "freed_mib": freed / 2**20,
    "after_trim_mib": after / 2**20,
    "trim_result": result,
}))
'''


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="RssAnon and glibc malloc_trim need Linux /proc and glibc",
)
def test_malloc_trim_reclaims_retained_host_pages():
    """Real measurement: freed heap pages stay resident until `malloc_trim`."""

    if host_memory._resolved_malloc_trim() is None:
        pytest.skip("glibc malloc_trim is unavailable on this platform")

    # Keep glibc from shrinking the heap itself when the blocks are freed, so
    # the measurement isolates what `malloc_trim(0)` returns to the OS.
    env = dict(os.environ)
    env.update(
        MALLOC_TRIM_THRESHOLD_="1073741824",
        MALLOC_MMAP_THRESHOLD_="1073741824",
        MALLOC_ARENA_MAX="1",
    )
    completed = subprocess.run(
        [sys.executable, "-c", _TRIM_MEASUREMENT_SCRIPT],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr

    measurement = json.loads(completed.stdout.strip().splitlines()[-1])
    retained_mib = measurement["freed_mib"] - measurement["base_mib"]
    if retained_mib < 32:
        pytest.skip(
            "glibc returned the freed heap on free despite MALLOC_TRIM_THRESHOLD_; "
            "there is no retention for malloc_trim to reclaim here"
        )
    assert measurement["trim_result"] == 1
    assert measurement["freed_mib"] - measurement["after_trim_mib"] >= 32


class _DummyPB:
    def __init__(self, iterable):
        self._iterable = list(iterable)

    def __iter__(self):
        return iter(self._iterable)

    def __len__(self):
        return len(self._iterable)

    def manual(self):
        return self

    def set(self, **_):
        return self

    def title(self, *_):
        return self

    def subtitle(self, *_):
        return self

    def draw(self):
        return self

    def next(self):
        return self

    def close(self):
        return None


class _DummyLogger:
    def pb(self, iterable):
        return _DummyPB(iterable)

    def info(self, *_, **__):
        return None

    def debug(self, *_, **__):
        return None

    def warning(self, *_, **__):
        return None

    warn = warning

    def error(self, *_, **__):
        return None


class _StageLooperDouble:
    def _track_inflight_layer(self, *_args, **_kwargs):
        pass

    def _clear_inflight_layer(self, *_args, **_kwargs):
        pass

    def layer_modules_for_prefix(self, _prefix, *, planning=False):
        del planning
        return None

    def _check_loop_stop(self):
        return False

    def _is_attention_module_name(self, _name):
        return False

    def _extract_moe_group_key(self, _name):
        return None

    def _resolve_batch_total(self, _num_batches, layer_inputs):
        return len(layer_inputs)

    def _collect_row_counts(self, layer_inputs):
        return [1 for _ in layer_inputs]

    def _emit_layer_complete(self, **kwargs):
        self._layer_events.append(kwargs)

    def _request_loop_stop(self, exc):
        self._stop_exc = exc

    def _subset_event_dispatch(self, **_kwargs):
        pass


def _dummy_processor(*, finalize_raises: bool = False):
    class DummyProcessor:
        def __init__(self):
            self.forward = ForwardConfig(
                required=True,
                replay_after_process=False,
                all_modules_in_single_pass=False,
            )
            self.inputs_cache = types.SimpleNamespace(
                layer_inputs=[[torch.zeros(1, 4)]],
                layer_input_kwargs=[{}],
                position_ids=[],
                attention_masks=[],
            )
            self.calibration_dataset = []
            self.log = []
            self.tasks = {}
            self.layer_count = 1

        def collect_memory_info(self, *_):
            return None

        def clear_cache_data(self):
            return None

        def receive_layer_inputs(self, inputs):
            self.inputs_cache.layer_inputs = inputs

        def name(self):
            return "dummy"

        def submodule_finalize(self, *_):
            if finalize_raises:
                raise RuntimeError("finalize failed")
            return None

        def finalize(self, *_):
            return None

        def log_plotly(self):
            return None

    return DummyProcessor()


def _dummy_gptq_model(*, wait_for_finalizers: bool):
    class DummyGptqModel:
        def __init__(self):
            self.model = torch.nn.Module()
            self.model.config = types.SimpleNamespace(model_type="llama")
            self.quantize_config = QuantizeConfig(
                bits=4,
                group_size=128,
                offload_to_disk=False,
                wait_for_submodule_finalizers=wait_for_finalizers,
            )
            self.lm_head = None

        def pre_quantize(self, module, module_path=None):
            return module

        def should_quantize_layer(self, *_args):
            return True

        def post_quantize(self, module):
            return module

        def lm_head_pre_quantize_generate_hook(self, value):
            return value

    return DummyGptqModel()


def _run_layers(monkeypatch, *, layer_count: int, with_finalize: bool) -> list[int]:
    trims: list[int] = []
    monkeypatch.setattr(
        stage_layer_module,
        "_trim_host_heap_after_layer",
        lambda: trims.append(1),
    )
    monkeypatch.setattr(stage_layer_module, "find_modules", lambda *_, **__: {})

    named_modules = [
        NamedModule(
            torch.nn.Linear(4, 4, bias=False),
            name="self_attn.q_proj",
            full_name=f"model.layers.{index}.self_attn.q_proj",
            layer_index=index,
        )
        for index in range(layer_count)
    ]

    def fake_run_subset_stage(looper, **kwargs):
        index = kwargs["layer_index"]
        processed = (
            {"self_attn.q_proj": named_modules[index]} if with_finalize else {}
        )
        return SubsetStageResult(
            processed_subset=processed,
            layer_inputs=kwargs["layer_inputs"],
            plan=kwargs["plan"],
        )

    monkeypatch.setattr(stage_layer_module, "run_subset_stage", fake_run_subset_stage)

    class DummyLooper(_StageLooperDouble):
        def __init__(self):
            self.gptq_model = _dummy_gptq_model(
                wait_for_finalizers=with_finalize
            )
            self.processors = [_dummy_processor()]
            self.input_embeddings_name = None
            self.output_embeddings_name = None
            self._quant_devices = [torch.device("cpu")]
            self._module_device_map = {}
            self._quant_device_lock = threading.Lock()
            self._moe_subset_threshold = 16
            self._dense_quant_devices = [torch.device("cpu")]
            self._moe_quant_devices = [torch.device("cpu")]
            self._dense_vram_strategy = types.SimpleNamespace()
            self._moe_vram_strategy = types.SimpleNamespace()
            self._dense_vram_strategy_explicit = False
            self._moe_vram_strategy_explicit = False
            self._layer_events = []
            self._stop_exc = None
            self._dangling = []

        def register_dangling_thread(self, thread):
            self._dangling.append(thread)

        def wait_dangling_threads(self):
            for thread in self._dangling:
                thread.join()
            self._dangling.clear()

        def create_named_modules(
            self,
            module,
            full,
            is_lm_head_module,
            layer_index,
            layers_prefix,
            names,
            processor,
            fallback,
            layer_module=None,
        ):
            return {
                "self_attn.q_proj": NamedModule(
                    module,
                    name="self_attn.q_proj",
                    full_name=f"model.layers.{layer_index}.self_attn.q_proj",
                    layer_index=layer_index,
                )
            }

    looper = DummyLooper()
    layers = [torch.nn.Linear(4, 4, bias=False) for _ in range(layer_count)]
    layer_names = [f"model.layers.{index}" for index in range(layer_count)]
    run_layer_stage(
        looper,
        layers=layers,
        layer_modules=[["self_attn.q_proj"] for _ in range(layer_count)],
        planning_layer_modules=[["self_attn.q_proj"] for _ in range(layer_count)],
        layer_names=layer_names,
        fallback=True,
        shared_kv_cache_dict={},
        pb=_DummyPB(range(layer_count)),
        layer_count=layer_count,
        region_timer=None,
        finalize_progress_cls=FinalizeProgressInfo,
        embed_quant_mode=QuantizeEmbed.OUTPUT,
        embed_only=False,
        logger=_DummyLogger(),
    )
    looper.wait_dangling_threads()
    return trims


def test_stage_layer_trims_once_per_layer(monkeypatch):
    trims = _run_layers(monkeypatch, layer_count=2, with_finalize=False)

    assert len(trims) == 2


def test_stage_layer_trims_after_submodule_finalize(monkeypatch):
    trims = _run_layers(monkeypatch, layer_count=1, with_finalize=True)

    assert len(trims) == 1
