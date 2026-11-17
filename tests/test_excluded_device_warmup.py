# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Excluded accelerators must stay unclaimed during quantization (#4320, #4653).

#4320 made the pool's device activation lazy, but three later paths still
touched an excluded GPU: ``torch_sync()`` synchronized every *visible* index,
``QuantizeConfig.compute_device_filter`` was never consulted by the warmup or
prewarm resolve, and the janitor's ``empty_cache`` sweep covered every pool
device. Each one creates a full primary context (hundreds of MiB) on a GPU the
caller excluded; the hardware test below runs a real quantization and checks
the excluded device from the driver's compute-app list.
"""

import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import gptqmodel
import gptqmodel.models.auto as auto
import gptqmodel.utils.threadx as threadx
from gptqmodel.utils.threadx import DeviceThreadPool, WarmupTask


_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny llama calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose so filtering keeps it",
] * 2

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _fake_device_pinning(monkeypatch):
    """Keep warmup bookkeeping off the real device ordinals.

    These tests assert which devices the pool *schedules*, not that the device
    exists. ``_device_ctx`` enters ``torch.cuda.device("cuda:1")``, which raises
    ``invalid device ordinal`` on a one-GPU host and silently aborts the warmup
    the test is waiting for (#4768).
    """

    @contextlib.contextmanager
    def _ctx(dev):
        yield

    monkeypatch.setattr(threadx, "_device_ctx", _ctx)


def test_warmup_devices_skips_excluded_device(monkeypatch):
    activated = []
    warmed = threading.Event()

    def warmup(device, ctx):
        warmed.set()

    monkeypatch.setattr(threadx, "_activate_thread_device", lambda device: activated.append(str(device)))
    pool = DeviceThreadPool(
        devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        workers={"cuda:per": 1},
        warmups={"cuda": WarmupTask(warmup)},
        warmup_devices=[torch.device("cuda:1")],
        empty_cache_every_n=0,
    )
    try:
        assert warmed.wait(timeout=2)
        time.sleep(0.2)
        assert activated == ["cuda:1"]
    finally:
        pool.shutdown(wait=True)


def test_default_pool_still_warms_every_accelerator(monkeypatch):
    activated = []
    warmed = threading.Event()
    warmed_count = 0
    lock = threading.Lock()

    def warmup(device, ctx):
        nonlocal warmed_count
        with lock:
            warmed_count += 1
            if warmed_count == 2:
                warmed.set()

    monkeypatch.setattr(threadx, "_activate_thread_device", lambda device: activated.append(str(device)))
    pool = DeviceThreadPool(
        devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        workers={"cuda:per": 1},
        warmups={"cuda": WarmupTask(warmup)},
        empty_cache_every_n=0,
    )
    try:
        assert warmed.wait(timeout=2)
        assert set(activated) == {"cuda:0", "cuda:1"}
    finally:
        pool.shutdown(wait=True)


class _RecordingPool:
    """Minimal pool that records the warmup allow-list it was resized with."""

    def __init__(self, warmup_devices):
        self.warmup_devices = warmup_devices
        self.set_calls = []

    def set_warmup_devices(self, warmup_devices):
        self.set_calls.append(warmup_devices)


def _patch_pool_rebuild(monkeypatch, auto):
    captured = {}
    pool = _RecordingPool(None)

    def fake_resize(warmup_devices=None, cpu_workers=None, *, create=True):
        captured["warmup_devices"] = warmup_devices
        pool.warmup_devices = warmup_devices
        return pool

    monkeypatch.setattr(auto, "env_flag", lambda *args, **kwargs: True)
    monkeypatch.setattr(gptqmodel, "resize_device_thread_pool", fake_resize)
    return captured, pool


def test_prebuild_passes_explicit_devices_to_pool(monkeypatch):
    captured, _pool = _patch_pool_rebuild(monkeypatch, auto)
    monkeypatch.setattr(
        auto,
        "_explicit_requested_devices",
        lambda device, **_kwargs: [torch.device("cuda:1")],
    )

    auto._prebuild_device_thread_pool(("cuda:1",))

    assert captured["warmup_devices"] == [torch.device("cuda:1")]


def test_prebuild_applies_compute_device_filter(monkeypatch):
    """The documented filter must narrow the warmed pool, not only the loopers."""

    captured, pool = _patch_pool_rebuild(monkeypatch, auto)
    monkeypatch.setattr(
        auto,
        "select_forward_devices",
        lambda base: [torch.device("cuda:0"), torch.device("cuda:1")],
    )
    config = SimpleNamespace(
        device=torch.device("cuda"),
        compute_device_filter=lambda devices: [d for d in devices if d.index != 0],
    )

    auto._prebuild_device_thread_pool(None, quantize_config=config)

    assert captured["warmup_devices"] == [torch.device("cuda:1")]
    assert pool.set_calls == [[torch.device("cuda:1")]]


def test_prebuild_without_restriction_keeps_family_warmup(monkeypatch):
    captured, _pool = _patch_pool_rebuild(monkeypatch, auto)
    config = SimpleNamespace(device=torch.device("cuda"), compute_device_filter=None)

    auto._prebuild_device_thread_pool(None, quantize_config=config)

    assert captured["warmup_devices"] is None


def test_prebuild_keeps_full_pool_when_filter_rejects_everything(monkeypatch):
    """The loopers fall back to the full pool, so the warmup must match."""

    captured, _pool = _patch_pool_rebuild(monkeypatch, auto)
    monkeypatch.setattr(
        auto,
        "select_forward_devices",
        lambda base: [torch.device("cuda:0"), torch.device("cuda:1")],
    )
    config = SimpleNamespace(device=torch.device("cuda"), compute_device_filter=lambda devices: [])

    auto._prebuild_device_thread_pool(None, quantize_config=config)

    assert captured["warmup_devices"] is None


def test_prewarm_target_follows_the_filtered_pool(monkeypatch):
    """The linear/quantize prewarm used the load kwarg, never the filter."""

    import gptqmodel.models.auto as auto

    monkeypatch.setattr(
        auto,
        "select_forward_devices",
        lambda base: [torch.device("cuda:0"), torch.device("cuda:1")],
    )
    config = SimpleNamespace(
        device=torch.device("cuda"),
        compute_device_filter=lambda devices: [d for d in devices if d.index != 0],
    )

    assert auto._resolve_prewarm_device(None, None, config) == torch.device("cuda:1")
    assert auto._resolve_prewarm_device(None, None) == torch.device("cuda")


def _patch_sync_surface(monkeypatch, torch_utils, claimed):
    """Stub the CUDA surface ``torch_sync()`` walks; ``claimed(index)`` is the driver."""

    synchronized = []
    monkeypatch.setattr(torch_utils, "HAS_CUDA", True)
    monkeypatch.setattr(torch_utils, "HAS_XPU", False)
    monkeypatch.setattr(torch_utils, "HAS_MPS", False)
    monkeypatch.setattr(torch_utils, "HAS_NPU", False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch_utils, "xla_runtime_claimed", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index=None: synchronized.append(index))
    monkeypatch.setattr(torch._C, "_cuda_hasPrimaryContext", claimed, raising=False)
    return synchronized


def test_torch_sync_skips_devices_without_a_context(monkeypatch):
    """A synchronize on an untouched index would create its context (#4653)."""

    import gptqmodel.utils.torch as torch_utils

    synchronized = _patch_sync_surface(monkeypatch, torch_utils, lambda index: index == 1)

    torch_utils.torch_sync()

    assert synchronized == [1]


def test_torch_sync_keeps_the_previous_behavior_without_the_query(monkeypatch):
    """A torch that cannot report contexts must not lose its barriers."""

    import gptqmodel.utils.torch as torch_utils

    synchronized = _patch_sync_surface(monkeypatch, torch_utils, lambda index: True)

    # Older wheels expose no primary-context query at all.
    monkeypatch.delattr(torch._C, "_cuda_hasPrimaryContext", raising=False)

    torch_utils.torch_sync()

    assert synchronized == [0, 1]


def test_torch_sync_with_explicit_device_is_unchanged(monkeypatch):
    import gptqmodel.utils.torch as torch_utils

    synchronized = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device=None: synchronized.append(device),
    )
    monkeypatch.setattr(torch._C, "_cuda_hasPrimaryContext", lambda index: False, raising=False)

    torch_utils.torch_sync(device=torch.device("cuda:0"))

    assert synchronized == [torch.device("cuda:0")]


def test_janitor_sweep_skips_devices_without_a_context(monkeypatch):
    """The GC sweep also called ``empty_cache`` on every pool device."""

    pool = DeviceThreadPool(
        devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        workers={"cuda:per": 1},
        warmup_devices=[],
        empty_cache_every_n=512,
    )
    swept = []
    monkeypatch.setattr(
        torch._C,
        "_cuda_hasPrimaryContext",
        lambda index: index == 1,
        raising=False,
    )
    monkeypatch.setattr(
        pool,
        "_run_empty_cache_for_device",
        lambda key, dev: swept.append(key) or 0.0,
    )
    try:
        # Reproduce the sweep-target selection the janitor performs.
        pending_targets = ["cuda:0", "cuda:1"]
        pool._gc_pending_physical = dict.fromkeys(pending_targets, 1)
        with pool._stats_lock:
            pool._gc_generation = 1
        pool._gc_event.set()
        deadline = time.time() + 10
        while time.time() < deadline and "cuda:1" not in swept:
            time.sleep(0.02)
    finally:
        pool.shutdown(wait=True)

    assert "cuda:1" in swept
    assert "cuda:0" not in swept


def test_set_warmup_devices_narrows_a_retained_pool(monkeypatch):
    """A retained pool must apply the new load's allow-list to its warmups."""

    activated = []
    warmed = threading.Event()

    def warmup(device, ctx):
        if str(device) == "cuda:1":
            warmed.set()

    monkeypatch.setattr(threadx, "_activate_thread_device", lambda device: activated.append(str(device)))
    pool = DeviceThreadPool(
        devices=[torch.device("cuda:0"), torch.device("cuda:1")],
        workers={"cuda:per": 1},
        warmups={"cuda": WarmupTask(warmup)},
        warmup_devices=[],
        empty_cache_every_n=0,
    )
    try:
        time.sleep(0.2)
        assert activated == []
        pool.set_warmup_devices([torch.device("cuda:1")])
        assert warmed.wait(timeout=2)
        time.sleep(0.2)
        assert "cuda:0" not in activated
    finally:
        pool.shutdown(wait=True)


def test_device_ctx_family_selector_does_not_claim_an_unused_device_0(monkeypatch):
    """``quantize_config.device`` holds the family, not "select GPU 0"."""

    from gptqmodel.models._const import DEVICE
    from gptqmodel.utils.looper_helpers import device_ctx

    observed = []

    def fake_device(index):
        observed.append(index)

        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Ctx()

    monkeypatch.setattr(torch.cuda, "device", fake_device)
    monkeypatch.setattr(torch._C, "_cuda_hasPrimaryContext", lambda index: False, raising=False)

    with device_ctx(DEVICE.CUDA):
        pass
    with device_ctx(DEVICE.ROCM):
        pass
    with device_ctx(torch.device("cuda:1")):
        pass
    with device_ctx("cuda:1"):
        pass

    # Only the explicitly pinned devices select a device at all.
    assert observed == [1, 1]


def test_device_ctx_family_selector_keeps_selecting_device_0_when_active(monkeypatch):
    """The historical "family means device 0" behavior survives where it applies."""

    from gptqmodel.models._const import DEVICE
    from gptqmodel.utils.looper_helpers import device_ctx

    observed = []

    def fake_device(index):
        observed.append(index)

        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, *exc):
                return False

        return _Ctx()

    monkeypatch.setattr(torch.cuda, "device", fake_device)
    monkeypatch.setattr(torch._C, "_cuda_hasPrimaryContext", lambda index: True, raising=False)

    with device_ctx(DEVICE.CUDA):
        pass

    assert observed == [0]


def _tiny_llama_dir(tmp_path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    texts = _CALIBRATION_TEXTS
    model_dir = tmp_path / "tiny-llama"
    model_dir.mkdir()
    torch.manual_seed(1234)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).to(torch.float16).eval().save_pretrained(model_dir)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        texts,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(model_dir)
    return model_dir


_EXCLUDED_DEVICE_SCRIPT = r"""
import json
import sys

import torch

model_dir = sys.argv[1]
mode = sys.argv[2]

from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel, CalibrationConfig
from gptqmodel.quantization import GPTQConfig

tokenizer = AutoTokenizer.from_pretrained(model_dir)
calibration = [
    dict(tokenizer(text, return_tensors="pt"))
    for text in [
        "the quick brown fox jumps over the lazy dog and keeps running through the field",
        "a tiny llama calibration sample with enough tokens to survive length filtering",
        "another synthetic calibration example deliberately verbose so filtering keeps it",
    ] * 2
]

if mode == "filter":
    config = GPTQConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        offload_to_disk=False,
        compute_device_filter=lambda devices: [d for d in devices if d.index != 0],
    )
    model = GPTQModel.load(model_dir, config, dtype=torch.float16, backend=BACKEND.TORCH)
else:
    config = GPTQConfig(bits=4, group_size=32, desc_act=False, offload_to_disk=False)
    model = GPTQModel.load(
        model_dir,
        config,
        dtype=torch.float16,
        backend=BACKEND.TORCH,
        device=("cuda:1",),
    )

query = getattr(torch._C, "_cuda_hasPrimaryContext", None)
before = bool(query(0)) if query is not None else None
model.quantize(
    CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
    backend=BACKEND.TORCH,
)
print(
    "RESULT="
    + json.dumps(
        {
            "before": before,
            "claimed": bool(query(0)) if query is not None else None,
            "peak0": int(torch.cuda.max_memory_allocated(0)),
            "peak1": int(torch.cuda.max_memory_allocated(1)),
        }
    ),
    flush=True,
)
"""


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("mode", ["filter", "loaddev"])
def test_quantize_never_claims_the_excluded_device(tmp_path, mode):
    """Both documented exclusion routes must leave GPU 0 without a context."""

    query = getattr(torch._C, "_cuda_hasPrimaryContext", None)
    if query is None:
        pytest.skip("this torch build cannot report per-device primary contexts")
    if torch.version.hip is not None:
        # HIP's primary-context query reports every visible device once *any*
        # of them is initialized, so it cannot attribute a context to a device.
        pytest.skip("HIP reports a process-wide primary context, not a per-device one")

    visible = _visible_cuda_pair()
    if visible is None:
        pytest.skip("requires numeric CUDA_VISIBLE_DEVICES entries")

    model_dir = _tiny_llama_dir(tmp_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("OMP_NUM_THREADS", "4")

    result = subprocess.run(
        [sys.executable, "-c", _EXCLUDED_DEVICE_SCRIPT, str(model_dir), mode],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    payload = next(
        json.loads(line[len("RESULT=") :])
        for line in result.stdout.splitlines()
        if line.startswith("RESULT=")
    )
    assert payload["before"] is False, "the excluded GPU was claimed before quantization"
    assert payload["peak1"] > 0, "the permitted device did not quantize"
    assert payload["claimed"] is False, (
        f"excluded GPU still has a primary context after quantize ({payload})"
    )
    assert payload["peak0"] == 0, f"excluded GPU got allocations ({payload})"


def _visible_cuda_pair():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return "0,1"
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if len(entries) < 2 or not all(entry.isdigit() for entry in entries[:2]):
        return None
    return ",".join(entries[:2])


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_excluded_cuda_device_has_no_context():
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi is not available")

    visible = _visible_cuda_pair()
    if visible is None:
        pytest.skip("requires numeric CUDA_VISIBLE_DEVICES entries")

    script = r"""
import os
import subprocess
import time

import torch

torch.set_num_threads(4)

from gptqmodel import get_device_thread_pool

try:
    pool = get_device_thread_pool(warmup_devices=[torch.device("cuda:1")])
except TypeError:
    # Older trees do not expose the warmup allow-list; building the default
    # pool there reproduces the bug this test guards against.
    pool = get_device_thread_pool()
holder = None


def work():
    global holder
    holder = torch.ones(1024 * 1024, device="cuda:1")
    return int(holder.numel())


print("TASK=", pool.do("cuda:1", work))
pool.wait("cuda:1")
time.sleep(2)
print("PID=", os.getpid())
print("ALLOC1=", int(torch.cuda.memory_allocated(1)))
print("GPUS_BEGIN")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
).strip())
print("GPUS_END")
print("APPS_BEGIN")
print(subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
    text=True,
).strip())
print("APPS_END")
"""

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("OPENBLAS_NUM_THREADS", "4")
    env.setdefault("NUMEXPR_NUM_THREADS", "4")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-4000:]

    lines = result.stdout.splitlines()
    pid = next(int(line.split("=", 1)[1]) for line in lines if line.startswith("PID="))
    alloc1 = next(int(line.split("=", 1)[1]) for line in lines if line.startswith("ALLOC1="))

    gpus_block = lines[lines.index("GPUS_BEGIN") + 1 : lines.index("GPUS_END")]
    apps_block = lines[lines.index("APPS_BEGIN") + 1 : lines.index("APPS_END")]

    uuid_by_index = {}
    for line in gpus_block:
        if not line.strip():
            continue
        index, uuid = [part.strip() for part in line.split(",", 1)]
        uuid_by_index[index] = uuid

    app_uuids = set()
    for line in apps_block:
        if not line.strip():
            continue
        row_pid, gpu_uuid, _used = [part.strip() for part in line.split(",", 2)]
        if int(row_pid) == pid:
            app_uuids.add(gpu_uuid)

    excluded_index, permitted_index = visible.split(",")
    excluded_uuid = uuid_by_index[excluded_index]
    permitted_uuid = uuid_by_index[permitted_index]

    assert alloc1 > 0, "the permitted device did not allocate"
    assert permitted_uuid in app_uuids, "nvidia-smi did not report the permitted device"
    assert excluded_uuid not in app_uuids, (
        f"excluded GPU {excluded_index} ({excluded_uuid}) still has a CUDA context; compute apps: {apps_block}"
    )
