# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Excluded accelerators must not get a CUDA context from pool warmup (#4320)."""

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

import gptqmodel.utils.threadx as threadx
from gptqmodel.utils.threadx import DeviceThreadPool, WarmupTask


pytestmark = [pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_prebuild_passes_explicit_devices_to_pool(monkeypatch):
    import gptqmodel
    import gptqmodel.models.auto as auto

    captured = {}
    monkeypatch.setattr(auto, "env_flag", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        auto,
        "_explicit_requested_devices",
        lambda device: [torch.device("cuda:1")],
    )
    monkeypatch.setattr(
        gptqmodel,
        "get_device_thread_pool",
        lambda warmup_devices=None: captured.setdefault("warmup_devices", warmup_devices),
    )

    auto._prebuild_device_thread_pool(("cuda:1",))

    assert captured["warmup_devices"] == [torch.device("cuda:1")]


def _visible_cuda_pair():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return "0,1"
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if len(entries) < 2 or not all(entry.isdigit() for entry in entries[:2]):
        return None
    return ",".join(entries[:2])


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
