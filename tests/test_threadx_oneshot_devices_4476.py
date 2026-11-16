# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""``DeviceThreadPool`` must accept one-shot device iterables (#4476).

The constructor declares ``devices: Iterable[DeviceLike]`` and traverses it
several times: affinity planning, the default intra-op budget, and worker
registration. A generator was therefore exhausted by the first pass, the pool
registered no workers, and ``submit("cpu", ...)`` raised
``ValueError: Device not in pool: cpu``. A reusable list worked, so the
supported input type silently behaved differently depending on whether the
caller passed a list or a generator.
"""

import torch

from gptqmodel.utils.threadx import DeviceThreadPool


def _run(pool, device: str = "cpu"):
    return pool.submit(device, lambda: 42).result(timeout=30)


def test_generator_devices_register_workers_like_a_list():
    list_pool = DeviceThreadPool(devices=[torch.device("cpu")], empty_cache_every_n=0)
    gen_pool = DeviceThreadPool(
        devices=(d for d in [torch.device("cpu")]),
        empty_cache_every_n=0,
    )
    try:
        assert sorted(list_pool._devices_by_key) == ["cpu"]
        assert sorted(gen_pool._devices_by_key) == ["cpu"]

        assert _run(list_pool) == 42
        assert _run(gen_pool) == 42
    finally:
        list_pool.shutdown()
        gen_pool.shutdown()


def test_generator_devices_survive_repeated_traversal():
    """Duplicate entries must still collapse to one registry key."""

    pool = DeviceThreadPool(
        devices=(d for d in [torch.device("cpu"), torch.device("cpu")]),
        empty_cache_every_n=0,
    )
    try:
        assert sorted(pool._devices_by_key) == ["cpu"]
        assert _run(pool) == 42
    finally:
        pool.shutdown()


def test_generator_devices_with_explicit_worker_count():
    pool = DeviceThreadPool(
        devices=(d for d in [torch.device("cpu")]),
        workers={"cpu": 2},
        empty_cache_every_n=0,
    )
    try:
        assert len(pool._worker_groups["cpu"]) == 2
        assert _run(pool) == 42
    finally:
        pool.shutdown()
