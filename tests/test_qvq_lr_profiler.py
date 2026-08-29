# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import Namespace

import pytest

from scripts import profile_qvq_lr_kernels


def _args(**overrides) -> Namespace:
    values = {
        "m": 1,
        "k": 2048,
        "n": 256,
        "warmup": 10,
        "iterations": 20,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"m": 0}, "--m must be positive"),
        ({"k": 0}, "--k and --n must be positive"),
        ({"n": 24}, "must be divisible"),
        ({"warmup": -1}, "--warmup must be non-negative"),
        ({"iterations": 0}, "--iterations must be positive"),
    ),
)
def test_profiler_rejects_empty_or_invalid_capture(overrides, message):
    with pytest.raises(ValueError, match=message):
        profile_qvq_lr_kernels._validate_args(_args(**overrides))


def test_profiler_surfaces_cuda_runtime_errors(monkeypatch):
    class Runtime:
        @staticmethod
        def cudaProfilerStart():
            return 7

    monkeypatch.setattr(profile_qvq_lr_kernels.torch.cuda, "cudart", lambda: Runtime())

    with pytest.raises(RuntimeError, match="cudaProfilerStart failed with CUDA status 7"):
        profile_qvq_lr_kernels._cuda_profiler_call("cudaProfilerStart")
