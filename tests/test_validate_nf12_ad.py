# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Validation CLI backend selection contracts."""

import sys

import pytest


def test_explicit_cuda_backend_reports_unavailable_native_kernel(monkeypatch):
    from gptqmodel.utils import nf12_ad
    from scripts import validate_nf12_ad

    monkeypatch.setattr(nf12_ad, "load", lambda: False)
    monkeypatch.setattr(nf12_ad, "last_error", lambda: "nvcc unavailable")

    with pytest.raises(RuntimeError, match="nvcc unavailable"):
        validate_nf12_ad.backend_module("cuda", "cuda:0")


def test_perf_only_failure_returns_nonzero_status(monkeypatch):
    from scripts import validate_nf12_ad

    def fail_perf(*args):
        raise RuntimeError("native backend unavailable")

    monkeypatch.setattr(validate_nf12_ad, "run_perf", fail_perf)
    monkeypatch.setattr(validate_nf12_ad, "print_reports", lambda results: None)
    monkeypatch.setattr(sys, "argv", ["validate_nf12_ad", "--backend", "cuda", "--perf-only"])

    assert validate_nf12_ad.main() == 1
