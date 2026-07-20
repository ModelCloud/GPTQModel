# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from gptqmodel.nn_modules.qlinear.marlin import (
    _packed_prefill_enabled,
    _should_use_packed_prefill,
)
from gptqmodel.utils import marlin as marlin_utils


def test_packed_prefill_auto_routing_is_on_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_MARLIN_PACKED_PREFILL", raising=False)
    assert _packed_prefill_enabled()

    monkeypatch.setenv("GPTQMODEL_MARLIN_PACKED_PREFILL", "0")
    assert not _packed_prefill_enabled()


def test_packed_prefill_dispatch_keeps_explicit_decode_on_marlin():
    assert not _should_use_packed_prefill(
        torch.ones((32, 1, 128), dtype=torch.float16),
        min_rows=1,
    )
    assert not _should_use_packed_prefill(
        torch.ones((1, 127, 128), dtype=torch.float16),
        min_rows=128,
    )
    assert _should_use_packed_prefill(
        torch.ones((1, 128, 128), dtype=torch.float16),
        min_rows=128,
    )


def test_packed_prefill_dispatch_uses_flattened_rows_for_2d_input():
    assert not _should_use_packed_prefill(
        torch.ones((127, 128), dtype=torch.bfloat16),
        min_rows=128,
    )
    assert _should_use_packed_prefill(
        torch.ones((128, 128), dtype=torch.bfloat16),
        min_rows=128,
    )


def test_packed_prefill_has_a_separate_generated_kernel_family():
    root = marlin_utils._ensure_generated_marlin_kernels()
    generator = (root / "generate_kernels.py").read_text(encoding="utf-8")
    kernel_h = (root / "kernel.h").read_text(encoding="utf-8")
    template_h = (root / "marlin_template.h").read_text(encoding="utf-8")
    gemm_cu = (root / "gptq_marlin.cu").read_text(encoding="utf-8")
    bf16_prefill = (root / "kernel_bf16_prefill_ku4b8.cu").read_text(encoding="utf-8")

    assert "PREFILL_CONFIGS" in generator
    assert "__global__ void MarlinPrefill" in kernel_h
    assert "#if MARLIN_DIRECT_PREFILL" in template_h
    assert "select_marlin_packed_prefill_config" in gemm_cu
    assert "q_type == vllm::kU4B8" in gemm_cu
    assert "group_size == 128" in gemm_cu
    assert bf16_prefill.count("template __global__ void MarlinPrefill<") == 4


def test_packed_prefill_source_does_not_define_a_dense_weight_cache():
    source = (
        marlin_utils._marlin_root().parents[1]
        / "gptqmodel"
        / "nn_modules"
        / "qlinear"
        / "marlin.py"
    ).read_text(encoding="utf-8")

    assert "dense_prefill" not in source.lower()
    assert "_dense_prefill_weight" not in source
