# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reference cold-build timings for torch.ops JIT extensions.

These values were measured on 2026-04-04 from clean temporary build roots on
the reference CUDA development host with ``MAX_JOBS`` unset. They are not used
for correctness and only provide a more realistic first-use progress estimate
than an open-ended spinner.

When a kernel changes materially, refresh the corresponding value by re-timing
its clean JIT build.
"""

from __future__ import annotations


JIT_COMPILE_BASELINE_SECONDS: dict[str, float] = {
    "gptqmodel_awq_ops": 61.640,
    "gptqmodel_exllamav2_awq_ops": 35.421,
    "gptqmodel_exllamav2_ops": 34.528,
    "gptqmodel_exllamav3_ops": 61.871,
    "gptqmodel_marlin_bf16_ops": 120.634,
    "gptqmodel_marlin_fp16_ops": 116.863,
    "gptqmodel_pack_block_cpu": 31.096,
    "gptqmodel_paroquant_rotation": 78.430,
    "gptqmodel_qqq_ops": 82.492,
}


def get_jit_compile_baseline_seconds(extension_name: str) -> float | None:
    """Return the recorded reference build duration for one JIT extension."""

    value = JIT_COMPILE_BASELINE_SECONDS.get(extension_name)
    if value is None:
        return None
    return float(value)
