# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Public Pangolin runtime API for external inference engines."""

import gptqmodel.nn_modules.triton_utils.planar as _planar
import gptqmodel.utils.pangolin as _pangolin


class PangolinAPI:
    """Stable facade for Pangolin and its planar fallback kernels."""

    __slots__ = ()

    bits = _pangolin.PANGOLIN_BITS
    max_m = _pangolin.PANGOLIN_MAX_M
    supported_m = _pangolin.PANGOLIN_SUPPORTED_M
    planar_fused_max_m = _planar.PLANAR_FUSED_MAX_M
    planar_gemv_max_m = _planar.PLANAR_GEMV_MAX_M

    supported = staticmethod(_pangolin.pangolin_supported)
    runtime_available = staticmethod(_pangolin.pangolin_runtime_available)
    ensure_runtime_available = staticmethod(_pangolin.ensure_pangolin_runtime_available)
    runtime_error = staticmethod(_pangolin.pangolin_runtime_error)

    cpu_supported = staticmethod(_pangolin.pangolin_cpu_supported)
    cpu_runtime_available = staticmethod(_pangolin.pangolin_cpu_runtime_available)
    ensure_cpu_runtime_available = staticmethod(
        _pangolin.ensure_pangolin_cpu_runtime_available
    )
    cpu_runtime_error = staticmethod(_pangolin.pangolin_cpu_runtime_error)

    g_idx_block_uniform = staticmethod(_pangolin.g_idx_block_uniform)
    gemv = staticmethod(_pangolin.pangolin_gemv)
    dequant = staticmethod(_planar.planar_dequant)
    planar_gemv = staticmethod(_planar.planar_gemv)
    planar_matmul = staticmethod(_planar.planar_matmul)


__all__ = ["PangolinAPI"]
