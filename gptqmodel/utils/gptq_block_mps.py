# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native Metal implementation of the quantization-time GPTQ block."""

from __future__ import annotations

import math
import sys
import threading
from operator import index
from typing import Any

import torch
from torch import Tensor

_MAX_BLOCK_COLUMNS = 128
_MAX_BLOCK_ROWS = 2**31 - 1
_MAX_BLOCK_ELEMENTS = 2**32 - 1
_MAX_QUANTIZED_CODE = 2**8 - 1
_MAX_FLOAT32 = torch.finfo(torch.float32).max
_MPS_SHADER_LIBRARY: Any | None = None
_MPS_SHADER_ERROR: str | None = None
_MPS_SHADER_INIT_LOCK = threading.Lock()

_MPS_SHADER_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

kernel void gptq_block(
    device float* weights [[buffer(0)]],
    device const float* hessian [[buffer(1)]],
    device float* scales [[buffer(2)]],
    device float* zeros [[buffer(3)]],
    device float* output [[buffer(4)]],
    device float* errors [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& columns [[buffer(7)]],
    constant uint& scale_columns [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    constant int& maxq [[buffer(10)]],
    constant uint& groupwise [[buffer(11)]],
    constant uint& find_params [[buffer(12)]],
    constant uint& symmetric [[buffer(13)]],
    device const float* importance [[buffer(14)]],
    constant uint& scale_search [[buffer(15)]],
    constant uint& candidate_count [[buffer(16)]],
    constant uint& grid [[buffer(17)]],
    constant float& mse [[buffer(18)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= rows) {
    return;
  }

  const uint base = row * columns;
  const uint scale_base = row * scale_columns;
  if (find_params != 0) {
    for (uint group = 0; group < scale_columns; ++group) {
      float xmin = 0.0f;
      float xmax = 0.0f;
      bool has_nan = false;
      bool has_nonfinite = false;
      const uint group_start = base + group * group_size;
      for (uint offset = 0; offset < group_size; ++offset) {
        const float value = weights[group_start + offset];
        has_nan = has_nan || isnan(value);
        has_nonfinite = has_nonfinite || !isfinite(value);
        xmin = min(xmin, value);
        xmax = max(xmax, value);
      }
      if (has_nan) {
        xmin = NAN;
        xmax = NAN;
      }
      if (symmetric != 0) {
        xmax = max(abs(xmin), xmax);
        if (xmin < 0.0f) {
          xmin = -xmax;
        }
      }
      if (xmin == 0.0f && xmax == 0.0f) {
        xmin = -1.0f;
        xmax = 1.0f;
      }

      if (has_nonfinite) {
        if (groupwise != 0) {
          scales[scale_base + group] = xmax / float(maxq);
          zeros[scale_base + group] = 0.0f;
        } else {
          const float scale = (xmax - xmin) / float(maxq);
          scales[scale_base + group] = scale;
          zeros[scale_base + group] = symmetric != 0
              ? (float(maxq) + 1.0f) * 0.5f
              : rint(-xmin / scale);
        }
        continue;
      }

      float best_scale = 0.0f;
      float best_zero = 0.0f;
      float best_loss = INFINITY;
      const uint evaluated_candidates = scale_search == 0 ? 1 : candidate_count;
      float importance_mean = 1.0f;
      bool use_unit_importance = false;
      if (scale_search == 2) {
        importance_mean = 0.0f;
        for (uint offset = 0; offset < group_size; ++offset) {
          const float value = importance[group * group_size + offset];
          importance_mean += isfinite(value) ? max(value, 0.0f) : 0.0f;
        }
        importance_mean /= float(group_size);
        if (!isfinite(importance_mean) || importance_mean <= 0.0f) {
          importance_mean = 1.0f;
          use_unit_importance = true;
        }
      }

      // Evaluate a small candidate tile together. Each lane still accumulates
      // offsets in the original order and candidates are compared in their
      // original order, preserving exact loss and first-tie semantics while
      // loading each weight (and importance value) only once per eight candidates.
      for (uint candidate_base = 0; candidate_base < evaluated_candidates; candidate_base += 8) {
        float candidate_scale[8];
        float candidate_zero[8];
        float candidate_loss[8] = {0.0f};
        for (uint lane = 0; lane < 8; ++lane) {
          const uint candidate = candidate_base + lane;
          if (candidate < evaluated_candidates) {
            const float shrink = scale_search == 0 ? 1.0f : 1.0f - float(candidate) / float(grid);
            const float candidate_xmin = xmin * shrink;
            const float candidate_xmax = xmax * shrink;
            if (groupwise != 0) {
              candidate_scale[lane] = candidate_xmax / float(maxq);
              candidate_zero[lane] = 0.0f;
            } else {
              candidate_scale[lane] = (candidate_xmax - candidate_xmin) / float(maxq);
              candidate_zero[lane] = symmetric != 0
                  ? (float(maxq) + 1.0f) * 0.5f
                  : rint(-candidate_xmin / candidate_scale[lane]);
            }
          }
        }

        if (scale_search != 0) {
          for (uint offset = 0; offset < group_size; ++offset) {
            const float value = weights[group_start + offset];
            const float raw_importance = scale_search == 2
                ? importance[group * group_size + offset]
                : 1.0f;
            const float normalized_importance = scale_search == 2
                ? (use_unit_importance
                    ? 1.0f
                    : (isfinite(raw_importance) ? max(raw_importance, 0.0f) / importance_mean : 0.0f))
                : 1.0f;
            for (uint lane = 0; lane < 8; ++lane) {
              if (candidate_base + lane < evaluated_candidates) {
                const float scale = candidate_scale[lane];
                const float zero = candidate_zero[lane];
                float reconstructed;
                if (groupwise != 0) {
                  reconstructed = scale * clamp(rint(value / scale), -float(maxq), float(maxq));
                } else {
                  reconstructed = scale * (clamp(rint(value / scale) + zero, 0.0f, float(maxq)) - zero);
                }
                const float error = abs(reconstructed - value);
                candidate_loss[lane] += scale_search == 2
                    ? error * error * normalized_importance
                    : (mse == 2.0f ? error * error : pow(error, mse));
              }
            }
          }
        }
        for (uint lane = 0; lane < 8; ++lane) {
          if (candidate_base + lane < evaluated_candidates && candidate_loss[lane] < best_loss) {
            best_loss = candidate_loss[lane];
            best_scale = candidate_scale[lane];
            best_zero = candidate_zero[lane];
          }
        }
      }
      scales[scale_base + group] = best_scale;
      zeros[scale_base + group] = best_zero;
    }
  }

  for (uint column = 0; column < columns; ++column) {
    const uint group = min(column / group_size, scale_columns - 1);
    const float scale = scales[scale_base + group];
    const float zero = zeros[scale_base + group];
    const float weight = weights[base + column];
    float quantized;

    if (groupwise != 0) {
      quantized = scale * clamp(rint(weight / scale), -float(maxq), float(maxq));
    } else {
      quantized = scale * (clamp(rint(weight / scale) + zero, 0.0f, float(maxq)) - zero);
    }

    output[base + column] = quantized;
    const float error = (weight - quantized) / hessian[column * columns + column];
    errors[base + column] = error;
    for (uint update_column = column; update_column < columns; ++update_column) {
      weights[base + update_column] -= error * hessian[column * columns + update_column];
    }
  }
}
"""


def gptq_block_mps_supported() -> bool:
    """Return whether this PyTorch build exposes runtime Metal shaders."""
    return (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and callable(getattr(torch.mps, "compile_shader", None))
    )


def _shader_library():
    """Compile one immutable library; launches do not share mutable state."""
    global _MPS_SHADER_ERROR, _MPS_SHADER_LIBRARY
    if _MPS_SHADER_LIBRARY is None:
        with _MPS_SHADER_INIT_LOCK:
            if _MPS_SHADER_LIBRARY is None:
                if _MPS_SHADER_ERROR is not None:
                    raise RuntimeError(_MPS_SHADER_ERROR)
                if not gptq_block_mps_supported():
                    raise RuntimeError(
                        "MPS GPTQ block quantization requires macOS and torch.mps.compile_shader"
                    )
                try:
                    _MPS_SHADER_LIBRARY = torch.mps.compile_shader(_MPS_SHADER_SOURCE)
                except Exception as exc:
                    _MPS_SHADER_ERROR = f"Metal GPTQ shader compilation failed: {exc}"
                    raise RuntimeError(_MPS_SHADER_ERROR) from exc
    return _MPS_SHADER_LIBRARY


def _integer_argument(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from exc


def _storage_pointer(tensor: Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def gptq_block_mps(
    weights: Tensor,
    hessian_inverse: Tensor,
    scale: Tensor,
    zero: Tensor,
    maxq: int,
    group_size: int,
    *,
    groupwise: bool = False,
    find_params: bool = False,
    symmetric: bool = False,
    scale_search: str = "none",
    importance: Tensor | None = None,
    candidate_count: int = 0,
    grid: int = 100,
    mse: float = 2.0,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """Quantize one serial GPTQ column block with one Metal thread per row.

    The kernel mutates ``weights`` with the same sequential inverse-Hessian
    correction used by the eager implementation. No caller-owned tensor is
    retained after the asynchronous launch.
    """
    maxq = _integer_argument("maxq", maxq)
    group_size = _integer_argument("group_size", group_size)
    if not isinstance(groupwise, bool):
        raise TypeError(f"groupwise must be a bool, got {type(groupwise).__name__}")
    if not isinstance(find_params, bool):
        raise TypeError(f"find_params must be a bool, got {type(find_params).__name__}")
    if not isinstance(symmetric, bool):
        raise TypeError(f"symmetric must be a bool, got {type(symmetric).__name__}")
    scale_search_modes = {"none": 0, "mse": 1, "activation": 2}
    if scale_search not in scale_search_modes:
        raise ValueError(
            f"scale_search must be one of {tuple(scale_search_modes)}, got {scale_search!r}"
        )
    scale_search_value = scale_search_modes[scale_search]
    candidate_count = _integer_argument("candidate_count", candidate_count)
    grid = _integer_argument("grid", grid)
    try:
        mse = float(mse)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"mse must be a number, got {type(mse).__name__}") from exc
    if scale_search_value and not find_params:
        raise ValueError("scale_search requires find_params=True")
    if scale_search_value and candidate_count <= 0:
        raise ValueError(
            f"candidate_count must be positive for scale search, got {candidate_count}"
        )
    if candidate_count > _MAX_BLOCK_ELEMENTS:
        raise ValueError(
            f"candidate_count must be <= {_MAX_BLOCK_ELEMENTS}, got {candidate_count}"
        )
    if scale_search_value and grid <= 0:
        raise ValueError(f"grid must be positive for scale search, got {grid}")
    if grid > _MAX_BLOCK_ELEMENTS:
        raise ValueError(f"grid must be <= {_MAX_BLOCK_ELEMENTS}, got {grid}")
    if scale_search_value == 1 and (
        not math.isfinite(mse) or mse <= 0 or mse > _MAX_FLOAT32
    ):
        raise ValueError(
            f"mse must be finite and positive for MSE scale search, got {mse}"
        )
    if weights.ndim != 2:
        raise ValueError(
            f"weights must be two-dimensional, got shape {tuple(weights.shape)}"
        )

    rows, count = weights.shape
    if rows <= 0 or count <= 0:
        raise ValueError(
            f"weights dimensions must be positive, got shape {tuple(weights.shape)}"
        )
    if rows > _MAX_BLOCK_ROWS:
        raise ValueError(
            f"MPS block kernel supports rows <= {_MAX_BLOCK_ROWS}, got {rows}"
        )
    if count > _MAX_BLOCK_COLUMNS:
        raise ValueError(
            f"MPS block kernel supports count <= {_MAX_BLOCK_COLUMNS}, got {count}"
        )
    if rows * count > _MAX_BLOCK_ELEMENTS:
        raise ValueError(
            f"MPS block kernel supports rows * count <= {_MAX_BLOCK_ELEMENTS}, got {rows * count}"
        )
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if count % group_size != 0:
        raise ValueError(f"group_size {group_size} must divide count {count}")
    if maxq <= 0:
        raise ValueError(f"maxq must be positive, got {maxq}")
    if maxq > _MAX_QUANTIZED_CODE:
        raise ValueError(
            f"MPS block kernel supports maxq <= {_MAX_QUANTIZED_CODE}, got {maxq}"
        )

    operands = (weights, hessian_inverse, scale, zero)
    if any(tensor.dtype != torch.float32 for tensor in operands):
        raise TypeError("MPS GPTQ block kernel expects float32 weights/Hinv/scale/zero")
    devices = {tensor.device for tensor in operands}
    if len(devices) != 1:
        raise ValueError(
            f"MPS GPTQ block tensors must share one device, got {sorted(map(str, devices))}"
        )
    if weights.device.type != "mps":
        raise ValueError(
            f"MPS GPTQ block tensors must be MPS tensors, got {weights.device}"
        )
    if any(not tensor.is_contiguous() for tensor in operands):
        raise ValueError("MPS GPTQ block input tensors must be contiguous")
    if hessian_inverse.shape != (count, count):
        raise ValueError(
            f"hessian_inverse must have shape {(count, count)}, got {tuple(hessian_inverse.shape)}"
        )
    expected_scale_shape = (rows, count // group_size)
    if scale.shape != expected_scale_shape or zero.shape != expected_scale_shape:
        raise ValueError(
            f"scale/zero must have shape {expected_scale_shape}, got {tuple(scale.shape)}/{tuple(zero.shape)}"
        )
    if scale_search_value == 2:
        if importance is None:
            raise ValueError("activation scale search requires importance")
        if importance.shape != (expected_scale_shape[1], group_size):
            raise ValueError(
                "importance must have shape "
                f"{(expected_scale_shape[1], group_size)}, got {tuple(importance.shape)}"
            )
        if importance.dtype != torch.float32:
            raise TypeError("MPS GPTQ block importance must have dtype float32")
        if importance.device != weights.device:
            raise ValueError("MPS GPTQ block importance must share the weights device")
        if not importance.is_contiguous():
            raise ValueError("MPS GPTQ block importance must be contiguous")
    else:
        importance = hessian_inverse
    weights_storage = _storage_pointer(weights)
    if scale_search_value == 2 and _storage_pointer(importance) == weights_storage:
        raise ValueError(
            "MPS GPTQ block activation importance must not alias mutable weights"
        )
    if weights_storage in {
        _storage_pointer(hessian_inverse),
        _storage_pointer(scale),
        _storage_pointer(zero),
    }:
        raise ValueError(
            "mutable MPS GPTQ block weights must not alias Hinv/scale/zero"
        )
    if find_params and _storage_pointer(scale) == _storage_pointer(zero):
        raise ValueError("writable MPS GPTQ block scale/zero tensors must not alias")
    if find_params:
        read_only_storage = {
            weights_storage,
            _storage_pointer(hessian_inverse),
            _storage_pointer(importance),
        }
        if {_storage_pointer(scale), _storage_pointer(zero)}.intersection(
            read_only_storage
        ):
            raise ValueError(
                "writable MPS GPTQ block scale/zero tensors must not alias weights/Hinv/importance"
            )

    if out is None:
        quantized = torch.empty_like(weights, memory_format=torch.contiguous_format)
        errors = torch.empty_like(weights, memory_format=torch.contiguous_format)
    else:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("out must be a (quantized, errors) tuple")
        quantized, errors = out
        if quantized.shape != weights.shape or errors.shape != weights.shape:
            raise ValueError(
                f"out tensors must match weights shape {tuple(weights.shape)}, got "
                f"{tuple(quantized.shape)}/{tuple(errors.shape)}"
            )
        if quantized.dtype != torch.float32 or errors.dtype != torch.float32:
            raise TypeError("MPS GPTQ block out tensors must have dtype float32")
        if quantized.device != weights.device or errors.device != weights.device:
            raise ValueError("MPS GPTQ block out tensors must share the weights device")
        if not quantized.is_contiguous() or not errors.is_contiguous():
            raise ValueError("MPS GPTQ block out tensors must be contiguous")

    output_storage = {_storage_pointer(quantized), _storage_pointer(errors)}
    input_storage = {_storage_pointer(tensor) for tensor in (*operands, importance)}
    if len(output_storage) != 2 or output_storage.intersection(input_storage):
        raise ValueError(
            "MPS GPTQ block out tensors must not alias each other or any input"
        )

    library = _shader_library()
    library.gptq_block(
        weights,
        hessian_inverse,
        scale,
        zero,
        quantized,
        errors,
        rows,
        count,
        expected_scale_shape[1],
        group_size,
        maxq,
        int(groupwise),
        int(find_params),
        int(symmetric),
        importance,
        scale_search_value,
        candidate_count,
        grid,
        mse,
        threads=rows,
    )
    return quantized, errors


__all__ = ["gptq_block_mps", "gptq_block_mps_supported"]
