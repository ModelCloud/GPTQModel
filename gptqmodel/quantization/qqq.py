# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math
import platform
import sys
import threading
import time
from functools import lru_cache
from typing import Dict, Optional

import torch
import transformers
from torch import nn
from torch.nn import functional as F

from .. import QuantizeConfig
from ..looper.named_module import NamedModule
from ..quantization.config import FallbackStrategy, MSEConfig, SmoothMSE
from ..quantization.quantizer import HF_OPTIMUM, _mse_shrink_grid, _neutralize_dead_columns
from ..utils import setup_logger
from ..utils.device import get_device
from ..utils.kernel import resolve_kernel_config
from ..utils.p2p import safe_move_to
from .conv_padding import conv_padding_pairs, pad_mode
from .fallback_smooth import (
    affine_range_with_zero,
    element_saliency_weights,
    hessian_column_weights,
    mse_optimal_quant,
    smooth_applies,
    smooth_block,
    symmetric_range_from_bounds,
)
from .gptq import (
    _is_memory_allocation_failure,
    _move_static_groups,
    _propagate_block_error,
    _uniform_column_group,
    get_number_of_rows_and_cols,
    recover_damped_hessian_factor,
)


DEBUG = False

log = setup_logger()

_MLX_QQQ_MIN_ELEMENTS = 4_194_304


@lru_cache(maxsize=1)
def _mlx_qqq_quantization_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_array_to_torch(value, device: torch.device) -> torch.Tensor:
    return torch.from_dlpack(value).to(device=device).contiguous()


def _qqq_quantize_weight_mlx_to_torch(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    *,
    group_size: int,
):
    import mlx.core as mx

    from .mlx_qqq import qqq_quantize_weight_mlx_with_loss

    if weight.device.type == "mps":
        torch.mps.synchronize()
    result = qqq_quantize_weight_mlx_with_loss(
        mx.from_dlpack(weight.contiguous()),
        mx.from_dlpack(inverse_hessian.contiguous()),
        group_size=group_size,
    )
    return tuple(
        None if value is None else _mlx_array_to_torch(value, weight.device)
        for value in result
    )


def _should_use_mlx_qqq_quantization(
    weight: torch.Tensor,
    inverse_hessian: Optional[torch.Tensor],
    quantizer,
    *,
    group_size: int,
    static_groups: bool,
    blocksize: int,
) -> bool:
    expected_maxq = 7 if group_size == -1 else 15
    return (
        not DEBUG
        and inverse_hessian is not None
        and _mlx_qqq_quantization_available()
        and weight.device.type in {"cpu", "mps"}
        and inverse_hessian.device == weight.device
        and weight.numel() >= _MLX_QQQ_MIN_ELEMENTS
        and blocksize == 128
        and group_size in {-1, 128}
        and (group_size == -1 or weight.shape[1] % group_size == 0)
        and not static_groups
        and bool(quantizer.sym)
        and not bool(quantizer.mse)
        and int(quantizer.groupsize) == group_size
        and quantizer._maxq_value == expected_maxq
    )


def quantize(x, scale, zero, maxq, sym, groupsize):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    if groupsize != -1 or not sym:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)
    else:
        q = torch.clamp(torch.round(x / scale), -maxq, maxq)
        return scale * q


class Quantizer(nn.Module):
    def __init__(self, qcfg: Optional[QuantizeConfig] = None, shape=1):
        super(Quantizer, self).__init__()
        self.qcfg = qcfg
        self.register_buffer("maxq", torch.tensor(0))
        # Host mirror of the ``maxq`` buffer. ``configure()`` is the only writer,
        # so the block loop avoids a device-to-host scalar read per column.
        self._maxq_value = 0
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))
        # Host mirror of ``ready()``. ``find_params`` is the only writer of the
        # scale buffer, so callers in the quantization loop can avoid a
        # device-to-host reduction before every quantize call.
        self._params_ready = False

    def configure(
            self,
            bits,
            perchannel=False,
            sym=True,
            groupsize=-1,
            mse=False,
            norm=2.4,
            grid=100,
            maxshrink=0.8,
            trits=False,
    ):
        if groupsize != -1 or not sym:
            self.maxq = torch.tensor(2 ** bits - 1)
        else:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        self.perchannel = perchannel
        self.groupsize = groupsize
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self._maxq_value = int(self.maxq)

    def find_params(self, x, weight=False):
        dev = x.device
        if self.maxq.device != dev:
            self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            # The symmetric zero point below is fixed at ``(maxq + 1) // 2``,
            # so the range must be zero-centered for every group.  Mirroring
            # only ``xmin < 0`` left all-positive groups fitted as ``[0, max]``
            # and saturated the positive half of the grid at 7/15 of the
            # maximum for INT4.
            xmax = torch.maximum(torch.abs(xmin), xmax)
            xmin = -xmax
        both_zero = (xmin == 0) & (xmax == 0)
        xmin = torch.where(both_zero, xmin - 1, xmin)
        xmax = torch.where(both_zero, xmax + 1, xmax)

        if self._maxq_value < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            if self.groupsize != -1 or not self.sym:
                self.scale = (xmax - xmin) / self.maxq
                if self.sym:
                    self.zero = torch.full_like(self.scale, (self._maxq_value + 1) / 2)
                else:
                    self.zero = torch.round(-xmin / self.scale)
            else:
                self.scale = xmax / self.maxq
                self.zero = torch.zeros_like(self.scale)

        if self.mse:
            steps = int(self.maxshrink * self.grid)
            if steps > 0:
                groupwise = self.groupsize == -1 and self.sym
                # ``1 - i/grid`` in host float, rounded to ``xmin``'s dtype,
                # reproduces the values the per-step loop got from ``p * xmin``.
                p = torch.tensor(
                    [1 - i / self.grid for i in range(steps)],
                    dtype=xmin.dtype,
                    device=dev,
                ).unsqueeze(1)
                xmin1 = p * xmin.unsqueeze(0)
                xmax1 = p * xmax.unsqueeze(0)
                if self._maxq_value < 0:
                    # Ternary levels are stored directly in scale/zero and
                    # ``maxq == -1`` is a sentinel, so the affine grid would be
                    # negative. Shrink the extrema themselves and keep the
                    # unshrunk endpoints as the incumbent (#5601).
                    scale1 = xmax1
                    zero1 = xmin1
                else:
                    scale1 = (
                        (xmax1 - xmin1) / self.maxq
                        if not groupwise
                        else xmax1 / self.maxq
                    )
                    zero1 = (
                        torch.round(-xmin1 / scale1)
                        if not self.sym
                        else self.zero.unsqueeze(0).expand(steps, x.shape[0])
                    )
                self.scale, self.zero = _mse_shrink_grid(
                    x,
                    scale1,
                    zero1,
                    float(self.norm),
                    maxq_value=self._maxq_value,
                    groupwise=groupwise,
                    # QQQ's ``quantize`` never guarded zero scales; keep that.
                    guard_zero_scale=False,
                    scale=self.scale,
                    zero=self.zero,
                    grid_budget=(
                        self.qcfg.mse.grid_budget
                        if isinstance(getattr(self.qcfg, "mse", None), MSEConfig)
                        else None
                    ),
                )
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        self._params_ready = True

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(
                x, self.scale, self.zero, self._maxq_value, self.sym, self.groupsize
            )
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return self._params_ready


class QQQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None):
        # self.lock = threading.Lock()

        # self.num_tied_handles = 0
        # if qcfg.tied_gptq_handle is not None:
        #     qcfg.tied_gptq_handle.num_tied_handles += 1

        # Flags indicating issues
        # self.issue_zero_samples = False
        # self.issue_nan_hessian = False
        # self.issue_non_invertible = False

        # self.W = module.weight
        self.rows, self.columns = get_number_of_rows_and_cols(module)
        if isinstance(module, NamedModule):
            self.layer = module.module
            self.name = module.name
            self._named_module = module
        else:
            self.name = HF_OPTIMUM
            self.layer = module
            self._named_module = None

        layer_device = self.layer.weight.device
        setattr(self.layer, "target_device", layer_device)

        self.dev = self.layer.weight.device

        self._validate_module(self.layer)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg

        self._original_rows = self.rows
        self._original_columns = self.columns
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.layer, "_tp_pad_info", None)

        if isinstance(pad_info, dict):
            pad_cols = int(pad_info.get("pad_cols", 0) or 0)
            pad_cols = max(pad_cols, 0)
        else:
            pad_cols = 0
        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols

        self.module_copy = None

        self.nsamples = 0

        self.quantizer = Quantizer(qcfg=self.qcfg)

        # fwd counter
        self.fwd_counter = 0

        self.fallback = self.qcfg.fallback
        self.expected_nsamples: Optional[float] = None

        self.lock = threading.Lock()
        self.H: Optional[torch.Tensor] = None
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        # Samples already folded into `H` by a previous materialization.
        self._materialized_samples = 0
        self._gsq_moments = {}
        from .gsq_scalar import gsq_enabled_for

        full_name = self._named_module.full_name if self._named_module is not None else self.name
        self._gsq_enabled = gsq_enabled_for(getattr(self.qcfg, "gsq", None), full_name)
        if self._gsq_enabled and (not isinstance(self.layer, nn.Linear) or self._tp_pad_cols):
            raise ValueError("QQQ GSQ requires an unpadded Linear module")
        if self._gsq_enabled and self.qcfg.gsq.learn_scales:
            raise ValueError("QQQ GSQ supports fixed scales only")

    @staticmethod
    def _validate_module(module):
        assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                   transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"
        if isinstance(module, (nn.Conv1d, nn.Conv2d)) and module.groups != 1:
            raise ValueError(
                "QQQ does not support grouped convolution capture: "
                f"`{type(module).__name__}` has groups={module.groups}, but the "
                "Hessian is built from the full unfolded input while the weight "
                "width is per-group. Exclude grouped convolutions from QQQ "
                "quantization or use an ungrouped equivalent."
            )
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            # Resolve the padding contract once, at validation, so an
            # unsupported setting fails before calibration instead of silently
            # capturing a Hessian for a different operator.
            conv_padding_pairs(module)

    @staticmethod
    def _truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor

        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor

        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    def _fallback_quantize(
        self,
        strategy: FallbackStrategy,
        hessian_diag: Optional[torch.Tensor] = None,
    ):
        maxq = 2 ** self.qcfg.bits - 1
        sigma = 3.0
        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        start_time = time.time()
        smooth_method = getattr(self.fallback, "smooth", None)
        search_mse = isinstance(smooth_method, SmoothMSE) and smooth_applies(
            smooth_method, group_size
        )
        mse_steps = 32
        mse_maxshrink = 0.8
        if isinstance(smooth_method, SmoothMSE):
            mse_steps = smooth_method.steps
            mse_maxshrink = smooth_method.maxshrink

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d)):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if self._tp_pad_cols:
            pad = torch.zeros(
                (W.shape[0], self._tp_pad_cols),
                dtype=W.dtype,
                device=W.device,
            )
            W = torch.cat((W, pad), dim=1)
        # The Hessian diagonal carries the captured input-channel energy; use
        # it so SmoothMSE does not clip weights that feed outlier activations.
        # Without a Hessian, fall back to the weight-magnitude proxy rather
        # than the uniform mean that rewards clipping a row's largest weight.
        col_weights = None
        if search_mse:
            col_weights = hessian_column_weights(
                getattr(self, "H", None), self.columns, device=W.device
            )
            if col_weights is None:
                col_weights = hessian_column_weights(
                    hessian_diag, self.columns, device=W.device
                )
        Q = torch.empty_like(W)
        scale_chunks = []
        zero_chunks = []

        def _affine_quantize(block_mod, lo, hi):
            """Quantize one group against a range that includes zero.

            ``lo``/``hi`` are the strategy's desired endpoints. Extending them
            to include zero keeps the derived integer zero point inside
            ``[0, maxq]`` instead of clamping a positive-only group to zero and
            shifting every value by roughly the group minimum.
            """

            if self.qcfg.sym:
                # Every symmetric kernel assumes zero = 2 ** (bits - 1);
                # Marlin does not even read the stored qzeros. Zero-center the
                # interval the strategy selected so STDCLIP/MEAN/MEDIAN keep
                # their clipping decision instead of remeasuring the block.
                lo, hi = symmetric_range_from_bounds(lo, hi)
            else:
                lo, hi = affine_range_with_zero(lo, hi)
            scale = torch.clamp((hi - lo) / maxq, min=1e-8)
            if self.qcfg.sym:
                zero = torch.full_like(scale, (maxq + 1) // 2)
            else:
                zero = torch.clamp(torch.round(-lo / scale), 0, maxq)
            q = torch.clamp(torch.round(block_mod / scale + zero), 0, maxq)
            return (q - zero) * scale, scale, zero

        for start in range(0, self.columns, group_size):
            end = min(start + group_size, self.columns)
            block = W[:, start:end]

            if search_mse:
                weights = element_saliency_weights(block) if col_weights is None else col_weights[start:end]
                dequant, scale, zero = mse_optimal_quant(
                    block,
                    self.qcfg,
                    maxq,
                    steps=mse_steps,
                    maxshrink=mse_maxshrink,
                    weights=weights,
                )
            else:
                block_mod, scale_factor = smooth_block(
                    block,
                    self.fallback,
                    group_size=self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns,
                )
                if strategy == FallbackStrategy.MIDPOINT:
                    w_min = block_mod.min(dim=1, keepdim=True).values
                    w_max = block_mod.max(dim=1, keepdim=True).values
                    dequant, scale, zero = _affine_quantize(block_mod, w_min, w_max)
                elif strategy == FallbackStrategy.MEAN:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    max_dev = torch.max((block_mod - mean).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    dequant, scale, zero = _affine_quantize(
                        block_mod, mean - max_dev, mean + max_dev
                    )
                elif strategy == FallbackStrategy.MEDIAN:
                    median = block_mod.median(dim=1, keepdim=True).values
                    max_dev = torch.max((block_mod - median).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    dequant, scale, zero = _affine_quantize(
                        block_mod, median - max_dev, median + max_dev
                    )
                elif strategy == FallbackStrategy.STDCLIP:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    std = block_mod.std(dim=1, keepdim=True, unbiased=False)
                    std = torch.clamp(std, min=1e-8)
                    lo = mean - sigma * std
                    hi = mean + sigma * std
                    dequant, scale, zero = _affine_quantize(block_mod, lo, hi)
                elif strategy == FallbackStrategy.RTN:
                    self.quantizer.find_params(block_mod, weight=True)
                    dequant = self.quantizer.quantize(block_mod)
                    scale = self.quantizer.scale
                    zero = self.quantizer.zero
                else:
                    raise ValueError(f"Unsupported fallback strategy: {strategy}")

                if scale_factor is not None:
                    scale = scale * scale_factor
                    dequant = dequant * scale_factor

            Q[:, start:end] = dequant

            scale_block = scale if scale.dim() > 1 else scale.unsqueeze(1)
            zero_block = zero if zero.dim() > 1 else zero.unsqueeze(1)
            if scale_block.shape[1] > 1:
                scale_block = scale_block.mean(dim=1, keepdim=True)
            if zero_block.shape[1] > 1:
                zero_block = zero_block.mean(dim=1, keepdim=True)
            scale_chunks.append(scale_block)
            zero_chunks.append(zero_block)

        scale = torch.cat(scale_chunks, dim=1)
        zero = torch.cat(zero_chunks, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            scale = self._truncate_last_dim(scale, valid_cols)
            zero = self._truncate_last_dim(zero, valid_cols)
        else:
            valid_cols = self.columns

        g_idx = torch.arange(valid_cols, device=Q.device, dtype=torch.int32) // (group_size if group_size != -1 else self.columns)

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        if Q.shape != self.layer.weight.shape:
            Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
        else:
            Q = Q.to(self.layer.weight.dtype)

        scale_extra = None
        if group_size != self.columns:
            quantizer_extra = Quantizer(qcfg=self.qcfg)
            quantizer_extra.configure(
                bits=8,
                perchannel=True,
                groupsize=-1,
                sym=True,
                mse=False,
            )
            # The exported INT8 channel scale must describe the quantized
            # weight, not the original one: GPTQ error propagation can push
            # max|Q| above max|W|, and a scale fitted to W then saturates the
            # packed codes (and no longer reproduces the in-memory `wq`).
            # ``Q`` is back in the source ``Conv1D`` [in, out] orientation;
            # the per-channel scale belongs to the output axis.
            extra_weight = Q.t() if isinstance(self.layer, transformers.Conv1D) else Q
            quantizer_extra.find_params(extra_weight.clone(), weight=True)
            scale_extra = quantizer_extra.scale

        duration = time.time() - start_time
        mean_abs_err = (Q - self.layer.weight.data).abs().mean().item()
        avg_loss = f"fallback({strategy.value}): {mean_abs_err:.7f}"
        damp_percent = 0.0
        self.H = None
        return Q, scale, zero, g_idx, duration, avg_loss, damp_percent, scale_extra, self.nsamples

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if isinstance(self.layer, nn.Linear) or isinstance(
                self.layer, transformers.Conv1D
        ):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv1d):
            # nn.Unfold only accepts integer zero padding, so apply the
            # convolution's own padding (including its mode) first and unfold
            # the already-padded input. This keeps the Hessian describing the
            # operator that will actually run.
            (left, right), = conv_padding_pairs(self.layer)
            inp = F.pad(
                inp.unsqueeze(2),
                (left, right, 0, 0),
                mode=pad_mode(self.layer.padding_mode),
            )
            unfold = nn.Unfold(
                (1, self.layer.kernel_size[0]),
                dilation=(1, self.layer.dilation[0]),
                padding=0,
                stride=(1, self.layer.stride[0]),
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        elif isinstance(self.layer, nn.Conv2d):
            # `conv_padding_pairs` reports pairs in the module's spatial order
            # (height, width); `F.pad` wants the last dimension first.
            (top, bottom), (left, right) = conv_padding_pairs(self.layer)
            inp = F.pad(
                inp,
                (left, right, top, bottom),
                mode=pad_mode(self.layer.padding_mode),
            )
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=0,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        if self._gsq_enabled:
            from .gsq_qqq import qqq_calibration_moments

            hq, cross, count = qqq_calibration_moments(inp.t())
            with self.lock:
                previous = self._gsq_moments.get(inp.device)
                if previous is None:
                    self._gsq_moments[inp.device] = (hq, cross, count)
                else:
                    previous[0].add_(hq)
                    previous[1].add_(cross)
                    self._gsq_moments[inp.device] = (previous[0], previous[1], previous[2] + count)

        batch_token_size = inp.shape[1]
        if self.qcfg.mock_quantization and not self._gsq_enabled:
            # Mock quantization never consumes the Hessian, so skip the
            # ``columns**2`` accumulation buffer and its matmul. Only the sample
            # count survives, because it is reported with the quantized block
            # (#5839).
            with self.lock:
                self.fwd_counter += 1
                self.nsamples += batch_token_size
            return

        dev = torch.device(get_device(inp))
        inp = inp.float()
        if self._tp_pad_cols:
            pad = inp.new_zeros((self._tp_pad_cols, inp.shape[1]))
            inp = torch.cat((inp, pad), dim=0)

        with self.lock:
            self.fwd_counter += 1
            existing = self._device_hessian_partials.get(dev)
            if existing is None:
                existing = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=dev,
                )
                self._device_hessian_partials[dev] = existing

            # Accumulate xtx directly into the per-device buffer to avoid
            # allocating a separate output tensor for every batch.
            existing.addmm_(inp, inp.t(), beta=1.0, alpha=1.0)
            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_token_size
            self.nsamples += batch_token_size

    def materialize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        if target_device is None:
            target_device = self.dev
        target_device = torch.device(target_device)

        pending_samples = sum(self._device_sample_counts.values())
        total_samples = self._materialized_samples + pending_samples
        if total_samples == 0:
            self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=target_device)
            self.nsamples = 0
            return self.H

        # A repeat materialization with no new batches returns the existing
        # Hessian unchanged instead of clearing the statistics.
        if pending_samples == 0:
            if getattr(self, "H", None) is not None:
                self.nsamples = self._materialized_samples
                # The cached Hessian is a read-only view of the statistics, so
                # an explicit target device is honoured by moving the result
                # rather than by relocating the cached state.
                if self.H.device == target_device and self.H.dtype == torch.float32:
                    return self.H
                return safe_move_to(self.H, target_device, dtype=torch.float32)
            self._materialized_samples = 0
            self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=target_device)
            self.nsamples = 0
            return self.H

        # Merge per-device partials into a single Hessian on the target device.
        # Pop and delete each partial as it is added so the peak memory stays
        # at result + one partial instead of result + all partials.
        if self._materialized_samples and getattr(self, "H", None) is not None:
            # `H` stores `2 * sum(x^T x) / samples`, so scale it back to the
            # sample sum before adding the new partials.
            result = safe_move_to(self.H, target_device, dtype=torch.float32)
            result = result * (float(self._materialized_samples) * 0.5)
        else:
            self._materialized_samples = 0
            result = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=target_device)
        while self._device_hessian_partials:
            dev, partial = self._device_hessian_partials.popitem()
            if partial.device == target_device and partial.dtype == torch.float32:
                result.add_(partial)
            else:
                # Partials are keyed by the device that captured them, so this
                # is a cross-device move. On an asymmetric ROCm peer topology a
                # direct copy segfaults, so route it through the helper.
                result.add_(safe_move_to(partial, target_device, dtype=torch.float32))
            del partial

        result.mul_(2.0 / float(total_samples))
        self.H = result
        self.nsamples = total_samples
        self._materialized_samples = total_samples
        self._device_sample_counts.clear()
        return self.H

    def adopt_hessian_from(self, leader: "QQQ") -> None:
        """Replace this task's Hessian with a private copy of ``leader``'s.

        Both modules must consume the identical input tensor (see ``:in=<tag>``
        in ``module_tree``).  ``quantize`` mutates ``H`` in place, so the copy
        must be independent.  The leader's per-device partials are already
        folded into ``H`` by :meth:`materialize_hessian`, which keeps the peak
        at the same two matrices the independent path holds (leader ``H`` plus
        the follower's copy).
        """

        if leader is self:
            return
        if leader.columns != self.columns:
            raise ValueError(
                f"QQQ: cannot share Hessian from `{leader.name}` ({leader.columns} columns) "
                f"with `{self.name}` ({self.columns} columns)."
            )

        source = leader.materialize_hessian(self.dev)
        with leader.lock:
            nsamples = leader.nsamples
            fwd_counter = leader.fwd_counter
            materialized_samples = leader._materialized_samples
            copy = safe_move_to(
                source.detach(),
                self.dev,
                dtype=torch.float32,
                copy=True,
            )

        with self.lock:
            self.H = copy
            self.nsamples = nsamples
            self.fwd_counter = fwd_counter
            self._materialized_samples = materialized_samples
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()

    @torch.inference_mode()
    def quantize(self, blocksize=128):
        if not self._gsq_enabled:
            return self._quantize_impl(blocksize=blocksize)
        from .gsq_qqq import qqq_codes_to_packer_weight, refine_qqq_codes

        if not self._gsq_moments:
            raise ValueError("QQQ GSQ requires collected calibration moments")
        teacher = self.layer.weight.detach().float().clone()
        hessian = sum(safe_move_to(part[0], teacher.device) for part in self._gsq_moments.values())
        cross = sum(safe_move_to(part[1], teacher.device) for part in self._gsq_moments.values())
        result = list(self._quantize_impl(blocksize=blocksize))
        weight, scales = result[:2]
        width = weight.shape[1]
        group_size = self.qcfg.group_size
        resolved = width if group_size == -1 else group_size
        raw = scales[:, torch.arange(width, device=scales.device) // resolved]
        rounded = (weight / raw).round()
        if not torch.isfinite(rounded).all():
            raise ValueError("QQQ GSQ baseline inverse contains nonfinite codes")
        codes = ((rounded + 8).clamp(0, 15) if resolved != width
                 else rounded.clamp(-15, 15).remainder(16)).long()
        best, before, after, history = refine_qqq_codes(
            codes, scales, target=teacher, group_size=group_size,
            hessian=hessian, cross_moment=cross, channel_scales=result[7], config=self.qcfg.gsq)
        if after < before:
            result[0] = qqq_codes_to_packer_weight(best, scales, group_size=group_size, dtype=weight.dtype)
        self.gsq_diagnostics = {"before": before, "after": after, "history": history,
                                    "objective": "asymmetric_quadratic_without_constant"}
        self._gsq_moments.clear()
        return tuple(result)

    def _reject_unsupported_activation_order(self) -> None:
        """Fail before Hessian work when the config asks for act ordering.

        QQQ's packed runtime stores contiguous group scales and has no
        ``g_idx`` buffer, so ``desc_act=True`` cannot be represented in the
        checkpoint.
        """

        if self.qcfg.desc_act:
            raise ValueError(
                "QQQ quantization does not support `desc_act=True`; "
                "use `desc_act=False`."
            )

    @torch.inference_mode()
    def _quantize_impl(
            self,
            blocksize=128,
    ):
        start = time.time()
        self._reject_unsupported_activation_order()

        from ..utils.fallback import (
            resolve_fallback_strategy,
            resolve_threshold,
            should_use_fallback,
        )

        # ``mock_quantization`` exists to skip the heavy calibration math, so it
        # must not invert a Hessian or propagate errors. Select the
        # data-independent RTN grid before materializing anything and keep the
        # configured QQQ layout (#5839). The NaN-loss retry sets the same flag
        # and re-enters here, so it lands on this path too.
        if self.qcfg.mock_quantization:
            self.H = None
            self._materialized_samples = 0
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()
            return self._fallback_quantize(FallbackStrategy.RTN)

        resolved_strategy = resolve_fallback_strategy(self.fallback)

        # Consume the Hessian this pass asks for. A previous explicit
        # materialization may have left `self.H` on another device, and the
        # cached branch returns a device-corrected value without moving the
        # cache, so the return value is the one to quantize with.
        self.H = self.materialize_hessian(self.layer.weight.device)

        # No calibration rows reached this module (for example an unrouted
        # expert). A zero Hessian makes every column look calibration-dead, so
        # ``_neutralize_dead_columns`` would zero the whole weight and return an
        # all-zero module. Absence of evidence is not evidence that every column
        # is dead: use the data-independent RTN grid instead, matching GPTQ's
        # zero-sample guard.
        if self.nsamples == 0:
            resolved_strategy = FallbackStrategy.RTN
            log.warn(
                "Quantization: Module `%s` -> no calibration samples reached this module; "
                "forcing `%s` fallback quantization instead of quantizing a zero Hessian.",
                self.name,
                resolved_strategy.value,
            )
            self.H = None
            self._materialized_samples = 0
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()
            return self._fallback_quantize(resolved_strategy)

        # A module that received some calibration rows but fewer than the
        # configured ``Fallback.threshold`` has a near rank-deficient Hessian
        # (a rarely routed MoE expert is the typical case). GPTQ applies the
        # documented low-sample policy before it uses the Hessian; QQQ used to
        # consult the threshold only after a failed inverse, so these modules
        # silently ran the full error-propagation loop. Apply the same check
        # here, before the block pass, so the configured fallback is honoured
        # for every module the policy selects.
        if should_use_fallback(self.fallback, float(self.nsamples), self.expected_nsamples):
            threshold_raw, is_percent = resolve_threshold(self.fallback, self.expected_nsamples)
            threshold_text = str(getattr(self.fallback, "threshold", None))
            extra = f", threshold_raw={threshold_raw}" if threshold_raw is not None and is_percent else ""
            log.warn(
                "Quantization: Module `%s` -> Using `%s` fallback quantization "
                "(observed %s samples, threshold=%s%s, max_total=%s).",
                self.name,
                resolved_strategy.value,
                self.nsamples,
                threshold_text,
                extra,
                self.expected_nsamples,
            )
            # The fallback grid is data-independent, but SmoothMSE still
            # weights its search by the captured input-channel energy. Keep
            # only the diagonal (``columns`` floats) before releasing the
            # near-singular matrix (#5300).
            hessian_diag = (
                self.H.diagonal().detach().clone()
                if isinstance(getattr(self.fallback, "smooth", None), SmoothMSE)
                else None
            )
            self.H = None
            self._materialized_samples = 0
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()
            return self._fallback_quantize(
                resolved_strategy, hessian_diag=hessian_diag
            )

        percdamp = self.qcfg.damp_percent
        groupsize = self.qcfg.group_size
        actorder = self.qcfg.desc_act
        static_groups = self.qcfg.static_groups

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d)):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if self._tp_pad_cols:
            pad = torch.zeros(
                (W.shape[0], self._tp_pad_cols),
                dtype=W.dtype,
                device=W.device,
            )
            W = torch.cat((W, pad), dim=1)

        tick = time.time()

        H = self.H
        del self.H
        # The quantizer consumes and mutates this Hessian; it is no longer the
        # materialized calibration state, so later batches start a fresh merge.
        self._materialized_samples = 0
        _neutralize_dead_columns(H, W)

        # Fit the whole-row grid after dead-column masking: a channel absent
        # from every calibration sample carries no signal, so its weight must
        # not set the per-channel scale that every live column is quantized
        # with (#5611).
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        g_idx = []
        scale = []
        zero = []
        now_idx = 1
        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i: (i + groupsize)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        threshold_raw, is_percent = resolve_threshold(self.fallback, self.expected_nsamples)
        fallback_configured = threshold_raw is not None

        # Damping recovery is shared with GPTQ (#5858): probe the configured
        # `damp_percent` first, then the additive `damp_auto_increment` grid,
        # then a diagonal floor. `damp_auto_increment=0` stays the explicit
        # no-recovery control, so a rank-deficient Hessian still fails there.
        # QQQ keeps its own factor convention: a lower Cholesky factor, then
        # the inverse's upper factor.
        orig_diag = H.diagonal().clone()
        npu_block_size = (
            resolve_kernel_config(self.qcfg).npu_cholesky_block_size
            if H.device.type == "npu" else None
        )

        def _recover_factor(hessian: torch.Tensor):
            """Return ``(inverse factor, applied damp)`` for ``hessian``."""

            return recover_damped_hessian_factor(
                hessian,
                damp_percent=float(percdamp),
                damp_auto_increment=float(self.qcfg.damp_auto_increment),
                name=self.name,
                npu_block_size=npu_block_size,
            )

        try:
            Hinv, used_damp = _recover_factor(H)
        except Exception as exc:
            # Reuse GPTQ's allocation classifier and CPU retry: cuSOLVER /
            # hipSOLVER report a failed workspace allocation as
            # ``*SOLVER_STATUS_ALLOC_FAILED``, which the old bare ``raise``
            # turned into an aborted run. A non-positive-definite
            # factorization is a numerical failure instead, and the shared
            # policy already returned ``None`` for it (issue #4400).
            if H.device.type != "cuda" or not _is_memory_allocation_failure(exc):
                raise
            log.warn(
                "Quantization: Module `%s` -> CUDA OOM during Hessian inverse on %s; "
                "falling back to CPU. Due to this fallback, the calculation may take "
                "much longer than normal.",
                self.name,
                H.device,
            )
            cpu_device = torch.device("cpu")
            # The failed attempt may have left the last diagonal floor applied.
            H.diagonal().copy_(orig_diag)
            H = H.to(device=cpu_device)
            W = W.to(device=cpu_device)
            if actorder:
                # ``perm``/``invperm`` were built from the CUDA Hessian.
                # The block pass now runs on the host, so the epilogue's
                # ``Q[:, invperm]`` must not index a CPU tensor with a
                # CUDA index tensor (issue #4538).
                perm = perm.to(device=cpu_device)
                invperm = invperm.to(device=cpu_device)
            self.quantizer.find_params(W, weight=True)
            if static_groups:
                scale, zero = _move_static_groups(groups, cpu_device)
            Hinv, used_damp = _recover_factor(H)

        del H
        if Hinv is None:
            fallback_requested = should_use_fallback(
                self.fallback,
                float(self.nsamples),
                self.expected_nsamples,
            )
            if fallback_requested:
                extra = f", threshold_raw={threshold_raw}" if threshold_raw is not None and is_percent else ""
                log.warn(
                    "Quantization: Module `%s` -> Using `%s` fallback quantization (observed %s samples, threshold=%s%s, max_total=%s).",
                    self.name,
                    resolved_strategy.value,
                    self.nsamples,
                    self.fallback,
                    extra,
                    self.expected_nsamples,
                )
                # The selected fallback is data-independent: it must never
                # continue into the block loop with ``Hinv=None``. RTN used to
                # fall through here and fail with
                # ``TypeError: 'NoneType' object is not subscriptable``.
                return self._fallback_quantize(resolved_strategy)
            raise torch._C._LinAlgError(
                f"QQQ: Hessian remained non positive-definite for `{self.name}` "
                f"with `damp_percent={float(percdamp):.5f}` and "
                f"`damp_auto_increment={float(self.qcfg.damp_auto_increment):.5f}`"
            )

        # Report the additive damping that was actually applied.
        damp = used_damp * torch.mean(orig_diag)

        use_mlx = _should_use_mlx_qqq_quantization(
            W,
            Hinv,
            self.quantizer,
            group_size=groupsize,
            static_groups=static_groups,
            blocksize=blocksize,
        )
        scale_extra = None
        if use_mlx:
            Q, mlx_scale, mlx_zero, _, mlx_loss = (
                _qqq_quantize_weight_mlx_to_torch(
                    W,
                    Hinv,
                    group_size=groupsize,
                )
            )
            scale.append(mlx_scale)
            zero.append(mlx_zero)
            loss_sum = float(mlx_loss.item())
        else:
            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            # The eager path issues one tiny launch per column. On CUDA the
            # shared GPTQ block-update kernel folds the whole 128-column block
            # into a single launch; it implements the asymmetric `zero` form,
            # so it is only used for grouped weights.
            fused_block_update = None
            if (
                Hinv is not None
                and W.device.type == "cuda"
                and blocksize == 128
                and groupsize != -1
                and type(self.quantizer) is Quantizer
                and not self.qcfg.mock_quantization
                and resolve_kernel_config(self.qcfg).gptq_fused_block_update
            ):
                from .gptq_cuda import block_update_available, gptq_block_update

                if block_update_available():
                    fused_block_update = gptq_block_update

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                # ``Q`` is written through the block view and ``Losses`` through its
                # block slice; both are fully overwritten per block, so no staging
                # copy or zero-fill is required.
                Q1 = Q[:, i1:i2]
                Err1 = torch.empty_like(W1)
                Losses1 = Losses[:, i1:i2]
                Hinv1 = Hinv[i1:i2, i1:i2]

                if fused_block_update is not None and count == 128:
                    # Group parameters depend on W, which the eager column loop
                    # leaves untouched until the block finishes, so every group
                    # boundary inside the block can be resolved before launch.
                    uniform_group = i1 // groupsize == (i2 - 1) // groupsize and (
                        not static_groups or not actorder
                    )
                    if uniform_group:
                        # Every column of this block resolves to the same
                        # quantizer group, so the per-column bookkeeping
                        # collapses to one lookup and a constant group index.
                        if not static_groups:
                            if i1 % groupsize == 0:
                                self.quantizer.find_params(
                                    W[:, i1: i1 + groupsize], weight=True
                                )

                            if (i1 // groupsize) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            self.quantizer = groups[i1 // groupsize]

                        block_scale = self.quantizer.scale.flatten().unsqueeze(0)
                        block_zero = self.quantizer.zero.flatten().unsqueeze(0)
                        column_group = _uniform_column_group(count, W.device)
                    else:
                        block_scales = []
                        block_zeros = []
                        column_groups = []
                        parameter_indices = {}
                        for i in range(count):
                            column = i1 + i
                            if not static_groups:
                                if column % groupsize == 0:
                                    self.quantizer.find_params(
                                        W[:, column: column + groupsize], weight=True
                                    )

                                if (column // groupsize) - now_idx == -1:
                                    scale.append(self.quantizer.scale)
                                    zero.append(self.quantizer.zero)
                                    now_idx += 1
                            else:
                                idx = column
                                if actorder:
                                    idx = perm[idx]
                                self.quantizer = groups[idx // groupsize]

                            key = id(self.quantizer.scale)
                            group_index = parameter_indices.get(key)
                            if group_index is None:
                                group_index = len(block_scales)
                                parameter_indices[key] = group_index
                                block_scales.append(self.quantizer.scale.flatten())
                                block_zeros.append(self.quantizer.zero.flatten())
                            column_groups.append(group_index)

                        block_scale = torch.stack(block_scales)
                        block_zero = torch.stack(block_zeros)
                        column_group = torch.tensor(
                            column_groups, device=W.device, dtype=torch.int32
                        )
                    # QQQ's own quantizer has no code-bound state; keep the
                    # natural ``[0, maxq]`` grid bitwise unchanged.
                    fused_block_update(
                        W1,
                        Hinv1,
                        block_scale,
                        block_zero,
                        column_group,
                        None,
                        None,
                        Q1,
                        Err1,
                        Losses1,
                        self.quantizer._maxq_value,
                    )
                else:
                    for i in range(count):
                        w = W1[:, i]
                        d = Hinv1[i, i]

                        if groupsize != -1:
                            if not static_groups:
                                if (i1 + i) % groupsize == 0:
                                    self.quantizer.find_params(
                                        W[:, (i1 + i): (i1 + i + groupsize)], weight=True
                                    )

                                if ((i1 + i) // groupsize) - now_idx == -1:
                                    scale.append(self.quantizer.scale)
                                    zero.append(self.quantizer.zero)
                                    now_idx += 1
                            else:
                                idx = i1 + i
                                if actorder:
                                    idx = perm[idx]
                                self.quantizer = groups[idx // groupsize]

                        q = quantize(
                            w.unsqueeze(1),
                            self.quantizer.scale,
                            self.quantizer.zero,
                            self.quantizer._maxq_value,
                            self.quantizer.sym,
                            self.quantizer.groupsize,
                        ).flatten()
                        Q1[:, i] = q
                        Losses1[:, i] = (w - q) ** 2 / d ** 2

                        err1 = (w - q) / d
                        W1[:, i:] -= err1.unsqueeze(1).matmul(
                            Hinv1[i, i:].unsqueeze(0)
                        )
                        Err1[:, i] = err1

                Losses1.div_(2)

                _propagate_block_error(W[:, i2:], Err1, Hinv[i1:i2, i2:])

                if DEBUG:
                    self.layer.weight.data[:, :i2] = Q[:, :i2]
                    self.layer.weight.data[:, i2:] = W[:, i2:]
                    print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    print(torch.sum(Losses))
            loss_sum = torch.sum(Losses).item()
            del Losses

        print("time %.2f" % (time.time() - tick))
        print("error", loss_sum)

        if Hinv is not None:
            del Hinv
            if self.nsamples != 0:
                avg_loss = loss_sum / self.nsamples

                if math.isnan(avg_loss):
                    print("Losses sum item:", loss_sum)
                    if fallback_configured:
                        log.info(f"Quantization: Failed due to `NaN` loss for `{self.name}`, use mock quantization retry for `{self.name}`")
                        self.qcfg.mock_quantization = True
                        return self.quantize(blocksize=blocksize)
                    else:
                        raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`, please try increasing calibration data samples or enable fallback=True")
            else:
                if fallback_configured:
                    log.warn(f"Quantization: Module `{self.name}` -> using fail safe mode. Please check if calibration data is sufficient.")
                else:
                    log.warn(f"Quantization: `{self.name}` is not activated due to model inference logic (MoE)")
                avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999
        else:
            avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999

        groupsize = groupsize if groupsize != -1 else self.columns
        if actorder:
            # A CPU retry can leave the permutation on the Hessian's device
            # while ``Q`` is on the host; index on ``Q``'s device so the
            # epilogue is device-agnostic (issue #4538).
            perm = perm.to(device=Q.device)
            invperm = invperm.to(device=Q.device)
        if static_groups and actorder:
            g_idx = (perm // groupsize).to(dtype=torch.int32, device=Q.device)
        else:
            g_idx = torch.arange(self.columns, dtype=torch.int32, device=Q.device) // groupsize
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx = g_idx[:valid_cols]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        if Q.shape != self.layer.weight.shape:
            Q = Q.reshape(self.layer.weight.shape).to(self.layer.weight.dtype)
        else:
            Q = Q.to(self.layer.weight.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            scale = self._truncate_last_dim(scale, valid_cols)
            zero = self._truncate_last_dim(zero, valid_cols)

        Q = safe_move_to(Q, self.layer.weight.data.device)

        # post int8 quant
        if scale_extra is None and groupsize != self.columns:
            quantizer_extra = Quantizer(qcfg=self.qcfg)
            quantizer_extra.configure(
                bits=8,
                perchannel=True,
                groupsize=-1,
                sym=True,
                mse=False,
            )
            # Derive the INT8 channel scale from the quantized weight `Q` that
            # is about to become `layer.weight` (as upstream QQQ does), not
            # from the pre-quantization `W`. Error propagation can raise
            # max|Q| above max|W|; a W-fitted scale then clamps packed codes
            # and the export no longer equals the in-memory `wq`.
            # ``Q`` is back in the source ``Conv1D`` [in, out] orientation;
            # the per-channel scale belongs to the output axis.
            extra_weight = Q.t() if isinstance(self.layer, transformers.Conv1D) else Q
            quantizer_extra.find_params(extra_weight.clone(), weight=True)
            scale_extra = quantizer_extra.scale

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, scale_extra, self.nsamples

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self._materialized_samples = 0
        self._device_hessian_partials.clear()
        self._device_sample_counts.clear()
        self.Losses = None
        self.Trace = None
        if hasattr(self, "quantizer"):
            del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy
        if self._named_module is not None:
            self._named_module.state.pop("tp_pad_info", None)
        torch.cuda.empty_cache()
