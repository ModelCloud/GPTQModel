# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Based on original gptq algorithm and code from https://github.com/IST-DASLab/gptq

import contextlib
import math
import os
import sys
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..quantization.config import FailSafeStrategy, SmoothMSE
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.torch import torch_sync
from .failsafe_smooth import mse_optimal_quant, smooth_block
from .gar import compose_final_perm, compute_global_perm, compute_local_perms, invert_perm
from .quantizer import HF_OPTIMUM, Quantizer


log = setup_logger()

lock = threading.Lock()

# Shared workspaces are cached globally per device so that concurrent GPTQ
# instances reuse temporary buffers instead of repeatedly allocating large
# tensors during Hessian accumulation. Each device retains at most a single
# workspace; when size or dtype requirements change, the prior buffer is
# discarded to avoid unbounded cache growth.
_WORKSPACE_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_WORKSPACE_LOCKS: Dict[Tuple[str, Optional[int]], threading.Lock] = {}
_BF16_SUPPORT_CACHE: Dict[Tuple[str, Optional[int]], bool] = {}


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    dev = torch.device(device)
    return dev.type, dev.index


def _workspace_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    return _device_cache_key(device)


def _needs_workspace_resize(
    workspace: Optional[torch.Tensor],
    dtype: torch.dtype,
    required_rows: int,
    cols: int,
) -> bool:
    if workspace is None:
        return True
    if workspace.ndim != 2:
        return True
    if workspace.dtype != dtype:
        return True
    if workspace.shape[1] != cols:
        return True
    if workspace.shape[0] < required_rows:
        return True
    return False


@contextlib.contextmanager
def _lease_workspace(
    device: torch.device,
    dtype: torch.dtype,
    cols: int,
    required_rows: int,
) -> Tuple[torch.Tensor, bool]:
    key = _workspace_cache_key(device)
    lock = _WORKSPACE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        workspace = _WORKSPACE_CACHE.pop(key, None)
        reused = workspace is not None and not _needs_workspace_resize(
            workspace,
            dtype,
            required_rows,
            cols,
        )
        if not reused:
            rows = max(required_rows, 1)
            workspace = torch.empty((rows, cols), dtype=dtype, device=device)
    try:
        yield workspace, reused
    finally:
        with lock:
            _WORKSPACE_CACHE[key] = workspace


def _device_supports_bfloat16(device: torch.device) -> bool:
    cache_key = _device_cache_key(device)
    cached = _BF16_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dev = torch.device(device)
    if dev.type == "meta":
        _BF16_SUPPORT_CACHE[cache_key] = False
        return False

    try:
        a = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        b = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        _ = torch.matmul(a, b)
        support = True
    except Exception:
        support = False

    _BF16_SUPPORT_CACHE[cache_key] = support
    return support


def get_number_of_rows_and_cols(layer: nn.Module):
    # return layer.weight.shape[0], np.prod(layer.weight.shape[1:])
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        # transformers.Conv1D: weight shape is (n_in, n_out)
        return layer.weight.shape[1], layer.weight.shape[0]
    else:
        # weight shape is (n_out, n_in)
        return layer.weight.shape[0], np.prod(layer.weight.shape[1:])


class GPTQ:
    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None):
        self.lock = threading.Lock()

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
            self.module = module.module
            self.name = module.name
            self._named_module = module
        else:
            self.name = HF_OPTIMUM
            self.module = module
            self._named_module = None

        self._original_rows = self.rows
        self._original_columns = self.columns
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.module, "_tp_pad_info", None)
        if isinstance(pad_info, dict):
            pad_cols = int(pad_info.get("pad_cols", 0) or 0)
            pad_cols = max(pad_cols, 0)
        else:
            pad_info = None
            pad_cols = 0

        self._tp_pad_info = pad_info
        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols

        module_device = get_device(self.module)
        setattr(self.module, "target_device", module_device)

        if module_device.type == "meta":
            self._final_hessian_device_hint = torch.device("cpu")
        else:
            self._final_hessian_device_hint = torch.device(module_device)

        self.validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg

        self.module_copy = None

        self.H = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd counter
        self.fwd_counter = 0

        self.failsafe = self.qcfg.failsafe
        self.expected_nsamples: Optional[float] = None

        self.H: Optional[torch.Tensor] = None

        # Store per-device Hessian contributions so multi-GPU calibration can
        # keep local accumulators and merge only once when quantization begins.
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self._hessian_dirty: bool = False

        self._borrow_workspace_stats = {
            "requests": 0,
            "staging_requests": 0,
            "staging_hits": 0,
            "staging_misses": 0,
            "materialized_requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
        }
        self._borrow_workspace_totals = {
            "requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
            "staging_hits": 0,
            "staging_misses": 0,
        }
        self._borrow_workspace_last_summary: Optional[Dict[str, object]] = None
        self._borrow_workspace_stage_dtype: Optional[torch.dtype] = None
        self._borrow_workspace_last_chunk_rows: Optional[int] = None

    @staticmethod
    def validate_module(module):
        assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
                                   transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"

    # def has_hessian_issues(self) -> bool:
    #     return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def mock_hessian_inverse(self, H: torch.Tensor):
        """Mock hessian inverse for fast testing"""
        damp = self.qcfg.damp_percent
        # Return identity matrix instead of complex inversion
        identity = torch.eye(H.shape[0], dtype=torch.float32, device=H.device)
        return identity, damp

    def clone_module(self, copy=True, device: torch.device = None):
        if not device:
            device = self.module.weight.data.device

        clone = self.module.weight.data.to(copy=copy, device=device)

        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        if self._tp_pad_cols:
            pad = torch.zeros(
                (clone.shape[0], self._tp_pad_cols),
                dtype=clone.dtype,
                device=clone.device,
            )
            clone = torch.cat((clone, pad), dim=1)

        return clone.float()

    @staticmethod
    def truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor

        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor

        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        batch_token_size, xtx, device = self.process_batch(inp)
        if batch_token_size == 0 or xtx is None:
            return

        dev = torch.device(device)

        with self.lock:
            self.fwd_counter += 1

            existing = self._device_hessian_partials.get(dev)
            if existing is None:
                self._device_hessian_partials[dev] = xtx
            else:
                existing.add_(xtx)
                del xtx

            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_token_size
            self.nsamples += batch_token_size
            self._hessian_dirty = True

    def preferred_staging_dtype(self, input_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        device = torch.device(device)

        staging_dtype = self.qcfg.hessian.staging_dtype
        if staging_dtype == torch.float32:
            return torch.float32

        if input_dtype not in (torch.float16, torch.bfloat16):
            return torch.float32

        if staging_dtype == torch.bfloat16:
            if not _device_supports_bfloat16(device):
                return torch.float32
            return torch.bfloat16

        if staging_dtype == torch.float16:
            return torch.float16

        return torch.float32

    def resolve_hessian_chunk_size(self, rows: int, stage_dtype: torch.dtype) -> Optional[int]:
        if rows == 0:
            return None

        cfg_chunk = self.qcfg.hessian.chunk_size
        if cfg_chunk is not None:
            return max(1, min(cfg_chunk, rows))

        bytes_budget = self.qcfg.hessian.chunk_bytes
        if bytes_budget is not None:
            bytes_per_row = self.columns * torch.tensor([], dtype=stage_dtype).element_size()
            if bytes_per_row > 0:
                chunk_rows = bytes_budget // bytes_per_row
                if chunk_rows > 0:
                    return max(1, min(int(chunk_rows), rows))
            return 1

        return None

    @contextlib.contextmanager
    def borrow_materialized_chunk_fp32(
        self,
        chunk: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        if rows == 0:
            yield chunk.new_zeros((0, self.columns), dtype=torch.float32)
            return

        device = chunk.device
        stage_dtype = self.preferred_staging_dtype(chunk.dtype, device)

        stats = self._borrow_workspace_stats
        stats["requests"] += 1

        with _lease_workspace(device, stage_dtype, self.columns, rows) as (
            staging_workspace,
            staging_reused,
        ):
            stats["staging_requests"] += 1
            if staging_reused:
                stats["staging_hits"] += 1
            else:
                stats["staging_misses"] += 1

            staging_view = staging_workspace[:rows, :]
            staging_view.copy_(chunk.to(dtype=stage_dtype))

            if stage_dtype == torch.float32:
                stats["materialized_requests"] += 1
                if staging_reused:
                    stats["materialized_hits"] += 1
                else:
                    stats["materialized_misses"] += 1

                try:
                    yield staging_view
                finally:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
            else:
                with _lease_workspace(
                    device,
                    torch.float32,
                    self.columns,
                    rows,
                ) as (
                    fp32_workspace,
                    fp32_reused,
                ):
                    stats["materialized_requests"] += 1
                    if fp32_reused:
                        stats["materialized_hits"] += 1
                    else:
                        stats["materialized_misses"] += 1

                    try:
                        fp32_view = fp32_workspace[:rows, :]
                        fp32_view.copy_(staging_view.to(torch.float32))
                        yield fp32_view
                    finally:
                        if device.type == "cuda":
                            torch.cuda.current_stream(device).synchronize()

    def compute_hessian_xtx(self, matrix: torch.Tensor) -> torch.Tensor:
        rows = matrix.shape[0]
        if rows == 0:
            return torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        stage_dtype = self.preferred_staging_dtype(matrix.dtype, matrix.device)
        chunk_size = self.resolve_hessian_chunk_size(rows, stage_dtype)
        self._borrow_workspace_stage_dtype = stage_dtype
        self._borrow_workspace_last_chunk_rows = chunk_size if chunk_size is not None else rows

        if chunk_size is None:
            mat32 = matrix.to(dtype=torch.float32)
            xtx = torch.matmul(mat32.T, mat32)
            del mat32
            torch_sync(device=xtx.device)
            return xtx

        xtx_accum = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        for start in range(0, rows, chunk_size):
            rows_this = min(chunk_size, rows - start)
            source = matrix[start:start + rows_this]
            with self.borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                materialized32 = materialized
                xtx_accum.add_(torch.matmul(materialized32.T, materialized32))

        torch_sync(device=xtx_accum.device)
        return xtx_accum

    def process_batch(self, inp: torch.Tensor) -> Tuple[int, Optional[torch.Tensor], torch.device]:
        # print(f"inp = {inp}")
        # print(f"self.module = {self.module} device = {self.module.target_device}")
        inp_device = get_device(inp)

        #inp = inp.to(device=self.module.target_device, dtype=torch.float32)

        # input reshaping
        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            reshaped_inp = inp.reshape(-1, inp.shape[-1])
        else:
            if isinstance(self.module, nn.Conv1d):
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    1,
                )
                unfold = nn.Unfold(
                    self.module.kernel_size + (1,),
                    dilation=self.module.dilation + (1,),
                    padding=self.module.padding + (0,),
                    stride=self.module.stride + (1,),
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            else:
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                unfold = nn.Unfold(
                    self.module.kernel_size,
                    dilation=self.module.dilation,
                    padding=self.module.padding,
                    stride=self.module.stride,
                )
                # output size (batch_size, channels * \prod kernel_size, num_patches)
                reshaped_inp = unfold(reshaped_inp)
            reshaped_inp = reshaped_inp.transpose(1, 2).flatten(0, 1)

        # Delay dtype conversion until we materialize Hessian chunks to avoid unnecessary temporaries
        reshaped_inp = reshaped_inp.contiguous()
        if self._tp_pad_cols:
            pad = reshaped_inp.new_zeros((reshaped_inp.shape[0], self._tp_pad_cols))
            reshaped_inp = torch.cat((reshaped_inp, pad), dim=1)
            del pad
        canonical_device = torch.device(inp_device)

        batch_token_size = reshaped_inp.shape[0]

        if batch_token_size == 0:
            del reshaped_inp
            return 0, None, canonical_device

        try:
            xtx = self.compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
        except RuntimeError as exc:
            if (
                torch.device(inp_device).type == "cuda"
                and "out of memory" in str(exc).lower()
            ):
                log.warn(
                    "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                    getattr(self, "name", "<unknown>"),
                )
                reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                del reshaped_inp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                canonical_device = torch.device("cpu")
                xtx = self.compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
                xtx = xtx.detach()
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            xtx = xtx.detach()
            del reshaped_inp

        self._snapshot_borrow_workspace_stats(context="process_batch")
        return batch_token_size, xtx, canonical_device

    def _select_hessian_target_device(self, requested: Optional[torch.device]) -> torch.device:
        if requested is not None:
            return torch.device(requested)

        hint = getattr(self, "_final_hessian_device_hint", None)
        if hint is not None:
            return torch.device(hint)

        if self._device_hessian_partials:
            partial_device = next(iter(self._device_hessian_partials.keys()))
            return torch.device(partial_device)

        return torch.device("cpu")

    def materialize_global_hessian(self, target_device: Optional[torch.device] = None) -> None:
        device = self._select_hessian_target_device(target_device)

        with self.lock:
            if not self._hessian_dirty and self.H is not None:
                if self.H.device != device:
                    self.H = self.H.to(device=device)
                return

            total_samples = sum(self._device_sample_counts.values())

            # Reuse the existing tensor when possible to avoid an extra allocation.
            reuse_buffer = (
                self.H is not None
                and self.H.shape == (self.columns, self.columns)
                and self.H.device == device
            )

            result_accum: torch.Tensor
            if reuse_buffer and self.H.dtype == torch.float32:
                result_accum = self.H
                result_accum.zero_()
            else:
                torch_sync(device) # try to avoid torch.AcceleratorError: CUDA error: unspecified launch failure
                result_accum = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=device,
                )

            if total_samples == 0:
                self.H = result_accum
                self.nsamples = 0
                self._hessian_dirty = False
                self._final_hessian_device_hint = device
                self._device_hessian_partials.clear()
                self._device_sample_counts.clear()
                return

            for partial_device, partial in self._device_hessian_partials.items():
                if partial.device != result_accum.device or partial.dtype != torch.float32:
                    # TODO FIXME multi-3090 using P2P is revaling an issue where result_accum and/or partial is not ready for consolidation on the main thread
                    # when parials are calculated on the individual
                    try:
                        result_accum.add_(partial.to(device=result_accum.device, dtype=torch.float32))
                    except:
                        log.warn(f"Quantization: Module `{self.name}` -> Retry partial.to 1/2 in 0.25s")
                        time.sleep(0.25)
                        try:
                            result_accum.add_(partial.to(device=result_accum.device, dtype=torch.float32))
                        except:
                            log.warn(f"Quantization: Module `{self.name}` -> Retry partial.to 2/2 in 0.75s")
                            time.sleep(0.75)
                            result_accum.add_(partial.to(device=result_accum.device, dtype=torch.float32))
                else:
                    result_accum.add_(partial)

            result_accum.mul_(2.0 / float(total_samples))

            self.H = result_accum
            self.nsamples = total_samples
            self._hessian_dirty = False
            self._final_hessian_device_hint = result_accum.device
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()
            del result_accum

    def finalize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        self.materialize_global_hessian(target_device=target_device)
        if self.H is None:
            self.H = self.create_H(target_device)
        return self.H

    def create_H(self, target_device):
        return torch.zeros((self.columns, self.columns), dtype=torch.float32,
                           device=self._select_hessian_target_device(target_device))

    def _failsafe_quantize(self, strategy: FailSafeStrategy, blocksize: int):
        """Apply a lightweight quantization fallback using the requested strategy."""
        maxq = 2 ** self.qcfg.bits - 1
        sigma = 3.0
        effective_group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        start_time = time.time()
        smooth_method = getattr(self.failsafe, "smooth", None)
        mse_steps = 32
        mse_maxshrink = 0.8
        if isinstance(smooth_method, SmoothMSE):
            mse_steps = smooth_method.steps
            mse_maxshrink = smooth_method.maxshrink

        target_device = self.H.device if self.H is not None else self.module.weight.device
        W = self.clone_module(device=target_device)
        Q = torch.empty_like(W)
        scale_chunks = []
        zero_chunks = []

        for start in range(0, self.columns, effective_group_size):
            end = min(start + effective_group_size, self.columns)
            block = W[:, start:end]

            if isinstance(smooth_method, SmoothMSE):
                dequant, scale, zero = mse_optimal_quant(
                    block,
                    self.qcfg,
                    maxq,
                    steps=mse_steps,
                    maxshrink=mse_maxshrink,
                )
            else:
                block_mod, scale_factor = smooth_block(
                    block,
                    self.failsafe,
                    group_size=effective_group_size,
                )
                if strategy == FailSafeStrategy.MIDPOINT:
                    w_min = block_mod.min(dim=1, keepdim=True).values
                    w_max = block_mod.max(dim=1, keepdim=True).values
                    mid = (w_max + w_min) / 2.0
                    scale = torch.clamp((w_max - w_min) / maxq, min=1e-8)
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - mid) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (mid / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FailSafeStrategy.MEAN:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    max_dev = torch.max((block_mod - mean).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    scale = (2 * max_dev) / maxq
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - mean) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (mean / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FailSafeStrategy.MEDIAN:
                    median = block_mod.median(dim=1, keepdim=True).values
                    max_dev = torch.max((block_mod - median).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    scale = (2 * max_dev) / maxq
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - median) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (median / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FailSafeStrategy.STDCLIP:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    std = block_mod.std(dim=1, keepdim=True, unbiased=False)
                    std = torch.clamp(std, min=1e-8)
                    lo = mean - sigma * std
                    hi = mean + sigma * std
                    scale = torch.clamp((hi - lo) / maxq, min=1e-8)
                    zero = torch.round(-lo / scale)
                    zero = torch.clamp(zero, 0, maxq)
                    q = torch.round(block_mod / scale + zero)
                    q = torch.clamp(q, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FailSafeStrategy.RTN:
                    self.quantizer.find_params(block_mod, weight=True)
                    dequant = self.quantizer.quantize(block_mod)
                    scale = self.quantizer.scale
                    zero = self.quantizer.zero
                else:
                    raise ValueError(f"Unsupported failsafe strategy: {strategy}")

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
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)
        else:
            valid_cols = self.columns

        group_size = effective_group_size if effective_group_size != -1 else self.columns
        g_idx = torch.arange(valid_cols, device=Q.device, dtype=torch.int32) // group_size

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        Q = Q.to(device=self.module.weight.data.device, non_blocking=False)
        mean_abs_err = (Q - self.module.weight.data).abs().mean().item()
        duration = time.time() - start_time
        avg_loss = f"failsafe({strategy.value}): {mean_abs_err:.7f}"
        damp = 0.0

        self.H = None
        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    # FIXME, optimum needs fasterquant, we need to remove it
    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
            act_group_aware: Optional[bool] = None,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        if act_group_aware is not None:
            self.qcfg.act_group_aware = act_group_aware
        self.qcfg._resolve_activation_ordering(actorder, act_group_aware)
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    @torch.inference_mode()
    def hessian_inverse(self, H: torch.Tensor):
        # Capture a writable view of the Hessian diagonal so we can restore it between attempts.
        diag_view = H.diagonal()
        orig_diag = diag_view.clone()

        # When a block is numerically singular, pure damping can stall at 1.0.
        # Prepare a tiny diagonal floor (relative to the largest entry) that we
        # only inject if the normal damping loop fails. Keeping the scale near 1e-6
        # of the dominant entry keeps the bias negligible for healthy layers while
        # still rescuing pathological Hessian blocks.
        base_abs_max = torch.max(orig_diag.abs()).item()
        if not math.isfinite(base_abs_max) or base_abs_max == 0.0:
            base_abs_max = 1.0
        floor_base = base_abs_max * 1e-6
        max_floor_attempts = 6
        used_damp = self.qcfg.damp_percent
        last_error = None

        attempt = 0
        while attempt <= max_floor_attempts:
            if attempt == 0:
                current_diag = orig_diag
            else:
                floor_increment = floor_base * math.pow(10.0, attempt - 1)
                current_diag = torch.clamp(orig_diag + floor_increment, min=floor_increment)
                if attempt == 1:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Applying Hessian diagonal floor (+{floor_increment:.2e}) to recover positive definiteness.")
                else:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Increasing Hessian diagonal floor to +{floor_increment:.2e}.")

            diag_view.copy_(current_diag)
            mean = torch.mean(current_diag)
            damp = self.qcfg.damp_percent

            damp_recovery_started = False
            recovery_initial_damp = None
            recovery_last_damp = None

            while 0 < damp < 1:
                try:
                    diag_view.add_(damp * mean)
                    H2 = torch.linalg.cholesky(H)
                    Hinv_result = torch.linalg.cholesky(torch.cholesky_inverse(H2), upper=True)
                    diag_view.copy_(current_diag)
                    del H2
                    used_damp = damp
                    if damp_recovery_started:
                        log.warn(
                            f"Quantization: Module `{self.name}` -> Damp recovery succeeded at `damp_percent={damp:.5f}` "
                            f"(started at {recovery_initial_damp:.5f})."
                        )
                    return Hinv_result, used_damp
                except torch._C._LinAlgError as e:
                    last_error = e
                    diag_view.copy_(current_diag)
                    if self.qcfg.damp_auto_increment != 0:
                        if not damp_recovery_started:
                            damp_recovery_started = True
                            recovery_initial_damp = damp
                            log.warn(
                                f"Quantization: Module `{self.name}` -> Starting damp recovery at "
                                f"`damp_percent={damp:.5f}`, increment step `{self.qcfg.damp_auto_increment:.5f}`."
                            )
                        damp += self.qcfg.damp_auto_increment
                        recovery_last_damp = damp
                    else:
                        log.warn(
                            f"Quantization: Module `{self.name}` -> Hessian Cholesky failed with `damp_percent={damp:.5f}` and no auto increment configured.")
                        break

            if damp_recovery_started:
                final_damp = recovery_last_damp if recovery_last_damp is not None else damp
                log.warn(
                    f"Quantization: Module `{self.name}` -> Damp recovery failed after reaching `damp_percent={final_damp:.5f}`."
                )

            attempt += 1

        log.error(
            f"Quantization: Module `{self.name}` -> Hessian remained non positive-definite after diagonal floor attempts. Last `damp_percent` tried = {damp:.5f}.")
        if last_error is not None:
            log.debug(f"Hessian failure detail: {last_error}")
        return None, 1.0

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        target_device = getattr(self.module, "target_device", None)
        from ..utils.failsafe import resolve_failsafe_strategy, resolve_threshold, should_use_failsafe

        resolved_strategy = resolve_failsafe_strategy(self.failsafe)
        fallback_requested = should_use_failsafe(
            self.failsafe,
            float(self.nsamples),
            self.expected_nsamples,
        )
        threshold_raw, is_percent = resolve_threshold(self.failsafe, self.expected_nsamples)
        failsafe_configured = threshold_raw is not None

        if fallback_requested:
            use_hessian = False
            threshold_text = str(getattr(self.failsafe, "threshold", None))
            threshold_info = f", threshold_raw={threshold_raw}" if threshold_raw is not None and is_percent else ""
            log.warn(
                f"Quantization: Module `{self.name}` -> "
                f"Using `{resolved_strategy.value}` failsafe quantization (observed {self.nsamples} samples, threshold={threshold_text}{threshold_info}, max_total={self.expected_nsamples})."
            )
            self.H = self.create_H(target_device=target_device)

            return self._failsafe_quantize(resolved_strategy, blocksize)
        else:
            use_hessian = True
            self.finalize_hessian(target_device=target_device)

        # Temporarily disable torch.compile due to compatibility issues with torch 2.8
        # Will re-enable once the issue is fixed
        # if not TORCH_GTE_28 and not self.qcfg.mock_quantization:
        #     self.hessian_inverse = torch_compile(self.hessian_inverse)

        if self.qcfg.mock_quantization:
            # Use simplified hessian inverse (identity matrix)
            self.hessian_inverse = self.mock_hessian_inverse

        # if self.device.type not in ["mps", "cpu"]:
        #     self.module.weight.data = self.module.weight.data.cpu()

        # TODO: waiting for pytorch implementation of ops for MPS
        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

        if self.module_copy is None:
            # log.info("copy W to cuda_1")
            W = self.clone_module(device=self.H.device)
        else:
            W = self.module_copy.to(device=self.H.device)
            del self.module_copy

        self.quantizer.find_params(W, weight=True)

        # H = self.H.to(device=self.H.device)

        if use_hessian:
            dead = torch.diag(self.H) == 0
            self.H[dead, dead] = 1
            W[:, dead] = 0

        # g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if self.qcfg.static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, self.qcfg.group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i: (i + self.qcfg.group_size)], weight=True)

                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if self.qcfg.desc_act and use_hessian:
            perm = torch.argsort(torch.diag(self.H), descending=True)
            W = W[:, perm]
            self.H = self.H[perm][:, perm]
            invperm = torch.argsort(perm)

        elif self.qcfg.act_group_aware and use_hessian:
            diag_h = torch.diag(self.H)
            local_perms, local_values = compute_local_perms(
                diag_h, self.qcfg.group_size, return_values=True
            )
            global_perm = compute_global_perm(
                diag_h,
                self.qcfg.group_size,
                precomputed_values=local_values,
            )
            del local_values
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            W = W[:, final_perm]
            self.H = self.H[final_perm][:, final_perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        if use_hessian:
            Hinv, damp = self.hessian_inverse(self.H)
        else:
            Hinv, damp = None, 0.0

        # Use simplified loop when mock_quantization is active
        if self.qcfg.mock_quantization:
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2]
                Q1 = torch.zeros_like(W1)

                # Handle group quantization parameters efficiently (similar to original)
                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        # Find parameters for entire groups at once (optimized)
                        group_start_cols = list(range(i1, i2, self.qcfg.group_size))
                        for group_start in group_start_cols:
                            group_end = min(group_start + self.qcfg.group_size, self.columns)
                            if group_start < group_end:
                                self.quantizer.find_params(W[:, group_start:group_end], weight=True)
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                    else:
                        # Static groups - use pre-computed groups
                        for i in range(count):
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]

                    # Vectorized quantization for the entire block (major optimization)
                    if len(scale) > 0 and len(zero) > 0:
                        # Use latest scale and zero for the entire block
                        latest_scale = scale[-1]
                        latest_zero = zero[-1]

                        # Vectorized quantization using broadcasting
                        # Reshape scales and zeros to match block dimensions
                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)

                        # Apply quantization formula using the cloned weights W1
                        maxq_val = 2 ** self.qcfg.bits - 1
                        if self.qcfg.sym:
                            # Symmetric quantization: Q = scale * clamp(round(x/scale), -maxq/2, maxq/2)
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            # Asymmetric quantization: Q = scale * (clamp(round(x/scale) + zero, 0, maxq) - zero)
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        # Fallback to individual quantization if no scale/zero available
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q
                else:
                    # No grouping - vectorized quantization for entire block
                    maxq_val = 2 ** self.qcfg.bits - 1
                    if hasattr(self.quantizer, 'scale') and hasattr(self.quantizer, 'zero'):
                        latest_scale = self.quantizer.scale
                        latest_zero = self.quantizer.zero

                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)

                        if self.qcfg.sym:
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        # Fallback to individual quantization
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q

                Q[:, i1:i2] = Q1
        else:
            # Original heavy loop for normal quantization
            effective_block = blocksize
            if Hinv is None and self.qcfg.group_size and self.qcfg.group_size > 0:
                # Align RTN fallback work chunks to group boundaries to avoid
                # redundant quantizer reconfiguration across partial groups.
                effective_block = self.qcfg.group_size

            for i1 in range(0, self.columns, effective_block):
                i2 = min(i1 + effective_block, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1) if Hinv is not None else None
                Losses1 = torch.zeros_like(W1) if Hinv is not None else None

                if Hinv is not None:
                    Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    if Hinv is not None:
                        d = Hinv1[i, i]

                    if self.qcfg.group_size != -1:
                        if not self.qcfg.static_groups:
                            if (i1 + i) % self.qcfg.group_size == 0:
                                self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + self.qcfg.group_size)], weight=True)

                            if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]

                            self.quantizer = groups[idx // self.qcfg.group_size]

                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    if Hinv is not None:
                        Losses1[:, i] = (w - q) ** 2 / d**2
                        err1 = (w - q) / d
                        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                        Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                if Hinv is not None:
                    Losses[:, i1:i2] = Losses1 / 2
                    W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                del W1, Q1, Err1, Losses1
                if Hinv is not None:
                    del Hinv1

        # TODO: why is there a torch_sync here? There are no streaming ops here?
        # torch_sync(device=self.module.target_device)

        if Hinv is not None:
            del Hinv
            if self.nsamples != 0:
                avg_loss = torch.sum(Losses).item() / self.nsamples

                if math.isnan(avg_loss):
                    print("Losses sum item:", torch.sum(Losses).item())
                    if failsafe_configured:
                        log.info(f"Quantization: Failed due to `NaN` loss for `{self.name}`, use mock quantization retry for `{self.name}`")
                        self.qcfg.mock_quantization = True
                        return self.quantize(blocksize=blocksize)
                    else:
                        raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`, please try increasing calibration data samples or enable failsafe=True")
            else:
                if failsafe_configured:
                    log.warn(f"Quantization: Module `{self.name}` -> using fail safe mode. Please check if calibration data is sufficient.")
                else:
                    log.warn(f"Quantization: `{self.name}` is not activated due to model inference logic (MoE)")
                avg_loss = f"{resolved_strategy.value} failsafe" if failsafe_configured else 999999999
        else:
            avg_loss = f"{resolved_strategy.value} failsafe" if failsafe_configured else 999999999

        del Losses
        del self.H
        del W

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]

        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if self.qcfg.desc_act and use_hessian:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
            del perm, invperm

        elif self.qcfg.act_group_aware and use_hessian:
            inv_final = invert_perm(final_perm)
            Q = Q[:, inv_final]
            inv_global_perm = invert_perm(global_perm)
            inv_global_perm_list = inv_global_perm.tolist()
            temp_scale = [scale[i] for i in inv_global_perm_list]
            scale = temp_scale
            temp_zero = [zero[i] for i in inv_global_perm_list]
            zero = temp_zero
            del final_perm, inv_final, global_perm, inv_global_perm, inv_global_perm_list, local_perms

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx = g_idx[:valid_cols]

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)

        Q = Q.to(device=self.module.weight.data.device, non_blocking=False)

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def borrow_materialized_chunk_stats(self, reset: bool = False) -> Dict[str, int]:
        stats = dict(self._borrow_workspace_stats)
        if reset:
            for key in self._borrow_workspace_stats:
                self._borrow_workspace_stats[key] = 0
        return stats

    def _snapshot_borrow_workspace_stats(self, *, context: str) -> None:
        stats = self.borrow_materialized_chunk_stats(reset=True)
        total_requests = int(stats.get("requests", 0) or 0)
        if total_requests == 0:
            return

        materialized_hits = int(stats.get("materialized_hits", 0) or 0)
        materialized_misses = int(stats.get("materialized_misses", 0) or 0)
        staging_hits = int(stats.get("staging_hits", 0) or 0)
        staging_misses = int(stats.get("staging_misses", 0) or 0)
        chunk_rows = self._borrow_workspace_last_chunk_rows
        stage_dtype = self._borrow_workspace_stage_dtype
        stage_dtype_str = str(stage_dtype) if stage_dtype is not None else "n/a"
        hit_rate = materialized_hits / total_requests if total_requests else 0.0

        summary = {
            "context": context,
            "requests": total_requests,
            "materialized_hits": materialized_hits,
            "materialized_misses": materialized_misses,
            "staging_hits": staging_hits,
            "staging_misses": staging_misses,
            "chunk_rows": chunk_rows,
            "staging_dtype": stage_dtype_str,
            "hit_rate": hit_rate,
        }
        self._borrow_workspace_last_summary = summary

        totals = self._borrow_workspace_totals
        totals["requests"] += total_requests
        totals["materialized_hits"] += materialized_hits
        totals["materialized_misses"] += materialized_misses
        totals["staging_hits"] += staging_hits
        totals["staging_misses"] += staging_misses

    def log_workspace_stats(self, *, context: str, reset: bool = True) -> None:
        totals = self._borrow_workspace_totals
        total_requests = int(totals.get("requests", 0) or 0)
        if total_requests == 0:
            if reset:
                self.reset_workspace_stats()
            return

        total_hits = int(totals.get("materialized_hits", 0) or 0)
        total_misses = int(totals.get("materialized_misses", 0) or 0)
        total_hit_rate = total_hits / total_requests if total_requests else 0.0

        last = self._borrow_workspace_last_summary or {}
        last_requests = int(last.get("requests", 0) or 0)
        last_hits = int(last.get("materialized_hits", 0) or 0)
        last_misses = int(last.get("materialized_misses", 0) or 0)
        last_hit_rate = float(last.get("hit_rate", 0.0) or 0.0)
        rows_label = last.get("chunk_rows", "n/a")
        stage_dtype = last.get("staging_dtype", "n/a")

        log.info(
            "GPTQ workspace cache [%s]: module=%s rows=%s staging_dtype=%s "
            "requests=%d hits=%d misses=%d hit_rate=%.2f total_requests=%d "
            "total_hits=%d total_misses=%d total_hit_rate=%.2f",
            context,
            getattr(self, "name", "<unknown>"),
            rows_label,
            stage_dtype,
            last_requests,
            last_hits,
            last_misses,
            last_hit_rate,
            total_requests,
            total_hits,
            total_misses,
            total_hit_rate,
        )

        if reset:
            self.reset_workspace_stats()

    def reset_workspace_stats(self) -> None:
        for key in self._borrow_workspace_stats:
            self._borrow_workspace_stats[key] = 0
        for key in self._borrow_workspace_totals:
            self._borrow_workspace_totals[key] = 0
        self._borrow_workspace_last_summary = None
        self._borrow_workspace_stage_dtype = None
        self._borrow_workspace_last_chunk_rows = None

    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy

        if self._named_module is not None:
            self._named_module.state.pop("tp_pad_info", None)

        target = getattr(self, "module", None)
        if target is not None:
            del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
