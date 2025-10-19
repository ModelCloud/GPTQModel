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
from ..utils.device import get_device
from ..utils.logger import setup_logger
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
def _lease_workspace(device: torch.device, dtype: torch.dtype, cols: int, required_rows: int):
    key = _workspace_cache_key(device)
    lock = _WORKSPACE_LOCKS.setdefault(key, threading.Lock())
    with lock:
        workspace = _WORKSPACE_CACHE.pop(key, None)
        if _needs_workspace_resize(workspace, dtype, required_rows, cols):
            rows = max(required_rows, 1)
            workspace = torch.empty((rows, cols), dtype=dtype, device=device)
    try:
        yield workspace
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

        self._validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg

        self.module_copy = None

        self.H = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd counter
        self.fwd_counter = 0

        self.fail_safe = False

        self.H: Optional[torch.Tensor] = None

        # Store per-device Hessian contributions so multi-GPU calibration can
        # keep local accumulators and merge only once when quantization begins.
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self._hessian_dirty: bool = False

    @staticmethod
    def _validate_module(module):
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

    def _mock_hessian_inverse(self, H: torch.Tensor):
        """Mock hessian inverse for fast testing"""
        damp = self.qcfg.damp_percent
        # Return identity matrix instead of complex inversion
        identity = torch.eye(H.shape[0], dtype=torch.float32, device=H.device)
        return identity, damp

    def _clone_module(self, copy=True, device: torch.device = None):
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
    def _truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
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

    def _preferred_staging_dtype(self, input_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        device = torch.device(device)

        if not self.qcfg.hessian_use_bfloat16_staging:
            return torch.float32

        if input_dtype not in (torch.float16, torch.bfloat16):
            return torch.float32

        if not _device_supports_bfloat16(device):
            return torch.float32

        return torch.bfloat16

    def _resolve_hessian_chunk_size(self, rows: int, stage_dtype: torch.dtype) -> Optional[int]:
        if rows == 0:
            return None

        cfg_chunk = self.qcfg.hessian_chunk_size
        if cfg_chunk is not None:
            return max(1, min(cfg_chunk, rows))

        bytes_budget = self.qcfg.hessian_chunk_bytes
        if bytes_budget is not None:
            bytes_per_row = self.columns * torch.tensor([], dtype=stage_dtype).element_size()
            if bytes_per_row > 0:
                chunk_rows = bytes_budget // bytes_per_row
                if chunk_rows > 0:
                    return max(1, min(int(chunk_rows), rows))
            return 1

        return None

    @contextlib.contextmanager
    def _borrow_materialized_chunk_fp32(
        self,
        chunk: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        if rows == 0:
            yield chunk.new_zeros((0, self.columns), dtype=torch.float32)
            return

        device = chunk.device
        stage_dtype = self._preferred_staging_dtype(chunk.dtype, device)

        with _lease_workspace(device, stage_dtype, self.columns, rows) as staging_workspace:
            staging_view = staging_workspace[:rows, :]
            staging_view.copy_(chunk.to(dtype=stage_dtype))

            if stage_dtype == torch.float32:
                try:
                    yield staging_view
                finally:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
            else:
                with _lease_workspace(device, torch.float32, self.columns, rows) as fp32_workspace:
                    try:
                        fp32_view = fp32_workspace[:rows, :]
                        fp32_view.copy_(staging_view.to(torch.float32))
                        yield fp32_view
                    finally:
                        if device.type == "cuda":
                            torch.cuda.current_stream(device).synchronize()

    def _compute_hessian_xtx(self, matrix: torch.Tensor) -> torch.Tensor:
        rows = matrix.shape[0]
        if rows == 0:
            return torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        stage_dtype = self._preferred_staging_dtype(matrix.dtype, matrix.device)
        chunk_size = self._resolve_hessian_chunk_size(rows, stage_dtype)

        if chunk_size is None:
            mat32 = matrix.to(dtype=torch.float32)
            return torch.matmul(mat32.T, mat32)

        xtx_accum = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        for start in range(0, rows, chunk_size):
            rows_this = min(chunk_size, rows - start)
            source = matrix[start:start + rows_this]
            with self._borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                materialized32 = materialized
                xtx_accum.add_(torch.matmul(materialized32.T, materialized32))

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
        canonical_device = torch.device(inp_device)

        batch_token_size = reshaped_inp.shape[0]

        if batch_token_size == 0:
            del reshaped_inp
            return 0, None, canonical_device

        try:
            xtx = self._compute_hessian_xtx(reshaped_inp).to(dtype=torch.float32)
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
                xtx = self._compute_hessian_xtx(reshaped_inp_cpu).to(dtype=torch.float32)
                xtx = xtx.detach()
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            xtx = xtx.detach()
            del reshaped_inp

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

    def _materialize_global_hessian(self, target_device: Optional[torch.device] = None) -> None:
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
                    tmp = partial.to(device=result_accum.device, dtype=torch.float32)
                    result_accum.add_(tmp)
                    del tmp
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
        self._materialize_global_hessian(target_device=target_device)
        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=self._select_hessian_target_device(target_device))
        return self.H

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
        damp = self.qcfg.damp_percent
        mean = torch.mean(torch.diag(H))

        orig_diag = H.diag().clone()
        while 0 < damp < 1:
            try:
                H.diagonal().add_(damp * mean)
                H2 = torch.linalg.cholesky(H)
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(H2), upper=True)
                H.diagonal().copy_(orig_diag)
                del H2
                break
            except torch._C._LinAlgError as e:
                H.diagonal().copy_(orig_diag)
                if self.qcfg.damp_auto_increment != 0:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Current `damp_percent = {damp:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`")
                    damp += self.qcfg.damp_auto_increment
                else:
                    log.warn(
                        "Quantization: Module `{self.name}` -> Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp:.5f}`")
                    raise e

        if not (0 < damp < 1):
            log.error(
                f"Quantization: Module `{self.name}` -> `damp_percent` must between 0 and 1. current is {damp}. Module cannot be correctly processed.")
            return None, 1.0

        return Hinv, damp

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        target_device = getattr(self.module, "target_device", None)
        self.finalize_hessian(target_device=target_device)

        # Temporarily disable torch.compile due to compatibility issues with torch 2.8
        # Will re-enable once the issue is fixed
        # if not TORCH_GTE_28 and not self.qcfg.mock_quantization:
        #     self.hessian_inverse = torch_compile(self.hessian_inverse)

        if self.qcfg.mock_quantization:
            # Use simplified hessian inverse (identity matrix)
            self.hessian_inverse = self._mock_hessian_inverse

        # if self.device.type not in ["mps", "cpu"]:
        #     self.module.weight.data = self.module.weight.data.cpu()

        # TODO: waiting for pytorch implementation of ops for MPS
        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

        if self.module_copy is None:
            # log.info("copy W to cuda_1")
            W = self._clone_module(device=self.H.device)
        else:
            W = self.module_copy.to(device=self.H.device)
            del self.module_copy

        self.quantizer.find_params(W, weight=True)

        # H = self.H.to(device=self.H.device)

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

        if self.qcfg.desc_act:
            perm = torch.argsort(torch.diag(self.H), descending=True)
            W = W[:, perm]
            self.H = self.H[perm][:, perm]
            invperm = torch.argsort(perm)

        elif self.qcfg.act_group_aware:
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

        Hinv, damp = self.hessian_inverse(self.H)

        # Use simplified loop when mock_quantization is active
        if self.qcfg.mock_quantization or (self.fail_safe and self.fwd_counter == 0):
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
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)

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

        # TODO: why is there a torch_sync here? There are no streaming ops here?
        # torch_sync(device=self.module.target_device)

        if Hinv is not None:
            del Hinv
            if self.nsamples != 0:
                avg_loss = torch.sum(Losses).item() / self.nsamples

                if math.isnan(avg_loss):
                    print("Losses sum item:", torch.sum(Losses).item())
                    if self.fail_safe:
                        log.info(f"Quantization: Failed due to `NaN` loss for `{self.name}`, use mock quantization retry for `{self.name}`")
                        self.qcfg.mock_quantization = True
                        return self.quantize(blocksize=blocksize)
                    else:
                        raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`, please try increasing calibration data samples or enable fail_safe=True")
            else:
                if self.fail_safe:
                    log.warn(f"Quantization: Module `{self.name}` -> using fail safe mode. Please check if calibration data is sufficient.")
                else:
                    log.warn(f"Quantization: `{self.name}` is not activated due to model inference logic (MoE)")
                avg_loss = 999999999
        else:
            avg_loss = 999999999

        del Losses
        del self.H

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]

        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if self.qcfg.desc_act:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        elif self.qcfg.act_group_aware:
            inv_final = invert_perm(final_perm)
            Q = Q[:, inv_final]
            inv_global_perm = invert_perm(global_perm)
            inv_global_perm_list = inv_global_perm.tolist()
            temp_scale = [scale[i] for i in inv_global_perm_list]
            scale = temp_scale
            temp_zero = [zero[i] for i in inv_global_perm_list]
            zero = temp_zero

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
            scale = self._truncate_last_dim(scale, valid_cols)
            zero = self._truncate_last_dim(zero, valid_cols)

        Q = Q.to(device=self.module.weight.data.device, non_blocking=False)

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

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
