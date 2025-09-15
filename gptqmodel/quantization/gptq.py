# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import math
import os
import sys
import threading
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.torch import HAS_CUDA, HAS_XPU, TORCH_GTE_28, device_next, torch_compile, torch_sync
from .gar import compose_final_perm, compute_global_perm, compute_local_perms, invert_perm
from .quantizer import HF_OPTIMUM, Quantizer

log = setup_logger()

# TODO: move this to a locking class
# --------------------------------------------------------------------------------------
# Per-device lock registry to guard device-specific critical sections (like tensor moves)
# --------------------------------------------------------------------------------------
_device_locks = {}                 # {(device_type, index): threading.Lock()}
_device_locks_guard = threading.Lock()  # guards the registry itself


def _device_key(dev) -> tuple:
    """
    Normalize a device into a hashable (type, index) key.
    Examples:
      torch.device('cuda', 0) -> ('cuda', 0)
      torch.device('xpu')     -> ('xpu', -1)
      'cuda:1'                -> ('cuda', 1)
      'cpu'                   -> ('cpu', -1)
    """
    if isinstance(dev, torch.device):
        return (dev.type, dev.index if dev.index is not None else -1)
    if isinstance(dev, str):
        try:
            d = torch.device(dev)
            return _device_key(d)
        except Exception:
            return ("str", dev)  # last-resort string key
    # Unknown type — stringify
    return ("unknown", str(dev))


def _get_device_lock(dev) -> threading.Lock:
    key = _device_key(dev)
    with _device_locks_guard:
        lk = _device_locks.get(key)
        if lk is None:
            lk = threading.Lock()
            _device_locks[key] = lk
        return lk
# --------------------------------------------------------------------------------------

lock = threading.Lock()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# TODO: is there a buffer init threading init bug in torch.linalg?
# bypass strange threading bug by warming up torch.linalg.cholesky to setup internal setup calls
if HAS_CUDA or HAS_XPU:
    tmp_eye = torch.eye(64, dtype=torch.float32, device="cuda" if HAS_CUDA else "xpu")
    torch.linalg.cholesky(tmp_eye)
    del tmp_eye


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
            self.module = module.module
            self.name = module.name
        else:
            self.name = HF_OPTIMUM
            self.module = module
            # emulate NamedModule properties
            self.module.target_device, self.module.target_device_stream = device_next()

        self._validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg

        self.module_copy = None

        self.H = None
        self.nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd input buffer
        self.fwd_inputs_buffered = False
        self.fwd_inputs_buffered_data = []

        # fwd counter
        self.fwd_counter = 0

        self.fail_safe = False

        self.H = torch.zeros((self.columns, self.columns),
                                 dtype=torch.float32)

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

        return clone.float()

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        self.fwd_counter += 1

        if self.fwd_inputs_buffered:
            # with torch_streamCtx(self.module.target_device_stream):
            #     self.fwd_inputs_buffered_data.append(inp.to(device=self.module.target_device, non_blocking=True))
            self.fwd_inputs_buffered_data.append(inp.to(device=self.module.target_device, non_blocking=False))
        else:
            self.process_batch(inp)

    def process_batch(self, inp: torch.Tensor):
        inp = inp.to(device=self.module.target_device, dtype=torch.float32)

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

        batch_token_size = reshaped_inp.shape[0]

        if self.H.device != reshaped_inp.device:
            self.H = self.H.to(device=reshaped_inp.device)

        # moe model may receive an empty batch, return early
        if batch_token_size == 0:
            return batch_token_size, reshaped_inp, 0, 0

        beta = self.nsamples / (self.nsamples + batch_token_size)
        alpha = 2.0 / (self.nsamples + batch_token_size)

        self.H.addmm_(reshaped_inp.T, reshaped_inp, beta=beta, alpha=alpha)

        # update number of collected samples
        self.nsamples += batch_token_size

        # inp returned here is flattened/reshaped original inp
        # return batch_token_size, reshaped_inp, alpha, beta
        del batch_token_size, reshaped_inp, alpha, beta

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
            act_group_aware=False,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.desc_act = actorder
        self.qcfg.act_group_aware = act_group_aware
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    @torch.inference_mode()
    def hessian_inverse(self, H: torch.Tensor):

        damp = self.qcfg.damp_percent
        diag = torch.arange(self.columns, device=H.device)
        mean = torch.mean(torch.diag(H))
        while 0 < damp < 1:
            try:
                H2 = H.clone()
                H2[diag, diag] += damp * mean
                # TODO call to torch.linalg is not threadsafe? Porque no? Esta muy mal.
                H2 = torch.linalg.cholesky(H2)
                Hinv = torch.linalg.cholesky(torch.cholesky_inverse(H2), upper=True)
                del H, H2
                break
            except torch._C._LinAlgError as e:
                if self.qcfg.damp_auto_increment != 0:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Current `damp_percent = {damp:.5f}` is too low, auto-incrementing by `{self.qcfg.damp_auto_increment:.5f}`")
                    damp += self.qcfg.damp_auto_increment
                else:
                    log.warn(
                        "Quantization: Module `{self.name}` -> Please increase damp or nsamples for calibration data to avoid the following quant error: current damp_percent=`{damp_percent:.5f}`")
                    raise e

        if not (0 < damp < 1):
            log.error(
                f"Quantization: Module `{self.name}` -> `damp_percent` must between 0 and 1. current is {damp}. Module cannot be correctly processed.")
            # raise ValueError(f"Quantization: `damp_percent` must between 0 and 1. current is {damp}")
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

        # Temporarily disable torch.compile due to compatibility issues with torch 2.8
        # Will re-enable once the issue is fixed
        if not TORCH_GTE_28 and not self.qcfg.mock_quantization:
            self.hessian_inverse = torch_compile(self.hessian_inverse)

        if self.qcfg.mock_quantization:
            # Use simplified hessian inverse (identity matrix)
            self.hessian_inverse = self._mock_hessian_inverse

        # process buffered inputs
        if len(self.fwd_inputs_buffered_data) > 0:
            torch_sync(device=self.module.target_device)

            for inp in self.fwd_inputs_buffered_data:
                self.process_batch(inp)

            # release buffer
            del self.fwd_inputs_buffered_data

        # if self.device.type not in ["mps", "cpu"]:
        #     self.module.weight.data = self.module.weight.data.cpu()

        # TODO: waiting for pytorch implementation of ops for MPS
        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

        if self.module_copy is None:
            # log.info("copy W to cuda_1")
            W = self._clone_module(device=self.module.target_device)
        else:
            W = self.module_copy.to(device=self.module.target_device)
            del self.module_copy

        self.quantizer.find_params(W, weight=True)

        H = self.H.to(device=self.module.target_device)

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
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
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        elif self.qcfg.act_group_aware:
            diag_h = torch.diag(H)
            local_perms = compute_local_perms(diag_h, self.qcfg.group_size)
            global_perm = compute_global_perm(diag_h, self.qcfg.group_size)
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            W = W[:, final_perm]
            H = H[final_perm][:, final_perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        Hinv, damp = self.hessian_inverse(H)

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

        target_device = self.module.weight.data.device

        # limit one sync tensor move action per device due to cuda limits
        if Q.device != target_device:
            dev_lock = _get_device_lock(target_device)
            with dev_lock:
                Q = Q.to(device=target_device, non_blocking=False)

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def free(self):
        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy
        del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
