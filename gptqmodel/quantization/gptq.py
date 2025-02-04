# Copyright 2025 ModelCloud
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
import time

import torch
import torch.nn as nn
import transformers

from ..utils.logger import setup_logger
from ..utils.torch import torch_sync
from .quantizer import Quantizer

logger = setup_logger()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")

class GPTQ:
    def __init__(self, module: torch.nn.Module, name: str):
        self.module = module
        self.device = self.module.weight.device
        self.module_copy = self._clone_module()

        self.rows, self.columns = self.module_copy.shape[0], self.module_copy.shape[1]
        # self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0
        self.quantizer = Quantizer()

        # fwd input buffer
        self.fwd_inputs_buffered = False
        self.fwd_inputs_buffered_data = []


    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def _clone_module(self):
        clone = self.module.weight.data.clone()

        if isinstance(self.module, nn.Conv2d):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        return clone.float()

    def add_batch(self, inp, out):
        if self.fwd_inputs_buffered:
            self.fwd_inputs_buffered_data.append(inp.to(device=CPU))
        else:
            self.process_batch(inp)

    def process_batch(self, inp):
        inp = inp.to(device=self.device)
        # if os.environ.get("DEBUG"):
        #     self.inp1 = inp
        #     self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.module, nn.Linear) or isinstance(self.module, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if isinstance(self.module, nn.Conv2d):
            unfold = nn.Unfold(
                self.module.kernel_size,
                dilation=self.module.dilation,
                padding=self.module.padding,
                stride=self.module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        if not hasattr(self, "H"):
            self.H = torch.zeros((self.columns, self.columns), device=self.device)
        else:
            self.H *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    # wrapper for backward compat with optimum
    # TODO: mark for deprecation
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
    ):
        return self.quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    @torch.inference_mode()
    def quantize(
        self,
        blocksize=128,
        percdamp=0.01,
        damp_auto_increment=0.0015,
        group_size=-1,
        actorder=False,
        static_groups=False,
    ):
        start = time.time()

        # process buffered inputs
        for inp in self.fwd_inputs_buffered_data:
            self.process_batch(inp)

        # release buffer
        del self.fwd_inputs_buffered_data

        if self.device.type not in ["mps", "cpu"]:
            self.module.weight.data = self.module.weight.data.cpu()

        # TODO: waiting for pytorch implementation of ops for MPS
        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError("For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

        if self.module_copy is None:
            W = self._clone_module()
        else:
            W = self.module_copy
            self.module_copy = None

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + group_size)], weight=True)

                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        while 1 > percdamp > 0:
            try:
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(self.columns, device=self.device)
                H[diag, diag] += damp

                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H
                break
            except torch._C._LinAlgError as e:
                if damp_auto_increment != 0:
                    logger.warning(f"Current damp={percdamp:.5f} is too low, increased by {damp_auto_increment:.5f}")
                    percdamp += damp_auto_increment
                else:
                    logger.warning("Please increase damp or nsamples for calibration data to avoid the following quant error. ")
                    raise e

        if not (0 < percdamp < 1):
            raise ValueError(f"damp_percent must between 0 and 1. current is {percdamp}")

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + group_size)], weight=True)

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]

                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if os.environ.get("DEBUG"):
            #     self.layer.weight.data[:, :i2] = Q[:, :i2]
            #     self.layer.weight.data[:, i2:] = W[:, i2:]
            #
            #     logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
            #     logger.debug(torch.sum(Losses))

        torch_sync(self.device)

        avg_loss = torch.sum(Losses).item() / self.nsamples

        if math.isnan(avg_loss):
            print("Losses sum item:", torch.sum(Losses).item())
            raise ValueError("Quantization failed due to NaN loss")

        group_size = group_size if group_size != -1 else self.columns

        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]

        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            self.module.weight.data = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
        else:
            self.module.weight.data = Q.type_as(self.module.weight.data)

        # move back to self.dev
        self.module.weight.data = self.module.weight.data.to(device=self.device)

        # if os.environ.get("DEBUG"):
        #     logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        duration = time.time() - start
        return scale, zero, g_idx, duration, avg_loss, percdamp

    def free(self):
        # if os.environ.get("DEBUG"):
        #     self.inp1 = None
        #     self.out1 = None
        #     del self.inp1
        #     del self.out1

        if hasattr(self, "H"):
            del self.H
        del self.quantizer
        del self.module_copy
        del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
