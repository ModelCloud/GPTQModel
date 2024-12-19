# License: GPTQModel/licenses/LICENSE.apache
# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import math
import os
import sys
import time

import torch
import torch.nn as nn
import transformers

from ..utils.logger import setup_logger
from ..utils.torch import torch_empty_cache, torch_sync
from .quantizer import Quantizer


logger = setup_logger()

# TODO do we really need max precision?
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device

        self.layer_copy = self._clone_layer()

        self.rows, self.columns = self.layer_copy.shape[0], self.layer_copy.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0
        self.quantizer = Quantizer()

    def _clone_layer(self):
        # mps for m1+ is unified memory
        if self.device.type not in ["mps", "cpu"]:
            clone = self.layer.weight.data.cpu()
        else:
            clone = self.layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d):
            clone = clone.flatten(1)

        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        return clone.to(device=self.device, dtype=torch.float)

    def add_batch(self, inp, out):
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

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
        return self.hf_quantize(
            blocksize=blocksize,
            percdamp=percdamp,
            damp_auto_increment=damp_auto_increment,
            group_size=group_size,
            actorder=actorder,
            static_groups=static_groups)

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
        return self.quantize(
            blocksize=blocksize,
            percdamp=percdamp,
            damp_auto_increment=damp_auto_increment,
            group_size=group_size,
            actorder=actorder,
            static_groups=static_groups)

    @torch.inference_mode()
    def quantize(
        self,
        blocksize=128,
        percdamp=0.01,
        damp_auto_increment=0.0015,
        group_size=-1,
        actorder=False,
        static_groups=False,
        move_to_cpu=False,
    ):
        start = time.time()
        # TODO: waiting for pytorch implementation of ops for MPS
        if sys.platform == "darwin" and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError("For MacOS you must set env `PYTORCH_ENABLE_MPS_FALLBACK=1` before running quantization.")

        if self.layer_copy is None:
            W = self._clone_layer()
        else:
            W = self.layer_copy
            self.layer_copy = None

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

            if os.environ.get("DEBUG"):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]

                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))

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

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        Q = Q.cpu()
        torch_empty_cache(self.device)

        if Q.shape != self.layer.weight.shape:
            self.layer.weight.data = (Q.reshape(self.layer.weight.shape)
                .to(dtype=self.layer.weight.data.dtype, device=torch.device("cpu") if move_to_cpu else self.device))
        else:
            self.layer.weight.data = (Q.to(dtype=self.layer.weight.data.dtype, device=torch.device("cpu") if move_to_cpu else self.device))

        if move_to_cpu:
            self.device = torch.device("cpu")

        if os.environ.get("DEBUG"):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        duration = time.time() - start
        return scale, zero, g_idx, duration, avg_loss, percdamp

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None

        self.H = None
        self.Losses = None
        self.Trace = None

        self.quantizer = None
        self.layer_copy = None

        torch_empty_cache(self.device)


__all__ = ["GPTQ"]
