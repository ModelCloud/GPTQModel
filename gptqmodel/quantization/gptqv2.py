# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from https://github.com/Intelligent-Computing-Lab-Yale/GPTQv2
# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import transformers

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.torch import TORCH_GTE_28, torch_compile, torch_sync
from .gptq import GPTQ


class GPTQv2(GPTQ):
    def __init__(self, module: NamedModule, qcfg: Optional[QuantizeConfig] = None):
        from ..looper.native_processor import NATIVE_INPUTS_STATE_KEY  # avoid import loop

        super().__init__(module, qcfg)

        self.H = None
        self.dXXT = None

        self.native_inps = module.state.pop(NATIVE_INPUTS_STATE_KEY)

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        with self.lock:
            self.fwd_counter += 1
            self.process_batch(inp)

    # TODO FIXME: using v1 new process_batch kills v2 quantization quality, use original process_batch
    # sample counter based on batch request # instead of batched token #.
    #  def process_batch(self, inp):
    #     batch_token_size, reshaped_inp, alpha, beta = super().process_batch(inp)
    #     del inp
    #
    #     native_inp = self.native_inps.pop(0).to(device=DEVICE_1, dtype=torch.float32)
    #
    #     # input reshaping
    #     if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
    #         native_inp = native_inp.reshape(-1, native_inp.shape[-1])
    #     else:
    #         unfold = nn.Unfold(
    #             self.module.kernel_size,
    #             dilation=self.module.dilation,
    #             padding=self.module.padding,
    #             stride=self.module.stride,
    #         )
    #         # output size (batch_size, channels * \prod kernel_size, num_patches)
    #         native_inp = unfold(native_inp).transpose(1, 2).flatten(0, 1)
    #
    #     if self.dXXT is None:
    #         self.dXXT = torch.zeros((self.columns, self.columns),
    #                              dtype=torch.float32,
    #                              device=DEVICE_1)
    #
    #     self.dXXT.addmm_((native_inp.T-reshaped_inp.T), reshaped_inp, beta=beta, alpha=alpha)
    #     del native_inp, reshaped_inp

    def process_batch(self, inp):
        inp = inp.to(dtype=torch.float32)
        native_inp = self.native_inps.pop(0).to(device=inp.device, dtype=torch.float32)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            native_inp = native_inp.unsqueeze(0)

        batch_size = inp.shape[0]

        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                native_inp = native_inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            native_inp = native_inp.t()

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
            native_inp = unfold(native_inp)
            native_inp = native_inp.permute([1, 0, 2]).flatten(1)

        if self.H is None:
            self.H = torch.zeros((self.columns, self.columns),
                                 dtype=torch.float32,
                                 device=inp.device)
            self.dXXT = self.H.clone()
        else:
            self.H *= self.nsamples / (self.nsamples + batch_size)
            self.dXXT *= self.nsamples / (self.nsamples + batch_size)

        self.nsamples += batch_size
        inp = math.sqrt(2 / self.nsamples) * inp.float()

        self.H += inp.matmul(inp.t())
        native_inp = math.sqrt(2 / self.nsamples) * native_inp
        self.dXXT += (native_inp - inp).matmul(inp.t())

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        # TODO compilation failure for Torch >= 2.8
        if not TORCH_GTE_28:
            self.hessian_inverse = torch_compile(self.hessian_inverse)

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
            W = self.module_copy
            self.module_copy = None

        self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        self.dXXT[:, dead] = 0

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
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        Hinv, damp = self.hessian_inverse(H)
        P = self.qcfg.v2_alpha * ((self.dXXT @ Hinv.T).triu(diagonal=1)) @ Hinv
        del self.dXXT

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        if (i1 + i) % self.qcfg.group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i): (i1 + i + self.qcfg.group_size)], weight=True)

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
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(
                    P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])

        del Hinv
        del P

        torch_sync()

        avg_loss = torch.sum(Losses).item() / self.nsamples

        if math.isnan(avg_loss):
            print("Losses sum item:", torch.sum(Losses).item())
            raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`")

        del Losses

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]

        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if self.qcfg.desc_act:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
        else:
            Q = Q.type_as(self.module.weight.data)

        # Q = Q.to(device=DEVICE_1)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def free(self):
        super().free()

        if hasattr(self, 'dXXT'):
            del self.dXXT


__all__ = ["GPTQv2"]
