# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from https://github.com/Intelligent-Computing-Lab-Yale/GPTQv2
# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import math
import platform
import sys
import time
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
import transformers

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.fallback import resolve_fallback_strategy
from ..utils.logger import setup_logger
from ..utils.p2p import safe_move_to
from ..utils.torch import torch_sync
from .gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
    invert_perm,
)
from .gptq import GPTQ, _device_reduction_key
from .quantizer import _neutralize_dead_columns


log = setup_logger()


_MLX_FOEM_MIN_ELEMENTS = 256 * 1024


@lru_cache(maxsize=1)
def _mlx_foem_quantization_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _mlx_array_to_torch(value, device: torch.device) -> torch.Tensor:
    return torch.from_dlpack(value).to(device=device).contiguous()


def _quantize_foem_weight_mlx_to_torch(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    beta: float,
    sym: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    import mlx.core as mx

    from .mlx_foem import foem_quantize_weight_mlx

    if weight.device.type == "mps":
        torch.mps.synchronize()
    result = foem_quantize_weight_mlx(
        mx.from_dlpack(weight.contiguous()),
        mx.from_dlpack(inverse_hessian.contiguous()),
        bits=bits,
        group_size=group_size,
        beta=beta,
        sym=sym,
        return_residuals=True,
    )
    tensors = tuple(_mlx_array_to_torch(value, weight.device) for value in result[:3])
    residuals = _mlx_array_to_torch(result[3], torch.device("cpu"))
    diagonal = torch.diag(inverse_hessian).cpu()
    loss_terms = residuals.square() / diagonal.square().unsqueeze(0) / 2
    loss_sum = torch.sum(loss_terms, dtype=torch.float64)
    return *tensors, float(loss_sum.float().item())


def _should_use_mlx_foem_quantization(
    weight: torch.Tensor,
    inverse_hessian: torch.Tensor,
    qcfg: QuantizeConfig,
    *,
    blocksize: int,
) -> bool:
    group_size = qcfg.group_size
    return (
        _mlx_foem_quantization_available()
        and weight.device.type in {"cpu", "mps"}
        and inverse_hessian.device == weight.device
        and weight.ndim == 2
        and weight.numel() >= _MLX_FOEM_MIN_ELEMENTS
        and qcfg.foem is not None
        and qcfg.foem.alpha == 0
        and 0 <= qcfg.foem.beta <= 1
        and not qcfg.desc_act
        and not qcfg.act_group_aware
        and not qcfg.static_groups
        and not qcfg.mse
        and isinstance(qcfg.bits, int)
        and 2 <= qcfg.bits <= 8
        and group_size in {16, 32, 64, 128}
        and weight.shape[1] % group_size == 0
        and blocksize == group_size
    )


class FOEM(GPTQ):
    def __init__(self, module: NamedModule, qcfg: Optional[QuantizeConfig] = None, region_timer=None):
        from ..looper.native_processor import NATIVE_INPUTS_STATE_KEY  # avoid import loop

        super().__init__(module, qcfg, region_timer=region_timer)

        self.H = None
        self.dXXT = None
        self._foem_device_hessian = {}
        self._foem_device_dxxT = {}
        self._foem_device_counts = {}
        self._foem_active_device = None
        self._foem_active_count = 0


        self._routed_expert = bool(module.state.get("is_routed_expert", False))
        alpha_active = self.qcfg.foem.alpha != 0
        # Routed experts receive a different token subset on the quantized path
        # than on the full-precision pass, so their native inputs cannot be
        # paired by call order. Keep FOEM's beta term but disable the GPTAQ
        # (alpha) delta for them: dXXT = 0.
        self.gptaq = alpha_active and not self._routed_expert

        if alpha_active:
            self.native_inps = module.state.pop(NATIVE_INPUTS_STATE_KEY)
            if self._routed_expert:
                self.native_inps = []

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        with self.lock:
            self.fwd_counter += 1
            self.process_batch(inp, batch_index=batch_index)

    def _take_native_input(self, batch_index: Optional[int]) -> torch.Tensor:
        """Consume the native entry belonging to ``batch_index``.

        ``NativeProcessor`` tags each per-sample entry with its calibration
        batch. Scanning for that key makes pairing independent of the order in
        which data-parallel batches complete. A plain list is still accepted
        for direct callers and older captures.
        """

        entries = self.native_inps
        if not entries:
            raise RuntimeError(
                f"Quantization: Module `{self.name}` -> no native calibration input "
                f"remains for batch `{batch_index}`."
            )

        for index, entry in enumerate(entries):
            if not isinstance(entry, tuple) or len(entry) != 2:
                continue
            entry_batch, tensor = entry
            if batch_index is None or entry_batch == batch_index:
                del entries[index]
                return tensor

        if batch_index is not None and not isinstance(entries[0], tuple):
            # Legacy capture without batch keys: keep the historical pop order.
            return entries.pop(0)
        if batch_index is not None:
            raise RuntimeError(
                f"Quantization: Module `{self.name}` -> no native calibration input "
                f"for batch `{batch_index}`."
            )
        entry = entries.pop(0)
        return entry[1] if isinstance(entry, tuple) else entry

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

    def process_batch(self, inp, batch_index: Optional[int] = None):
        inp = inp.to(dtype=torch.float32)
        native_inp = None
        if self.gptaq:
            native_inp = self._take_native_input(batch_index).to(
                device=inp.device, dtype=torch.float32
            )
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            if self.gptaq:
                native_inp = native_inp.unsqueeze(0)

        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                if self.gptaq:
                    native_inp = native_inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            if self.gptaq:
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
            if self.gptaq:
                native_inp = unfold(native_inp)
                native_inp = native_inp.permute([1, 0, 2]).flatten(1)

        # FOEM reports the same token-normalized loss as GPTQ/GPTAQ. ``inp`` is
        # [columns, tokens] here, so ``shape[1]`` is the calibration token count,
        # not the number of forward requests.
        token_count = inp.shape[1]

        self._accumulate_device_batch(inp, native_inp, token_count)
        self.nsamples += token_count

    def _accumulate_device_batch(
        self,
        inp: torch.Tensor,
        native_inp: Optional[torch.Tensor],
        token_count: int,
    ) -> None:
        """Accumulate one batch into its own device's Hessian/dXXT partial."""

        device = inp.device
        use_native = self.gptaq

        if self._foem_active_device is not None and device != self._foem_active_device:
            self._foem_device_hessian[self._foem_active_device] = self.H
            if use_native:
                self._foem_device_dxxT[self._foem_active_device] = self.dXXT
            self._foem_device_counts[self._foem_active_device] = self._foem_active_count
            self._foem_active_device = None
            self._foem_active_count = 0
            self.H = None
            self.dXXT = None

        if self._foem_active_device is None and not self._foem_device_hessian:
            self._foem_active_device = device
            self._foem_active_count = 0
            self.H = torch.zeros(
                (self.columns, self.columns), dtype=torch.float32, device=device
            )
            if use_native:
                self.dXXT = torch.zeros_like(self.H)

        if self._foem_active_device is not None:
            hessian = self.H
            dxxT = self.dXXT
            device_count = self._foem_active_count
        else:
            hessian = self._foem_device_hessian.get(device)
            if hessian is None:
                hessian = torch.zeros(
                    (self.columns, self.columns), dtype=torch.float32, device=device
                )
            dxxT = self._foem_device_dxxT.get(device) if use_native else None
            if use_native and dxxT is None:
                dxxT = torch.zeros_like(hessian)
            device_count = self._foem_device_counts.get(device, 0)

        if device_count:
            ratio = device_count / (device_count + token_count)
            hessian.mul_(ratio)
            if use_native:
                dxxT.mul_(ratio)

        device_count += token_count
        scale = math.sqrt(2 / device_count)
        scaled_inp = scale * inp.float()
        hessian.add_(scaled_inp.matmul(scaled_inp.t()))
        if use_native:
            dxxT.add_((scale * native_inp - scaled_inp).matmul(scaled_inp.t()))

        if self._foem_active_device is not None:
            self._foem_active_count = device_count
        else:
            self._foem_device_hessian[device] = hessian
            if use_native:
                self._foem_device_dxxT[device] = dxxT
            self._foem_device_counts[device] = device_count

    def _finalize_foem_hessian(self, target_device: Optional[torch.device] = None) -> None:
        """Reduce per-device Hessian/dXXT partials in device order."""

        if not self._foem_device_hessian:
            return

        if target_device is None:
            target_device = getattr(self.module, "target_device", None)
        if target_device is None:
            target_device = min(self._foem_device_hessian, key=_device_reduction_key)
        target_device = torch.device(target_device)

        total = sum(self._foem_device_counts.values())
        if total <= 0:
            return

        hessian = torch.zeros(
            (self.columns, self.columns), dtype=torch.float32, device=target_device
        )
        dxxT = torch.zeros_like(hessian) if self.gptaq else None
        for device in sorted(self._foem_device_hessian, key=_device_reduction_key):
            weight = self._foem_device_counts[device] / total
            hessian.add_(
                safe_move_to(
                    self._foem_device_hessian[device],
                    target_device,
                    dtype=torch.float32,
                ),
                alpha=weight,
            )
            if dxxT is not None:
                dxxT.add_(
                    safe_move_to(
                        self._foem_device_dxxT[device],
                        target_device,
                        dtype=torch.float32,
                    ),
                    alpha=weight,
                )

        self.H = hessian
        self.dXXT = dxxT
        self._foem_device_hessian.clear()
        self._foem_device_dxxT.clear()
        self._foem_device_counts.clear()

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        self._finalize_foem_hessian()
        if self.nsamples == 0 or self.H is None:
            # No calibration rows reached this module (for example an unrouted
            # expert). GPTQ falls back to data-independent quantization; mirror
            # that here instead of dereferencing a missing Hessian.
            strategy = resolve_fallback_strategy(self.fallback)
            log.warn(
                "Quantization: Module `%s` -> FOEM received no calibration "
                "samples; using `%s` fallback quantization.",
                self.name,
                strategy.value,
            )
            if self.gptaq:
                self.native_inps = []
            self.dXXT = None
            return self._fallback_quantize(strategy, blocksize)

        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        # Keep `hessian_inverse` eager: torch.compile on this path can deadlock/hang
        # when a RuntimeError is raised inside the damp-recovery loop and the graph
        # is re-entered.

        if self.module_copy is None:
            # log.info("copy W to cuda_1")
            W = self.clone_module(device=self.H.device)
        else:
            W = self.module_copy
            self.module_copy = None

        self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        if self.gptaq:
            _neutralize_dead_columns(H, W, self.dXXT)
        else:
            _neutralize_dead_columns(H, W)

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
            if self.gptaq:
                self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)
        elif self.qcfg.act_group_aware:
            diag_h = torch.diag(H)
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
            final_perm = extend_perm_with_tail(final_perm, self.columns)
            W = W[:, final_perm]
            H = H[final_perm][:, final_perm]
            if self.gptaq:
                self.dXXT = self.dXXT[final_perm][:, final_perm]

        static_group_perm = None
        if self.qcfg.static_groups:
            if self.qcfg.desc_act:
                static_group_perm = perm
            elif self.qcfg.act_group_aware:
                static_group_perm = final_perm

        W_raw = W.detach().clone()

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        Hinv, damp = self.hessian_inverse(H)
        if Hinv is None:
            # Share GPTQ's data-independent RTN fallback instead of
            # dereferencing the missing inverse.
            self.H = None
            self.dXXT = None
            return self._rtn_fallback_after_hessian_failure(blocksize)
        if self.qcfg.foem is None:
            raise ValueError("FOEM requires `foem` configuration.")
        if self.gptaq:
            P = self.qcfg.foem.alpha * ((self.dXXT @ Hinv.T).triu(diagonal=1)) @ Hinv
            del self.dXXT

        if not self._tp_pad_cols and _should_use_mlx_foem_quantization(
            W,
            Hinv,
            self.qcfg,
            blocksize=blocksize,
        ):
            Q, scale, zero, loss_sum = _quantize_foem_weight_mlx_to_torch(
                W,
                Hinv,
                bits=self.qcfg.bits,
                group_size=self.qcfg.group_size,
                beta=self.qcfg.foem.beta,
                sym=self.qcfg.sym,
            )
            del Hinv, W_raw
            torch_sync()
            avg_loss = loss_sum / self.nsamples
            if math.isnan(avg_loss):
                raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`")
            g_idx = torch.arange(
                self.columns,
                dtype=torch.int32,
                device=Q.device,
            ) // self.qcfg.group_size
            if isinstance(self.module, transformers.Conv1D):
                Q = Q.t()
            Q = Q.reshape(self.module.weight.shape).type_as(self.module.weight.data)
            scale, zero = self._sanitize_scale_and_log(scale, zero)
            return (
                Q,
                scale,
                zero,
                g_idx,
                time.time() - start,
                avg_loss,
                damp,
                self.nsamples,
            )

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            W1_raw = W_raw[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            if self.gptaq:
                P1 = P[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                w_raw = W1_raw[:, i]
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
                        if static_group_perm is not None:
                            idx = static_group_perm[idx]

                        self.quantizer = groups[idx // self.qcfg.group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = ((w - q) - (w - w_raw) * self.qcfg.foem.beta) / d

                if self.gptaq:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                else:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                if i+1 < count:
                    W1[:, i+1] -= self.qcfg.foem.beta * (W1[:, i+1]-W1_raw[:, i+1])
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if self.gptaq:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1.matmul(P[i1:i2, i2:])
            else:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            # The in-block correction above pulls the next column back toward
            # its raw value before that column is quantized. The first column
            # of the next block needs the same correction at the block
            # boundary; the previous form targeted `W[:, i2+1]` after the loop
            # left `i == count - 1`, so it never ran and every block after the
            # first skipped beta for its first column.
            if i2 < self.columns:
                W[:, i2] -= self.qcfg.foem.beta * (W[:, i2] - W_raw[:, i2])

        del Hinv
        if self.gptaq:
            del P
        del W_raw, W1_raw, w_raw

        torch_sync()

        avg_loss = torch.sum(Losses).item() / self.nsamples

        if math.isnan(avg_loss):
            print("Losses sum item:", torch.sum(Losses).item())
            raise ValueError(f"Quantization: Failed due to `NaN` loss for `{self.name}`")

        del Losses

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = (perm // group_size).to(dtype=torch.int32, device=Q.device)
        else:
            g_idx = torch.arange(self.columns, dtype=torch.int32, device=Q.device) // group_size

        if self.qcfg.desc_act:
            invperm = invperm.to(device=Q.device)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
        elif self.qcfg.act_group_aware:
            inv_final = invert_perm(final_perm).to(device=Q.device)
            Q = Q[:, inv_final]
            if not self.qcfg.static_groups:
                inv_global_perm = invert_perm(global_perm)
                inv_global_perm_list = inv_global_perm.tolist()
                reordered_group_count = len(inv_global_perm_list)
                temp_scale = [scale[i] for i in inv_global_perm_list]
                temp_scale.extend(scale[reordered_group_count:])
                scale = temp_scale
                temp_zero = [zero[i] for i in inv_global_perm_list]
                temp_zero.extend(zero[reordered_group_count:])
                zero = temp_zero

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx = g_idx[:valid_cols]

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

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)

        scale, zero = self._sanitize_scale_and_log(scale, zero)

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def free(self):
        super().free()

        self._foem_device_hessian.clear()
        self._foem_device_dxxT.clear()
        self._foem_device_counts.clear()
        if hasattr(self, 'dXXT'):
            del self.dXXT


__all__ = ["FOEM"]
