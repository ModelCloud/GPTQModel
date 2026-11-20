# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from https://github.com/Intelligent-Computing-Lab-Yale/GPTQv2
# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..utils.fallback import resolve_fallback_strategy, should_use_fallback
from ..utils.logger import setup_logger
from ..utils.p2p import safe_move_to
from ..utils.torch import torch_sync
from .conv_padding import conv_padding_pairs, pad_mode
from .gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
    invert_perm,
)
from .gptq import GPTQ, _device_reduction_key, _move_static_groups
from .quantizer import _neutralize_dead_columns


log = setup_logger()


class GPTAQ(GPTQ):
    """GPTAQ quantizer with GPTQ-compatible public sample and loss reporting.

    GPTAQ's upstream Hessian recurrence is normalized by calibration requests.
    Keep that internal normalization intact while exposing token counts and a
    token-normalized average loss so logs remain comparable with GPTQ.
    """

    def __init__(self, module: NamedModule, qcfg: Optional[QuantizeConfig] = None, region_timer=None):
        from ..looper.native_processor import NATIVE_INPUTS_STATE_KEY  # avoid import loop

        super().__init__(module, qcfg, region_timer=region_timer)

        self.H = None
        self.dXXT = None
        self._gptaq_batch_request_count = 0
        # GPTAQ accumulates a private Hessian/dXXT on each input device and
        # reduces them in device order. ``self.H``/``self.dXXT`` remain the
        # live single-device buffers until a second device appears.
        self._gptaq_device_hessian = {}
        self._gptaq_device_dxxT = {}
        self._gptaq_device_counts = {}
        self._gptaq_active_device = None
        self._gptaq_active_count = 0

        self.native_inps = module.state.pop(NATIVE_INPUTS_STATE_KEY)
        # Routed experts receive a different token subset on the quantized path
        # than they did on the full-precision pass, so `native_inps` cannot be
        # paired by call order. Fall back to dXXT = 0 (plain GPTQ) for them;
        # the structural mark comes from the model's module plan.
        self._use_native_delta = not bool(module.state.get("is_routed_expert", False))
        if not self._use_native_delta:
            self.native_inps = []

    def add_batch(self, inp: torch.Tensor, out: Optional[torch.Tensor], batch_index: Optional[int] = None):
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

    def process_batch(
        self, inp: torch.Tensor, batch_index: Optional[int] = None
    ) -> Tuple[int, Optional[torch.Tensor], torch.device]:
        """Accumulate one batch; ``batch_index`` pairs it with its native input."""
        inp = inp.to(dtype=torch.float32)
        native_inp = None
        if self._use_native_delta:
            native_inp = self._take_native_input(batch_index).to(
                device=inp.device, dtype=torch.float32
            )
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            if self._use_native_delta:
                native_inp = native_inp.unsqueeze(0)

        batch_request_count = inp.shape[0]

        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, inp.shape[-1]))
                if self._use_native_delta:
                    native_inp = native_inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
            if self._use_native_delta:
                native_inp = native_inp.t()

        if isinstance(self.module, nn.Conv2d):
            # `nn.Unfold` only accepts integer zero padding, so apply the
            # layer's own padding (including its mode) and unfold the
            # already-padded input. This keeps the Hessian on the operator
            # that will actually run, matching GPTQ's capture (#3917, #5646).
            (top, bottom), (left, right) = conv_padding_pairs(self.module)
            pad = (left, right, top, bottom)
            mode = pad_mode(self.module.padding_mode)
            unfold = nn.Unfold(
                self.module.kernel_size,
                dilation=self.module.dilation,
                padding=0,
                stride=self.module.stride,
            )
            inp = unfold(F.pad(inp, pad, mode=mode))
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
            if self._use_native_delta:
                native_inp = unfold(F.pad(native_inp, pad, mode=mode))
                native_inp = native_inp.permute([1, 0, 2]).flatten(1)

        self._accumulate_device_batch(inp, native_inp, batch_request_count)
        self._gptaq_batch_request_count += batch_request_count
        token_count = inp.shape[1]
        self.nsamples += token_count
        # The Hessian partial is accumulated per device above, so no per-batch
        # ``xtx`` is handed back to the caller.
        return token_count, None, inp.device

    def _accumulate_device_batch(
        self,
        inp: torch.Tensor,
        native_inp: Optional[torch.Tensor],
        batch_request_count: int,
    ) -> None:
        """Accumulate one batch into its own device's Hessian/dXXT partial."""

        device = inp.device
        use_native = self._use_native_delta

        if self._gptaq_active_device is not None and device != self._gptaq_active_device:
            # A second device appeared: demote the live single-device buffers
            # into the partial map and continue accumulating per device.
            self._gptaq_device_hessian[self._gptaq_active_device] = self.H
            if use_native:
                self._gptaq_device_dxxT[self._gptaq_active_device] = self.dXXT
            self._gptaq_device_counts[self._gptaq_active_device] = self._gptaq_active_count
            self._gptaq_active_device = None
            self._gptaq_active_count = 0
            self.H = None
            self.dXXT = None

        if self._gptaq_active_device is None and not self._gptaq_device_hessian:
            # First batch: keep the historical single-device live buffers so
            # direct callers can inspect ``task.H``/``task.dXXT``.
            self._gptaq_active_device = device
            self._gptaq_active_count = 0
            self.H = torch.zeros(
                (self.columns, self.columns), dtype=torch.float32, device=device
            )
            if use_native:
                self.dXXT = torch.zeros_like(self.H)

        if self._gptaq_active_device is not None:
            hessian = self.H
            dxxT = self.dXXT
            device_count = self._gptaq_active_count
        else:
            hessian = self._gptaq_device_hessian.get(device)
            if hessian is None:
                hessian = torch.zeros(
                    (self.columns, self.columns), dtype=torch.float32, device=device
                )
            dxxT = self._gptaq_device_dxxT.get(device) if use_native else None
            if use_native and dxxT is None:
                dxxT = torch.zeros_like(hessian)
            device_count = self._gptaq_device_counts.get(device, 0)

        if device_count:
            ratio = device_count / (device_count + batch_request_count)
            hessian.mul_(ratio)
            if use_native:
                dxxT.mul_(ratio)

        device_count += batch_request_count
        scale = math.sqrt(2 / device_count)
        scaled_inp = scale * inp.float()
        hessian.add_(scaled_inp.matmul(scaled_inp.t()))
        if use_native:
            dxxT.add_((scale * native_inp - scaled_inp).matmul(scaled_inp.t()))

        if self._gptaq_active_device is not None:
            self._gptaq_active_count = device_count
        else:
            self._gptaq_device_hessian[device] = hessian
            if use_native:
                self._gptaq_device_dxxT[device] = dxxT
            self._gptaq_device_counts[device] = device_count

    def _finalize_gptaq_hessian(self, target_device: Optional[torch.device] = None) -> None:
        """Reduce per-device Hessian/dXXT partials in device order."""

        if not self._gptaq_device_hessian:
            return

        if target_device is None:
            target_device = getattr(self.module, "target_device", None)
        if target_device is None:
            target_device = min(self._gptaq_device_hessian, key=_device_reduction_key)
        target_device = torch.device(target_device)

        total = sum(self._gptaq_device_counts.values())
        if total <= 0:
            return

        hessian = torch.zeros(
            (self.columns, self.columns), dtype=torch.float32, device=target_device
        )
        dxxT = torch.zeros_like(hessian) if self._use_native_delta else None
        for device in sorted(self._gptaq_device_hessian, key=_device_reduction_key):
            weight = self._gptaq_device_counts[device] / total
            hessian.add_(
                safe_move_to(
                    self._gptaq_device_hessian[device],
                    target_device,
                    dtype=torch.float32,
                ),
                alpha=weight,
            )
            if dxxT is not None:
                dxxT.add_(
                    safe_move_to(
                        self._gptaq_device_dxxT[device],
                        target_device,
                        dtype=torch.float32,
                    ),
                    alpha=weight,
                )

        self.H = hessian
        self.dXXT = dxxT
        self._gptaq_device_hessian.clear()
        self._gptaq_device_dxxT.clear()
        self._gptaq_device_counts.clear()

    def _align_gptaq_hessian_device(self) -> None:
        """Keep ``dXXT`` on the same device as ``H`` before quantization.

        ``_finalize_gptaq_hessian`` returns early when every batch came from
        one device. A caller that rehomed only ``H`` (issue #4312) then leaves
        ``dXXT`` behind, and ``_neutralize_dead_columns`` mixes devices.
        """

        if self.H is None or self.dXXT is None:
            return
        if self.dXXT.device != self.H.device:
            self.dXXT = safe_move_to(self.dXXT, self.H.device, dtype=torch.float32)

    def adopt_hessian_from(self, leader: "GPTAQ") -> None:
        """Copy a sibling leader's ``H`` and ``dXXT`` instead of recomputing both.

        ``GPTAQ.add_batch`` depends only on the module input and the paired
        native input; both are bitwise identical for the members of one
        explicit ``:in=<tag>`` group (``NativeProcessor`` shares the sibling
        native entry storage).  The per-device partials are reduced on the
        leader first, so the follower holds exactly one private ``H`` and one
        private ``dXXT`` — the same two matrices it would have accumulated.
        """

        if leader is self:
            return
        if type(leader) is not type(self):
            raise TypeError(
                f"GPTAQ: cannot share Hessian with `{type(leader).__name__}` task `{leader.name}`."
            )
        if leader.columns != self.columns:
            raise ValueError(
                f"GPTAQ: cannot share Hessian from `{leader.name}` ({leader.columns} columns) "
                f"with `{self.name}` ({self.columns} columns)."
            )
        if leader._use_native_delta != self._use_native_delta:
            raise ValueError(
                f"GPTAQ: cannot share Hessian from `{leader.name}` (native delta="
                f"{leader._use_native_delta}) with `{self.name}` (native delta="
                f"{self._use_native_delta})."
            )

        leader._finalize_gptaq_hessian()
        with leader.lock:
            hessian = leader.H
            dxxT = leader.dXXT
            nsamples = leader.nsamples
            fwd_counter = leader.fwd_counter
            batch_request_count = leader._gptaq_batch_request_count
            if hessian is None:
                target_device = torch.device(getattr(self, "dev", "cpu"))
                hessian_copy = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=target_device,
                )
                dxxT_copy = None
            else:
                target_device = hessian.device
                hessian_copy = safe_move_to(
                    hessian.detach(), target_device, dtype=torch.float32, copy=True
                )
                dxxT_copy = (
                    None
                    if dxxT is None
                    else safe_move_to(
                        dxxT.detach(), target_device, dtype=torch.float32, copy=True
                    )
                )

        with self.lock:
            self.H = hessian_copy
            self.dXXT = dxxT_copy
            self.nsamples = nsamples
            self.fwd_counter = fwd_counter
            self._gptaq_batch_request_count = batch_request_count
            self._gptaq_device_hessian.clear()
            self._gptaq_device_dxxT.clear()
            self._gptaq_device_counts.clear()
            self._gptaq_active_device = None
            self._gptaq_active_count = 0
            # The follower no longer replays batches, so its native inputs are
            # dead references to the shared sibling storage.
            self.native_inps = []

    @staticmethod
    def _token_normalized_avg_loss(loss_sum: float, batch_request_count: int, token_count: int) -> float:
        """Convert GPTAQ's batch-normalized loss sum to GPTQ's token-normalized mean.

        GPTAQ scales its Hessian by ``2 / batch_request_count`` while GPTQ uses
        ``2 / token_count``. ``hessian_inverse`` returns the upper Cholesky
        factor of the inverse Hessian, so that scale changes the stored
        inverse-factor diagonal by its square root and the squared loss terms
        by its inverse. Convert that one linear factor before the final token
        average.
        """

        if token_count <= 0:
            raise ValueError("GPTAQ loss normalization requires at least one calibration token.")

        effective_batch_count = batch_request_count if batch_request_count > 0 else token_count
        batch_to_token = effective_batch_count / token_count
        return loss_sum * batch_to_token / token_count

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        self._finalize_gptaq_hessian()
        # Mirror GPTQ's shared low-sample decision: an unrouted expert can see a
        # few calibration rows, which is still not enough to trust its Hessian
        # (#5377).
        below_threshold = should_use_fallback(
            self.fallback,
            float(self.nsamples),
            self.expected_nsamples,
        )
        if self.nsamples == 0 or self.H is None or below_threshold:
            strategy = resolve_fallback_strategy(self.fallback)
            if self.nsamples == 0 or self.H is None:
                log.warn(
                    "Quantization: Module `%s` -> GPTAQ received no calibration "
                    "samples; using `%s` fallback quantization.",
                    self.name,
                    strategy.value,
                )
            else:
                log.warn(
                    "Quantization: Module `%s` -> GPTAQ observed %s samples, "
                    "below the fallback threshold; using `%s` fallback quantization.",
                    self.name,
                    self.nsamples,
                    strategy.value,
                )
            self.native_inps = []
            self.dXXT = None
            return self._fallback_quantize(strategy, blocksize)

        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        # Keep `hessian_inverse` eager: torch.compile on this path can deadlock/hang
        # when a RuntimeError is raised inside the damp-recovery loop and the graph
        # is re-entered.

        # Mirror GPTQ: the Hessian's device is the canonical quantization
        # device. Move a pre-cloned weight and the auxiliary `dXXT` there so
        # `_neutralize_dead_columns` never mixes devices (issue #4312).
        if self.module_copy is None:
            # log.info("copy W to cuda_1")
            W = self.clone_module(device=self.H.device)
        else:
            W = safe_move_to(self.module_copy, self.H.device)
            self.module_copy = None
        self._align_gptaq_hessian_device()

        H = self.H
        del self.H
        if self._use_native_delta:
            _neutralize_dead_columns(H, W, self.dXXT)
        else:
            _neutralize_dead_columns(H, W)

        # Fit the whole-row grid after dead-column masking: a channel absent
        # from every calibration sample carries no signal, so its weight must
        # not set the per-channel scale that every live column is quantized
        # with. GPTQ/QQQ got this ordering in #5611; the subclasses that
        # re-implement the block pass kept the old order (#5714).
        self.quantizer.find_params(W, weight=True)

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
            if self._use_native_delta:
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
            if self._use_native_delta:
                self.dXXT = self.dXXT[final_perm][:, final_perm]

        static_group_perm = None
        if self.qcfg.static_groups:
            if self.qcfg.desc_act:
                static_group_perm = perm
            elif self.qcfg.act_group_aware:
                static_group_perm = final_perm

        if self.qcfg.gptaq is None:
            raise ValueError("GPTAQ requires `gptaq` configuration.")
        result_device = torch.device(self.module.weight.data.device)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        def _inverse_and_delta():
            inverse, used_damp = self.hessian_inverse(H)
            if inverse is None or not self._use_native_delta:
                return inverse, used_damp, None
            # The native-delta product is another ``columns ** 2`` allocation,
            # so a solver workspace failure can surface here as well.
            return (
                inverse,
                used_damp,
                self.qcfg.gptaq.alpha
                * ((self.dXXT @ inverse.T).triu(diagonal=1))
                @ inverse,
            )

        def _prepare_cpu(cdu_device):
            nonlocal H, W, Q, Losses, perm, invperm, final_perm, global_perm
            nonlocal static_group_perm, scale, zero
            H = H.to(device=cdu_device)
            W = W.to(device=cdu_device)
            # ``Q``/``Losses`` were sized from the accelerator weight, but the
            # block pass now writes host tensors.
            Q = torch.zeros_like(W)
            Losses = torch.zeros_like(W)
            if self._use_native_delta:
                self.dXXT = self.dXXT.to(device=cdu_device)
            if self.qcfg.desc_act:
                # ``perm``/``invperm`` were built from the accelerator Hessian;
                # the epilogue indexes the host ``Q`` with them (issue #4538).
                perm = perm.to(device=cdu_device)
                invperm = invperm.to(device=cdu_device)
            elif self.qcfg.act_group_aware and not self.qcfg.static_groups:
                # The epilogue inverts ``global_perm`` on ``Q.device``.
                global_perm = global_perm.to(device=cdu_device)
            if self.qcfg.act_group_aware:
                final_perm = final_perm.to(device=cdu_device)
            if self.qcfg.static_groups:
                if self.qcfg.desc_act:
                    static_group_perm = perm
                elif self.qcfg.act_group_aware:
                    static_group_perm = final_perm
            if self.qcfg.group_size == -1 and self.qcfg.desc_act:
                # An ungrouped ``desc_act`` module never refits the quantizer in
                # the block loop, so rebuild the pre-reorder measurement from
                # the natural column order to match a run that never left the
                # host.
                self.quantizer.find_params(W[:, invperm], weight=True)
            else:
                self.quantizer.find_params(W, weight=True)
            if self.qcfg.static_groups:
                scale, zero = _move_static_groups(groups, cdu_device)

        (Hinv, damp, P), cpu_fallback_used = self._run_with_accelerator_oom_cpu_fallback(
            _inverse_and_delta,
            source_device=H.device,
            stage="Hessian inverse",
            prepare_cpu=_prepare_cpu,
        )
        if Hinv is None:
            # Share GPTQ's data-independent RTN fallback instead of
            # dereferencing the missing inverse.
            self.H = None
            self.dXXT = None
            return self._rtn_fallback_after_hessian_failure(blocksize)
        if self._use_native_delta:
            del self.dXXT

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2] if P is not None else None
            # The in-block updates below mutate ``W1``. The deferred native
            # delta product must see each column's pre-update working value,
            # exactly like the in-block ``P1`` term does; reading ``W1`` after
            # the loop would apply the correction to the post-update weight
            # (#5647).
            W1_pre = torch.empty_like(W1) if P is not None else None

            for i in range(count):
                w = W1[:, i]
                if W1_pre is not None:
                    W1_pre[:, i] = w
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

                err1 = (w - q) / d
                if P1 is None:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                else:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) - w.unsqueeze(1).matmul(
                        P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            if P is None:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            else:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) - W1_pre.matmul(P[i1:i2, i2:])

        del Hinv
        del P

        torch_sync()

        avg_loss = self._token_normalized_avg_loss(
            torch.sum(Losses).item(),
            self._gptaq_batch_request_count,
            self.nsamples,
        )

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
                # Dynamic parameters were measured in reordered group order;
                # restore them to the natural group order the packer expects.
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

        if cpu_fallback_used:
            log.info(
                "Quantization: Module `%s` -> CPU fallback complete; moving final quantized weights back to %s.",
                self.name,
                result_device,
            )
        Q = safe_move_to(Q, result_device)

        duration = time.time() - start

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def free(self):
        super().free()

        self._gptaq_device_hessian.clear()
        self._gptaq_device_dxxT.clear()
        self._gptaq_device_counts.clear()
        if hasattr(self, 'dXXT'):
            del self.dXXT


__all__ = ["GPTAQ"]
