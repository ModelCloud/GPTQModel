# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import threading
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import Module

from ..looper.loop_processor import (
    CalibrationConfig,
    ForwardConfig,
    LoopProcessor,
    ProcessorLifecycleConfig,
)
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..quantization.config import QuantizeConfig
from ..utils.logger import setup_logger
from ..utils.torch import CPU, DEVICE_1


log = setup_logger()

NATIVE_INPUTS_STATE_KEY = "native_inp"

# v2 requires that we also need to capture/store non-quantized inputs
class NativeProcessor(LoopProcessor):
    """Caches raw module inputs for later native or GPTAQ-style processing."""

    lifecycle = ProcessorLifecycleConfig(
        forward=ForwardConfig(
            required=True,
            replay_after_process=False,
            all_modules_in_single_pass=True,
            # This pass only records raw inputs. No quantization task follows
            # it, so a packed AUTO_NF12 wrapper would stay in the model tree and
            # the quantizer would find a parameterless module instead of the
            # dense source.
            allow_auto_dtype_cast=False,
        ),
    )

    def __init__(
        self,
        tokenizer,
        qcfg: QuantizeConfig,
        calibration: CalibrationConfig,
        forward: Optional[ForwardConfig] = None,
    ):
        """Initializes the processor and enables single-pass input capture."""

        super().__init__(
            tokenizer=tokenizer,
            qcfg=qcfg,
            calibration=calibration,
            forward=forward,
        )

        self.native_inp_caches = {}
        # GPTAQ and FOEM (alpha > 0) consume one native entry per *sample*:
        # GPTQProcessor preserves the batch keep-mask and calls `add_batch`
        # once per sample, while this processor captures one entry per
        # forward batch. Capturing per sample here keeps the two lists
        # aligned; other quantizers still receive one entry per batch.
        foem = getattr(qcfg, "foem", None)
        self._native_per_sample = bool(
            getattr(qcfg, "gptaq", None) is not None
            or (foem is not None and getattr(foem, "alpha", 0) != 0)
        )
        # Preserve the raw [B, S, H] input so the per-sample keep-mask can be
        # applied here exactly as GPTQProcessor applies it to its own input.
        self.preserve_batch_keep_mask = self._native_per_sample
        # Sibling modules (q/k/v, gate/up) receive the same source activation in
        # the same forward pass, so their per-sample masked entries are
        # bitwise-identical copies. Keep one entry per (batch, source tensor,
        # mask) and hand the same storage to every sibling. The cache is
        # thread-local because capture hooks run on data-parallel workers, and
        # it is dropped as soon as the active batch/mask changes so it never
        # pins a finished batch.
        self._native_share_tls = threading.local()
        # ``process()`` runs after a forward pass has captured every module of
        # the layer. Bumping the epoch there invalidates the share cache before
        # the next layer, even when a worker replays the same batch index and
        # keep-mask object and the allocator recycles the previous source
        # address.
        self._native_share_epoch = 0

    def _native_share_cache(self, batch_index: Optional[int], keep_mask) -> dict:
        """Return the sibling-share cache for the active (batch, mask) pair.

        The token changes when the capture moves to another calibration batch,
        another keep-mask, or another forward pass. Dropping the cache at that
        point releases the previous entries instead of keeping them alive for
        the rest of the layer, and prevents a recycled source address in a
        later layer from matching a stale entry.
        """

        tls = self._native_share_tls
        token = (
            self._native_share_epoch,
            batch_index,
            id(keep_mask) if keep_mask is not None else None,
        )
        if getattr(tls, "token", None) != token:
            tls.token = token
            tls.entries = {}
        return tls.entries

    def _split_batch_samples(
        self,
        inp: torch.Tensor,
        keep_mask,
        target_device,
        batch_index: Optional[int] = None,
    ) -> list:
        """Split one forward batch into per-sample masked native entries.

        ``keep_mask`` is the ``[B, S]`` sequence mask GPTQProcessor uses for
        the same forward pass. Samples with no kept tokens are skipped, which
        mirrors GPTQProcessor's per-sample `add_batch` calls. A missing or
        non-matching mask keeps the historical one-entry-per-batch layout.

        Siblings fed by the same source tensor and mask in the same batch reuse
        one entry, because their masked slices are bitwise-identical. This is
        the same aliasing the ``batch_size=1`` path already has. Consumers
        (GPTAQ/FOEM) only read the entry through ``.to``/``reshape``/``t`` and
        never mutate it in place, so the shared storage stays read-only. The
        source activation is likewise assumed to be unchanged between sibling
        hooks, which is the same assumption the ``batch_size=1`` alias path
        already makes.
        """

        if (
            not torch.is_tensor(inp)
            or inp.dim() < 3
            or not torch.is_tensor(keep_mask)
            or keep_mask.ndim != 2
            or keep_mask.shape[0] != inp.shape[0]
            or keep_mask.shape[1] != inp.shape[1]
        ):
            return [inp.to(device=target_device)]

        cache = self._native_share_cache(batch_index, keep_mask)
        key = (
            inp.untyped_storage().data_ptr(),
            inp.storage_offset(),
            tuple(inp.shape),
            tuple(inp.stride()),
            inp.dtype,
            inp.device,
        )
        shared = cache.get(key)
        if shared is None:
            mask = keep_mask.to(device=inp.device, dtype=torch.bool)
            entries = []
            for sample_index, sample_has_tokens in enumerate(mask.any(dim=1).tolist()):
                if not sample_has_tokens:
                    entries.append(None)
                    continue
                entries.append(inp[sample_index][mask[sample_index]].to(device=target_device))
            # Hold the source and mask so the storage/data_ptr in ``key`` cannot
            # be recycled by a later allocation while the entry is cached.
            shared = (inp, mask, tuple(entries))
            cache[key] = shared
        return [entry for entry in shared[2] if entry is not None]

    def set_calibration_dataset(self, calibration_dataset):
        """Rejects dataset replacement because capture setup is fixed at construction."""

        raise NotImplementedError("NativeProcessor's calibration_dataset cannot be modified")

    def preprocess(self, module: NamedModule):
        """Allocates the per-module cache used by the forward hook.

        Routed experts are marked by the module plan before this runs. Their
        native inputs cannot be paired with the quantized-path inputs (the
        router sends a different token subset), so GPTAQ/FOEM fall back to
        ``dXXT = 0`` for them and the capture is skipped to save memory. The
        empty list keeps ``module.state`` shape-compatible for the consumers.
        """

        self.native_inp_caches[module.name] = (
            None if module.state.get("is_routed_expert", False) else []
        )

    def is_skipped(self, module: NamedModule) -> bool:
        """Reports that native input capture currently runs for every eligible module."""

        # TODO: Add skipping certain modules
        return False

    def pre_process_fwd_hook(self, name: str) -> Callable[[Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """Builds the forward hook that captures detached native inputs for a module."""

        def tmp(module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
            """Copies the module input to the configured GPTAQ staging device."""

            if self.native_inp_caches.get(name) is None:
                # Routed expert: native and quantized routing differ.
                return

            # gptq is mutable.
            inp = inp[0].detach()

            if self.qcfg.gptaq is not None:
                gptaq_device = self.qcfg.gptaq.device
            elif self.qcfg.foem is not None:
                gptaq_device = self.qcfg.foem.device
            else:
                gptaq_device = "auto"
            if gptaq_device == "auto":
                target_device = DEVICE_1
            elif gptaq_device == "cpu":
                # slower but >= 4x vram memory reduction
                target_device = CPU
            elif isinstance(gptaq_device, str):
                target_device = torch.device(gptaq_device)
            elif isinstance(gptaq_device, torch.device):
                target_device = gptaq_device
            else:
                target_device = DEVICE_1

            if self._native_per_sample:
                keep_mask = getattr(getattr(self, "_mask_tls", None), "value", None)
                batch_index = self.current_batch_index()
                entries = self._split_batch_samples(
                    inp, keep_mask, target_device, batch_index=batch_index
                )
                # Tag each per-sample entry with the calibration batch it came
                # from. GPTAQ/FOEM replay batches can complete out of order
                # across devices, so the consumer must pair by this key rather
                # than by list position.
                self.native_inp_caches[name] += [(batch_index, entry) for entry in entries]
            else:
                self.native_inp_caches[name] += [inp.to(device=target_device)]
            del inp, out

        return tmp

    def process(
        self,
        module: NamedModule,
        device: torch.device = None,
        subset: Optional[Dict[str, NamedModule]] = None,
        previous_subset: Optional[Dict[str, NamedModule]] = None,
        subset_index: Optional[int] = None,
        subset_total: Optional[int] = None,
    ):
        """Moves the captured input list into module state for downstream use."""

        # A forward pass has finished capturing every module of this layer, so
        # the sibling cache for that pass is no longer needed.
        self._native_share_epoch += 1
        module.state[NATIVE_INPUTS_STATE_KEY] = self.native_inp_caches.pop(module.name) or []

    def submodule_finalize(self, module: NamedModule, model: BaseQModel, **kwargs):
        """Clears cached native inputs after module finalization."""

        module.state.pop(NATIVE_INPUTS_STATE_KEY, None)

    def finalize(self, model: BaseQModel, **kwargs):
        """Releases processor-level caches once the full loop has completed."""

        del self.native_inp_caches

    def verify_calibration_dataset(self, processor_index: int) -> bool:
        """Ensures a calibration dataset was provided before running capture."""

        if self.calibration_dataset is None:
            raise ValueError("NativeProcessor's calibration_dataset must be provided.")
        else:
            return True

    def name(self) -> str:
        """Returns the processor label used in logs and lifecycle reporting."""

        return "native"
