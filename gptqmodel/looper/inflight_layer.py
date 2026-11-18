# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Track the layer a quantization run is currently staging.

Both loopers move a layer to the quantization device before they touch it
(``BaseQModel.pre_quantize`` and the per-module weight-only prepare step), and
the success path relies on ``post_quantize`` to move it back. A
``StopMainLoop``/exception/Ctrl-C unwinds before that step, so the failure
cleanup needs to know which layer is mid-stage and where it came from. Without
it the in-flight layer keeps accelerator tensors while the rest of the model is
back on CPU, and every later forward raises a device-mismatch error (#4546,
#5011).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..models._const import CPU
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.looper_helpers import rehome_module_to_device


log = setup_logger()


class InflightLayerTracker:
    """Remember the staged layer and its pre-run home device."""

    # Class-level default so a looper that never stages a layer needs no
    # initialization hook and cleanup stays a no-op.
    _inflight_layer: Optional[Tuple[torch.nn.Module, torch.device]] = None

    def _track_inflight_layer(
        self,
        module: torch.nn.Module,
        home_device: Optional[torch.device] = None,
    ) -> None:
        """Remember the layer being staged and its pre-run home device."""

        if not isinstance(module, torch.nn.Module):
            return
        if home_device is None:
            try:
                home_device = get_device(module)
            except Exception:
                home_device = CPU
        # A lazy shell has no usable runtime placement to return to, so the
        # model's ordinary CPU home is the only safe destination.
        if getattr(home_device, "type", None) == "meta":
            home_device = CPU
        self._inflight_layer = (module, home_device)

    def _clear_inflight_layer(self) -> None:
        """Mark the current layer as complete so failure cleanup ignores it."""

        self._inflight_layer = None

    def _restore_inflight_layer(self) -> None:
        """Return a cancelled/failed in-flight layer to its pre-run device.

        The stage moves the whole layer to the compute device before a subset
        forward. The success path relies on ``post_quantize`` to move it back,
        but a ``StopMainLoop``/exception unwinds before that step, leaving every
        parameter and buffer of the layer on the accelerator (issue #4546).

        A cancelled subset can also unwind while its capture hooks are still
        installed, and the failure path never reached the stage's release. The
        inline ``forward_hook``/``forward_hook_last`` pair would then make every
        later forward raise ``StopForward`` (issue #5009), so clear it here as a
        safety net for the windows the stage guard cannot see.
        """

        inflight = self._inflight_layer
        self._inflight_layer = None
        if inflight is None:
            return
        module, home_device = inflight
        self._clear_layer_capture_hooks(module)
        self._unwrap_hooked_linears(module)
        try:
            rehome_module_to_device(
                module,
                home_device,
                move_parameters=True,
                move_buffers=True,
                root_module=getattr(self.gptq_model, "model", None),
            )
        except Exception:
            # Cleanup must never mask the cancellation/failure that got us
            # here; ``rehome_module_to_device`` already reports per-tensor
            # failures, so this only guards unexpected structural errors.
            log.warning(
                "failed to restore in-flight quantization layer to %s",
                home_device,
                exc_info=True,
            )

    def _unwrap_failed_run_hooked_linears(self) -> None:
        """Return every float projection of a failed run to a plain linear.

        A cancelled run can be interrupted anywhere in finalization, and the
        looper clears its in-flight tracker as soon as the layer's last
        processor returns -- before the asynchronous packers drain. Recovery
        therefore sweeps the whole finished model rather than only the tracked
        layer. Quantized kernels are not ``HookedLinear``, so completed layers
        are unaffected (issue #5089).
        """

        model = getattr(self.gptq_model, "model", None)
        if model is None:
            return
        self._unwrap_hooked_linears(model)

    @staticmethod
    def _unwrap_hooked_linears(module: torch.nn.Module) -> None:
        """Return the in-flight layer's float projections to plain linears.

        A cancelled layer can hold both packed modules and float modules that
        were staged for capture. ``HookedLinear.forward`` always runs in
        inference mode, so a default grad-mode forward through a layer that
        mixes its inference-mode outputs with a packed sibling's ordinary
        outputs raises "Inference tensors cannot be saved for backward". That
        is exactly the partially quantized layer a cancelled run leaves behind,
        and the cancelled-run contract promises a plain forward works.
        """

        from ..nn_modules.hooked_linear import unwrap_hooked_linears

        try:
            unwrap_hooked_linears(module)
        except Exception:
            # Cleanup must never mask the cancellation/failure that got us
            # here; the wrapper is only a capture convenience on a dead run.
            log.warning(
                "failed to restore the in-flight layer's float projections",
                exc_info=True,
            )

    @staticmethod
    def _clear_layer_capture_hooks(module: torch.nn.Module) -> None:
        """Drop capture hooks a cancelled subset stage did not release (#5009)."""

        for submodule in module.modules():
            if getattr(submodule, "forward_hook", None) is not None:
                submodule.forward_hook = None
            if getattr(submodule, "forward_hook_last", False):
                submodule.forward_hook_last = False


__all__ = ["InflightLayerTracker"]
