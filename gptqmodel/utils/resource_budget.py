# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""What a loaded model costs on this device set, and where each number came from.

Three kinds of numbers get mixed up when planning capacity, and mixing them
silently is how a wrong answer ships:

``measured``   read from the live process - per-device weight storage, a
               per-device memory snapshot. True at the moment of the read.
``declared``   read from the model config - the model's own maximum context.
``estimated``  derived from declared shapes and the weight dtype - the KV cache
               footprint, and everything computed from it.

A sharded model is not one budget. Each device holds the weights and the KV
cache of the layers assigned to it, so planning takes the *minimum* over devices
of what each device's own share of the layers can hold. When the layers' device
assignment cannot be resolved, planning returns ``None`` instead of quietly
assuming one card.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple

import torch

from .kv_cache import KVCacheEstimate, human_bytes


MEASURED = "measured"
DECLARED = "declared"
ESTIMATED = "estimated"


@dataclass(frozen=True)
class DeviceBudget:
    """One device's measured numbers."""

    index: int
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    weights_bytes: int = 0

    def kv_budget_bytes(self, reserve_fraction: float) -> Optional[int]:
        """(measured) Free memory on this device minus the reserve."""

        if self.free_bytes is None:
            return None
        reserve = max(0.0, min(0.9, float(reserve_fraction)))
        return int(self.free_bytes * (1.0 - reserve))


def device_memory_bytes(index: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """``(free_bytes, total_bytes)`` for one CUDA device (measured snapshot)."""

    if not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info() if index is None else torch.cuda.mem_get_info(index)
    except Exception:  # pragma: no cover - driver edge cases
        return None, None
    return int(free), int(total)


@dataclass(frozen=True)
class ResourceBudget:
    """Measured device numbers next to estimated cache requirements.

    ``devices``
        (measured) one entry per device the weights live on: free and total
        memory at build time, plus the weight bytes stored there.
    ``kv_cache``
        (estimated) the KV-cache footprint implied by the config.
    ``layer_devices``
        (measured) device index of each decoder layer, aligned with
        ``kv_cache.layers``; ``None`` when the mapping cannot be resolved.
    ``reserve_fraction``
        share of free device memory held back from the KV budget.
    """

    devices: Tuple[DeviceBudget, ...] = ()
    kv_cache: Optional[KVCacheEstimate] = None
    layer_devices: Optional[Tuple[int, ...]] = None
    reserve_fraction: float = 0.1

    #: Provenance of each accessor, so a caller can tell measured from estimated
    #: without reading the implementation.
    ACCURACY = {
        "devices": MEASURED,
        "weights_bytes": MEASURED,
        "kv_budget_bytes": MEASURED,
        "context_limit": DECLARED,
        "kv_cache": ESTIMATED,
        "kv_cache_bytes_for": ESTIMATED,
        "max_context_tokens": ESTIMATED,
        "max_sequences": ESTIMATED,
        "context_batch_plan": ESTIMATED,
        "format_plan": ESTIMATED,
    }

    @property
    def weights_bytes(self) -> int:
        """(measured) Storage of the model's parameters and buffers, all devices."""

        return sum(device.weights_bytes for device in self.devices)

    @property
    def device_count(self) -> int:
        return len(self.devices)

    @property
    def cuda_devices(self) -> Tuple[DeviceBudget, ...]:
        """Devices with real CUDA memory; index ``-1`` is host memory."""

        return tuple(device for device in self.devices if device.index >= 0)

    @property
    def host_weights_bytes(self) -> int:
        """(measured) Weight bytes held in host memory rather than VRAM."""

        return sum(device.weights_bytes for device in self.devices if device.index < 0)

    @property
    def is_sharded(self) -> bool:
        """True when the weights live on more than one CUDA device.

        A CPU-resident buffer (transformers keeps ``rotary_emb.inv_freq`` on the
        host) must not make a single-GPU load look sharded.
        """

        return len(self.cuda_devices) > 1

    @property
    def context_limit(self) -> Optional[int]:
        """(declared) The model's own maximum context, from its config."""

        return self.kv_cache.context_limit if self.kv_cache is not None else None

    @property
    def kv_budget_bytes(self) -> Optional[int]:
        """(measured) Sum of the per-device KV budgets.

        The sum is the aggregate the whole model can use *if the cache is
        distributed like the layers*; :meth:`max_context_tokens` does the
        per-device arithmetic and does not rely on that assumption.
        """

        budgets = [device.kv_budget_bytes(self.reserve_fraction) for device in self.cuda_devices]
        known = [budget for budget in budgets if budget is not None]
        return sum(known) if known else None

    def kv_cache_bytes_for(self, context_tokens: int, *, batch_size: int = 1) -> Optional[int]:
        """(estimated) KV-cache bytes for ``context_tokens`` at ``batch_size``."""

        if self.kv_cache is None:
            return None
        return self.kv_cache.kv_bytes_for(context_tokens, batch_size=batch_size)

    def _layer_subset(self, indices: Sequence[int]) -> Optional[KVCacheEstimate]:
        if self.kv_cache is None or not indices:
            return None
        return replace(self.kv_cache, layers=tuple(self.kv_cache.layers[i] for i in indices))

    def _per_device_limits(self, *, batch_size: int,
                           kv_budget_bytes: Optional[int]) -> Optional[Dict[int, int]]:
        """Per-device maximum context, or None when the mapping is unknown.

        A device whose layers hold no growing cache (linear attention, KV-shared
        or non-attention blocks) does not constrain the context: its own
        ``max_context_tokens`` is 0 by construction, and counting that 0 would
        report "no usable context" for a model whose other device has plenty.
        """

        if self.kv_cache is None or self.layer_devices is None:
            return None
        if len(self.layer_devices) != len(self.kv_cache.layers):
            return None

        limits: Dict[int, int] = {}
        for device in self.cuda_devices:
            owned = [i for i, index in enumerate(self.layer_devices) if index == device.index]
            if not owned:
                continue
            subset = self._layer_subset(owned)
            if subset is None or not subset.growing_layers:
                continue
            budget = device.kv_budget_bytes(self.reserve_fraction) if kv_budget_bytes is None else kv_budget_bytes
            if budget is None:
                return None
            limits[device.index] = subset.max_context_tokens(budget, batch_size=batch_size)
        return limits or None

    def max_context_tokens(self, *, batch_size: int = 1,
                           kv_budget_bytes: Optional[int] = None) -> Optional[int]:
        """(estimated) Largest usable context at ``batch_size``, capped by the model.

        With an explicit ``kv_budget_bytes`` the whole model is planned against
        that number. Otherwise the budget is per device: the answer is the
        minimum over devices of what each device's layers can hold.
        """

        if self.kv_cache is None:
            return None
        if kv_budget_bytes is not None:
            return self.kv_cache.max_context_tokens(int(kv_budget_bytes), batch_size=batch_size)
        if not self.is_sharded:
            devices = self.cuda_devices
            budget = devices[0].kv_budget_bytes(self.reserve_fraction) if devices else None
            return None if budget is None else self.kv_cache.max_context_tokens(budget, batch_size=batch_size)

        limits = self._per_device_limits(batch_size=batch_size, kv_budget_bytes=None)
        return min(limits.values()) if limits else None

    def max_sequences(self, context_tokens: int, *,
                      kv_budget_bytes: Optional[int] = None) -> Optional[int]:
        """(estimated) How many sequences of ``context_tokens`` fit the KV budget."""

        if self.kv_cache is None:
            return None
        if kv_budget_bytes is not None:
            return self.kv_cache.max_sequences(int(kv_budget_bytes), context_tokens)
        if not self.is_sharded:
            devices = self.cuda_devices
            budget = devices[0].kv_budget_bytes(self.reserve_fraction) if devices else None
            return None if budget is None else self.kv_cache.max_sequences(budget, context_tokens)

        if self.layer_devices is None or len(self.layer_devices) != len(self.kv_cache.layers):
            return None
        limits = []
        for device in self.cuda_devices:
            owned = [i for i, index in enumerate(self.layer_devices) if index == device.index]
            subset = self._layer_subset(owned)
            budget = device.kv_budget_bytes(self.reserve_fraction)
            if subset is None or not subset.growing_layers or budget is None:
                continue
            limits.append(subset.max_sequences(budget, context_tokens))
        return min(limits) if limits else None

    def context_batch_plan(self, *, kv_budget_bytes: Optional[int] = None,
                           batches: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> Tuple[Tuple[int, int], ...]:
        """(estimated) ``(batch_size, max_context_tokens)`` rows for the KV budget."""

        if self.kv_cache is None:
            return ()
        rows = []
        for batch in batches:
            context = self.max_context_tokens(batch_size=batch, kv_budget_bytes=kv_budget_bytes)
            if context:
                rows.append((int(batch), int(context)))
        return tuple(rows)

    def format_plan(self, *, kv_budget_bytes: Optional[int] = None,
                    batches: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> Optional[str]:
        """(estimated) One line: the context/batch plan this budget buys.

        A sharded model plans each device's own layers against that device's own
        budget and reports the minimum. The line is therefore labelled with the
        binding device's budget, not the sum of every device's free memory - the
        sum is memory no single card can spend on the whole layer stack.
        """

        if self.kv_cache is None:
            return None
        rows = self.context_batch_plan(kv_budget_bytes=kv_budget_bytes, batches=batches)
        if kv_budget_bytes is not None:
            return self.kv_cache.format_plan_rows(
                rows, kv_budget_bytes=int(kv_budget_bytes), suffix=" (estimated)")
        if not self.is_sharded:
            budget = self.kv_budget_bytes
            if not budget:
                return None
            return self.kv_cache.format_plan_rows(rows, kv_budget_bytes=budget, suffix=" (estimated)")
        limits = self._per_device_limits(batch_size=1, kv_budget_bytes=None)
        if not limits:
            return None
        binding = min(limits, key=limits.get)
        budget = next(
            (device.kv_budget_bytes(self.reserve_fraction)
             for device in self.cuda_devices if device.index == binding),
            None,
        )
        # When the model's own context limit binds, no device is the limit; the
        # rows formatter already says "(model limit)".
        capped_by_model = self.context_limit is not None and limits[binding] >= self.context_limit
        suffix = " (estimated)" if capped_by_model else f" (estimated, limited by device {binding})"
        return self.kv_cache.format_plan_rows(
            rows,
            kv_budget_bytes=budget,
            suffix=suffix,
        )

    def describe(self, contexts: Sequence[int] = (8192, 32768)) -> Tuple[str, ...]:
        """Log lines with every number's kind spelled out."""

        weights = f"Budget: weights {human_bytes(self.weights_bytes)} (measured)"
        if self.devices:
            parts = []
            for device in self.cuda_devices:
                if device.free_bytes is None or device.total_bytes is None:
                    parts.append(f"device {device.index} free unknown")
                    continue
                parts.append(
                    f"device {device.index} free {human_bytes(device.free_bytes)}"
                    f" of {human_bytes(device.total_bytes)}"
                )
            if parts:
                weights += ", " + ", ".join(parts) + " (measured)"
            if self.host_weights_bytes:
                weights += f", host {human_bytes(self.host_weights_bytes)} (measured)"
        lines = [weights]

        if self.kv_cache is None:
            lines.append("Budget: KV cache unknown (the config does not declare its attention shapes)")
            return tuple(lines)

        cache = self.kv_cache.describe(contexts).strip()
        if cache.startswith("KV cache:"):
            cache = cache[len("KV cache:"):].strip()
        lines.append(f"Budget: KV cache {cache} (estimated)")

        plan = self.format_plan()
        if plan:
            # `format_plan` is a complete line; keep the historical log prefix.
            body = plan[len("KV plan: "):] if plan.startswith("KV plan: ") else plan
            lines.append(f"Budget: KV plan {body}")
        elif self.is_sharded:
            lines.append(
                "Budget: KV plan unavailable (a sharded model needs its layer-to-device map"
                " and measured free memory)"
            )
        return tuple(lines)
