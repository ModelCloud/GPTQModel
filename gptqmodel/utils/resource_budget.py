# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""What a loaded model costs on this device, and where each number came from.

Three kinds of numbers get mixed up when planning capacity, and mixing them
silently is how a wrong answer ships:

``measured``   read from the live process - weight storage, a device-memory
               snapshot. True at the moment of the read, not a promise later.
``declared``   read from the model config - the model's own maximum context.
``estimated``  derived from declared shapes and the weight dtype - the KV cache
               footprint, and everything computed from it.

Every accessor below names its units and states its kind in the docstring, and
:meth:`ResourceBudget.describe` prints the same distinction into the log.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from .kv_cache import KVCacheEstimate, human_bytes


MEASURED = "measured"
DECLARED = "declared"
ESTIMATED = "estimated"


def device_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    """``(free_bytes, total_bytes)`` of the current CUDA device (measured).

    ``(None, None)`` without CUDA: the budget then carries no device numbers
    rather than a zero that would read as "no memory left".
    """

    if not torch.cuda.is_available():
        return None, None
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover - driver edge cases
        return None, None
    return int(free), int(total)


@dataclass(frozen=True)
class ResourceBudget:
    """Measured device numbers next to estimated cache requirements.

    ``weights_bytes``
        (measured) storage of the model's parameters and buffers.
    ``device_free_bytes`` / ``device_total_bytes``
        (measured) a snapshot of the device at the moment the budget was built.
    ``kv_cache``
        (estimated) the KV-cache footprint implied by the config.
    ``reserve_fraction``
        share of free device memory held back from the KV budget.
    """

    weights_bytes: int
    device_free_bytes: Optional[int] = None
    device_total_bytes: Optional[int] = None
    kv_cache: Optional[KVCacheEstimate] = None
    reserve_fraction: float = 0.1

    #: Provenance of each accessor, so a caller can tell measured from estimated
    #: without reading the implementation.
    ACCURACY = {
        "weights_bytes": MEASURED,
        "device_free_bytes": MEASURED,
        "device_total_bytes": MEASURED,
        "kv_budget_bytes": MEASURED,
        "context_limit": DECLARED,
        "kv_cache": ESTIMATED,
        "kv_cache_bytes_for": ESTIMATED,
        "max_context_tokens": ESTIMATED,
        "max_sequences": ESTIMATED,
        "context_batch_plan": ESTIMATED,
    }

    @property
    def context_limit(self) -> Optional[int]:
        """(declared) The model's own maximum context, from its config."""

        return self.kv_cache.context_limit if self.kv_cache is not None else None

    @property
    def kv_budget_bytes(self) -> Optional[int]:
        """(measured) Free device memory now, minus ``reserve_fraction``."""

        if self.device_free_bytes is None:
            return None
        reserve = max(0.0, min(0.9, float(self.reserve_fraction)))
        return int(self.device_free_bytes * (1.0 - reserve))

    def kv_cache_bytes_for(self, context_tokens: int, *, batch_size: int = 1) -> Optional[int]:
        """(estimated) KV-cache bytes for ``context_tokens`` at ``batch_size``."""

        if self.kv_cache is None:
            return None
        return self.kv_cache.kv_bytes_for(context_tokens, batch_size=batch_size)

    def max_context_tokens(self, *, batch_size: int = 1,
                           kv_budget_bytes: Optional[int] = None) -> Optional[int]:
        """(estimated) Largest usable context at ``batch_size``, capped by the model."""

        if self.kv_cache is None:
            return None
        budget = self.kv_budget_bytes if kv_budget_bytes is None else int(kv_budget_bytes)
        if budget is None:
            return None
        return self.kv_cache.max_context_tokens(budget, batch_size=batch_size)

    def max_sequences(self, context_tokens: int, *,
                      kv_budget_bytes: Optional[int] = None) -> Optional[int]:
        """(estimated) How many sequences of ``context_tokens`` fit the KV budget."""

        if self.kv_cache is None:
            return None
        budget = self.kv_budget_bytes if kv_budget_bytes is None else int(kv_budget_bytes)
        if budget is None:
            return None
        return self.kv_cache.max_sequences(budget, context_tokens)

    def context_batch_plan(self, *, kv_budget_bytes: Optional[int] = None,
                           batches: Sequence[int] = (1, 2, 4, 8, 16, 32)) -> Tuple[Tuple[int, int], ...]:
        """(estimated) ``(batch_size, max_context_tokens)`` rows for the KV budget."""

        if self.kv_cache is None:
            return ()
        budget = self.kv_budget_bytes if kv_budget_bytes is None else int(kv_budget_bytes)
        if budget is None:
            return ()
        return self.kv_cache.context_batch_plan(budget, batches)

    def describe(self, contexts: Sequence[int] = (8192, 32768)) -> Tuple[str, ...]:
        """Log lines with every number's kind spelled out."""

        weights = f"Budget: weights {human_bytes(self.weights_bytes)} (measured)"
        if self.device_free_bytes is not None and self.device_total_bytes is not None:
            weights += (
                f", device free {human_bytes(self.device_free_bytes)}"
                f" of {human_bytes(self.device_total_bytes)} (measured)"
            )
        lines = [weights]

        if self.kv_cache is None:
            lines.append("Budget: KV cache unknown (the config does not declare its attention shapes)")
            return tuple(lines)

        cache = self.kv_cache.describe(contexts).strip()
        # describe() opens with "KV cache:"; fold it under the budget label and
        # mark the whole line as the estimate it is.
        if cache.startswith("KV cache:"):
            cache = cache[len("KV cache:"):].strip()
        lines.append(f"Budget: KV cache {cache} (estimated)")
        budget = self.kv_budget_bytes
        if budget:
            plan = self.kv_cache.format_context_batch_plan(budget)
            plan = plan.replace("KV plan: ", "", 1)
            lines.append(f"Budget: KV plan {plan} (estimated)")
        return tuple(lines)
