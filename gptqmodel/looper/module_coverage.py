# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Coverage bookkeeping for layer-module specs that do not resolve.

``BaseQModel.layer_modules_strict = False`` exists for models whose layers do
not all expose the same modules (hybrid attention/Mamba stacks, dense and MoE
layers in one checkpoint, config-dependent LoRA modules, ...).  Before this
module existed, the loopers simply skipped a spec name that no layer contained,
so a definition that drifted from the installed Transformers layout quantized
only the attention/MLP modules it still recognized and silently left the rest
in the source dtype.

The tracker keeps per-spec hit/miss *layer* sets (not per-module counts) so an
expanded ``mlp.experts.{expert_index}.gate_proj`` spec reports once instead of
once per expert.  ``report()`` emits a warning for every spec that matched no
layer and a single informational line for specs that are only present in some
layers, which is the normal case for hybrid definitions.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, Optional

from ..utils.logger import setup_logger


log = setup_logger()

# Expanded expert indices (``mlp.experts.17.gate_proj``) are folded back into
# the placeholder form so one definition spec produces one report entry.
_NUMERIC_SEGMENT = re.compile(r"\.(\d+)(?=\.|$)")


def normalize_layer_module_name(name: str) -> str:
    """Collapse numeric path segments so expanded expert specs report once."""
    return _NUMERIC_SEGMENT.sub(".#", name)


class LayerModuleCoverageTracker:
    """Track which layer-module specs resolve, per layer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hit_layers: Dict[str, set[int]] = {}
        self._miss_layers: Dict[str, set[int]] = {}
        self._examples: Dict[str, str] = {}

    def record_hit(self, name: str, layer_index: Optional[int] = None) -> None:
        self._record(name, layer_index, hit=True)

    def record_miss(self, name: str, layer_index: Optional[int] = None) -> None:
        self._record(name, layer_index, hit=False)

    def _record(self, name: str, layer_index: Optional[int], *, hit: bool) -> None:
        key = normalize_layer_module_name(name)
        layer = -1 if layer_index is None else int(layer_index)
        with self._lock:
            bucket = self._hit_layers if hit else self._miss_layers
            bucket.setdefault(key, set()).add(layer)
            self._examples.setdefault(key, name)

    def snapshot(self) -> Dict[str, dict]:
        """Return an immutable-ish view of the collected coverage."""
        with self._lock:
            keys = set(self._hit_layers) | set(self._miss_layers)
            return {
                key: {
                    "example": self._examples[key],
                    "hit_layers": len(self._hit_layers.get(key, ())),
                    "miss_layers": len(self._miss_layers.get(key, ())),
                }
                for key in keys
            }

    def report(self, model_name: Optional[str] = None) -> None:
        """Log one warning per spec that never matched and a partial summary."""
        snapshot = self.snapshot()
        if not snapshot:
            return

        label = f" for `{model_name}`" if model_name else ""
        stale = sorted(
            ((key, stats) for key, stats in snapshot.items() if stats["hit_layers"] == 0),
            key=lambda item: item[0],
        )
        partial = sorted(
            (
                (key, stats)
                for key, stats in snapshot.items()
                if stats["hit_layers"] > 0 and stats["miss_layers"] > 0
            ),
            key=lambda item: item[0],
        )

        if stale:
            log.warn(
                f"Quantization coverage{label}: {len(stale)} layer-module spec(s) "
                "matched no layer and were NOT quantized. Fix the model definition "
                "(or mark intentionally non-quantized modules with `:!`/`:?`)."
            )
            for key, stats in stale:
                log.warn(
                    f"  - `{stats['example']}` (spec `{key}`): "
                    f"0 layer matches, {stats['miss_layers']} miss(es)"
                )
        elif snapshot:
            log.info(
                f"Quantization coverage{label}: all {len(snapshot)} layer-module "
                "spec(s) matched at least one layer."
            )

        if partial:
            for key, stats in partial:
                log.debug(
                    f"Quantization coverage{label}: optional spec `{key}` matched "
                    f"{stats['hit_layers']} layer(s), missed {stats['miss_layers']}."
                )
            log.info(
                f"Quantization coverage{label}: {len(partial)} spec(s) matched only "
                "some layers (expected for hybrid/optional modules); enable DEBUG "
                "logging for details."
            )


class LayerModuleCoverageMixin:
    """Provide a lazily created coverage tracker.

    Loopers are normally built through ``__init__``, which installs a tracker.
    Some callers construct a looper with ``__new__`` to exercise a single
    helper; the lazy property keeps those paths working without requiring every
    caller to remember the attribute.
    """

    @property
    def layer_module_coverage(self) -> LayerModuleCoverageTracker:
        tracker = self.__dict__.get("_layer_module_coverage")
        if tracker is None:
            tracker = LayerModuleCoverageTracker()
            self.__dict__["_layer_module_coverage"] = tracker
        return tracker

    @layer_module_coverage.setter
    def layer_module_coverage(self, tracker: LayerModuleCoverageTracker) -> None:
        self.__dict__["_layer_module_coverage"] = tracker


__all__ = [
    "LayerModuleCoverageMixin",
    "LayerModuleCoverageTracker",
    "normalize_layer_module_name",
]
