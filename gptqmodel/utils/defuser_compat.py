# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Compatibility shims for Defuser blocks that predate a Transformers change.

Defuser 0.0.30 replaces the GLM-4.x MoE block with a per-expert module whose
``forward`` passes the router result straight into ``route_tokens_to_experts``.
Transformers 5.22 routers return ``(router_logits, topk_weights, topk_indices)``
instead of the logits tensor, so the replacement calls ``.sigmoid()`` on a tuple
and every GLM-4.5/4.6/4.7 and GLM-4.7-Flash quantization run fails at the first
MoE layer with ``AttributeError: 'tuple' object has no attribute 'sigmoid'``.

The authoritative fix belongs in Defuser. Until the pinned Defuser release
unpacks the 5.22 tuple itself, this module wraps the replacement's
``route_tokens_to_experts`` so it accepts either shape. The wrapper is only
installed when the installed Defuser still forwards the raw router result, and
it delegates to the original method after unwrapping, so a later Defuser fix is
not masked.
"""

from __future__ import annotations

import inspect
from typing import Tuple

from .logger import setup_logger


log = setup_logger()

# Defuser replacement classes for the two GLM-4.x MoE flavors. Keep this list
# explicit: a blanket scan of every Defuser module would hide unrelated
# compatibility problems behind one shim.
_GLM4_ROUTER_CLASSES = (
    "defuser.modeling.unfused_moe.glm4_moe.LinearGlm4MoeMoE",
    "defuser.modeling.unfused_moe.glm4_moe_lite.LinearGlm4MoeLiteMoE",
)

_PATCH_ATTR = "_gptqmodel_glm4_router_tuple_compat"
_WARNED = False


def _forwards_raw_router_output(cls) -> bool:
    """Return whether ``cls.forward`` still passes the raw router result."""

    try:
        source = inspect.getsource(cls.forward)
    except (OSError, TypeError):
        # A frozen/compiled build has no source. Treat it as unfixed: the
        # wrapper is a no-op for tensor routers, so it is safe either way.
        return True
    return "self.route_tokens_to_experts(router_logits)" in source


def _wrap_route_tokens_to_experts(cls) -> bool:
    original = getattr(cls, "route_tokens_to_experts", None)
    if original is None or getattr(original, _PATCH_ATTR, False):
        return False

    def route_tokens_to_experts(self, router_logits):
        # Transformers 5.22: (router_logits, topk_weights, topk_indices).
        # The replacement re-derives routing from the logits, so discarding
        # the other two entries preserves its routing math exactly.
        if isinstance(router_logits, tuple):
            router_logits = router_logits[0]
        return original(self, router_logits)

    setattr(route_tokens_to_experts, _PATCH_ATTR, True)
    cls.route_tokens_to_experts = route_tokens_to_experts
    return True


def apply_glm4_moe_router_compat() -> Tuple[str, ...]:
    """Install the tuple-unpacking shim on the loaded Defuser GLM-4.x blocks.

    Idempotent and process-global: Defuser patches the replacement class before
    instantiating the model, so every block instance shares the patched method.
    Returns the class names that were patched by this call.
    """

    global _WARNED

    patched = []
    for path in _GLM4_ROUTER_CLASSES:
        module_path, _, class_name = path.rpartition(".")
        try:
            module = __import__(module_path, fromlist=[class_name])
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is None or not _forwards_raw_router_output(cls):
            continue
        if _wrap_route_tokens_to_experts(cls):
            patched.append(path)

    if patched and not _WARNED:
        _WARNED = True
        log.warn(
            "Defuser GLM-4.x MoE blocks still pass the Transformers 5.22 router "
            "tuple into `route_tokens_to_experts`; applied a local compatibility "
            "shim. Upgrade Defuser for the upstream fix."
        )
    return tuple(patched)


__all__ = ["apply_glm4_moe_router_compat"]
