# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from collections import defaultdict
from typing import Iterable, List, Tuple

from .torch import TorchQuantLinear


def configure_lookahead_chain(modules: Iterable[TorchQuantLinear]):
    """Wire a sequence of TorchQuantLinear modules for one-step lookahead.

    Each module in *modules* (except the last) will prefetch the next module's
    dequantized weights the moment it finishes its own forward call. The last
    module's ``lookahead_next`` pointer is cleared.
    """

    last = None
    for module in modules:
        if last is not None:
            last.enable_lookahead(True).set_lookahead_next(module)
        module.enable_lookahead(True)
        last = module
    if last is not None:
        last.set_lookahead_next(None)


def _clear_existing_links(modules: Iterable[TorchQuantLinear]):
    for module in modules:
        module.set_lookahead_next(None)


def configure_default_lookahead(model) -> None:
    """Eagerly decode the MLP projection trio when attention ``q_proj`` runs.

    For each transformer block this disables lookahead between
    ``self_attn.{q,k,v,o}_proj`` and instead wires ``q_proj`` to prefetch the
    block's ``mlp.{gate,up,down}_proj`` modules concurrently. Missing modules
    are skipped.
    """

    ordered_modules: List[Tuple[str, TorchQuantLinear]] = []
    for name, module in model.named_modules():
        if isinstance(module, TorchQuantLinear):
            ordered_modules.append((name, module))

    if not ordered_modules:
        return

    _clear_existing_links(module for _, module in ordered_modules)

    attn_order = ("q_proj", "k_proj", "v_proj", "o_proj")
    mlp_order = ("gate_proj", "up_proj", "down_proj")

    attn_blocks = defaultdict(dict)
    mlp_blocks = defaultdict(dict)

    for name, module in ordered_modules:
        if ".self_attn." in name:
            prefix, leaf = name.split(".self_attn.", maxsplit=1)
            leaf = leaf.split(".")[0]
            attn_blocks[prefix][leaf] = module
            continue
        if ".mlp." in name:
            prefix, leaf = name.split(".mlp.", maxsplit=1)
            leaf = leaf.split(".")[0]
            mlp_blocks[prefix][leaf] = module

    for block in set(list(attn_blocks.keys()) + list(mlp_blocks.keys())):
        attn = attn_blocks.get(block, {})
        mlp = mlp_blocks.get(block, {})

        q_module = attn.get("q_proj")
        attn_modules = [attn.get(key) for key in attn_order if attn.get(key) is not None]
        mlp_targets = [mlp.get(key) for key in mlp_order if mlp.get(key) is not None]

        # reset lookahead state on all participating modules within this block
        for module in attn_modules:
            module.set_lookahead_next(None)
            module.enable_lookahead(False)
        for module in mlp_targets:
            module.set_lookahead_next(None)
            module.enable_lookahead(False)

        if q_module is not None and mlp_targets:
            q_module.enable_lookahead(True).set_lookahead_next(tuple(mlp_targets))
            for target in mlp_targets:
                target.enable_lookahead(True)
