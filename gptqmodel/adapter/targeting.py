# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""PEFT-compatible resolution of which modules an adapter targets.

Which modules receive a LoRA adapter is decided by the adapter's own
``target_modules`` / ``exclude_modules``, not by which modules the checkpoint
happens to quantize. Before this module existed one global adapter was passed to
every quantized module, so a standard partial PEFT adapter (``q_proj`` /
``v_proj`` only) asserted while loading on the modules it does not cover
(issue #4210).

The matching rules mirror PEFT's ``check_target_module_exists``:

* a string ``target_modules`` is a regular expression matched with
  ``re.fullmatch``;
* a list/set entry is an exact module name or a dotted-suffix match
  (``q_proj`` matches ``model.layers.0.self_attn.q_proj`` but not
  ``...my_q_proj``);
* a list/set entry prefixed with ``re:`` is a regular expression matched with
  ``re.fullmatch``;
* ``all-linear`` selects every linear module (the caller only offers linear
  modules);
* ``exclude_modules`` uses the same matching and wins over a target match.
"""

import re
from typing import Iterable, Optional, Union


RE_PREFIX = "re:"
ALL_LINEAR = "all-linear"


def _matches_pattern(module_name: str, pattern: str) -> bool:
    """Match one list entry using PEFT's exact/suffix/``re:`` rules."""

    if pattern.startswith(RE_PREFIX):
        return re.fullmatch(pattern[len(RE_PREFIX):], module_name) is not None
    return module_name == pattern or module_name.endswith(f".{pattern}")


def _matches_any(module_name: str, patterns: Union[str, Iterable[str], None]) -> bool:
    """Match a PEFT ``target_modules``/``exclude_modules`` value."""

    if patterns is None:
        return False
    if isinstance(patterns, str):
        return re.fullmatch(patterns, module_name) is not None
    if isinstance(patterns, (list, tuple, set, frozenset)):
        return any(
            isinstance(pattern, str) and _matches_pattern(module_name, pattern)
            for pattern in patterns
        )
    return False


def module_matches_targets(
    module_name: str,
    target_modules: Optional[Union[str, Iterable[str]]] = None,
    exclude_modules: Optional[Union[str, Iterable[str]]] = None,
) -> bool:
    """Return whether a PEFT adapter applies to ``module_name``.

    ``target_modules=None`` keeps the historical attach-to-everything behavior:
    an adapter without a config (merged EoRA weights or a not-yet-saved output
    directory) has no target list to honour.
    """

    if target_modules is None:
        matched = True
    elif isinstance(target_modules, str) and target_modules == ALL_LINEAR:
        matched = True
    else:
        matched = _matches_any(module_name, target_modules)

    if not matched:
        return False

    if exclude_modules and _matches_any(module_name, exclude_modules):
        return False
    return True
