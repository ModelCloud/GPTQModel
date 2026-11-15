# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pickle state for kernels that store a `torch.compile` wrapper on the instance.

`optimize()` may replace an eager callable with a compiled wrapper. Those
wrappers capture the defining frame and cannot be pickled, so
`torch.save(model)` after `optimize()` raised `PicklingError`. One helper keeps
the rule identical for every kernel that opts in.
"""

from torch import nn


def compiled_module_getstate(module: nn.Module) -> dict:
    """Return a pickle state with compiled callables replaced by the eager path."""

    state = nn.Module.__getstate__(module)
    # FP8 and EXL3 Torch read this attribute directly, so keep it present.
    if "_compiled_forward" in state:
        state["_compiled_forward"] = None
    # Instance attributes that shadow the eager class methods (TorchLinear and
    # VPTQ). Removing them restores the class method on unpickle.
    state.pop("dequantize", None)
    state.pop("dequantize_weight", None)
    state["optimized"] = False
    return state
