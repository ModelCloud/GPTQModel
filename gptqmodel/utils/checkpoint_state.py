# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Track whether ``state_dict()`` is being collected for a checkpoint write.

Some kernels (Marlin, ExLlamaV2) rewrite the packed tensors into a private
runtime layout that has no inverse. A read-only ``state_dict()`` caller such as
PEFT still needs the tensors, but writing them to a checkpoint would publish a
file that no loader can consume. The context lets the serialization paths
declare their intent, so ``_save_to_state_dict`` can fail closed only there.

``ContextVar`` keeps the flag per thread and per async task and is re-entrant,
which matters because a save may call ``state_dict()`` from several modules.
"""

from contextlib import contextmanager
from contextvars import ContextVar


_SERIALIZING: ContextVar[bool] = ContextVar(
    "gptqmodel_checkpoint_serialization", default=False
)


@contextmanager
def checkpoint_serialization():
    """Mark the enclosed work as writing a checkpoint."""

    token = _SERIALIZING.set(True)
    try:
        yield
    finally:
        _SERIALIZING.reset(token)


def is_checkpoint_serialization() -> bool:
    """Whether the caller is collecting tensors for a checkpoint write."""

    return bool(_SERIALIZING.get())


__all__ = ["checkpoint_serialization", "is_checkpoint_serialization"]
