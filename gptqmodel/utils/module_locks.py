# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import contextlib
from threading import Lock, RLock
from typing import Dict, Iterator, Optional


__all__ = [
    "ROOT_PARENT_KEY",
    "get_parent_lock",
    "parent_lock_key",
    "parent_module_lock",
]


ROOT_PARENT_KEY = "<root>"

_PARENT_LOCKS: Dict[str, RLock] = {}
_PARENT_LOCKS_GUARD = Lock()


def parent_lock_key(module_name: Optional[str]) -> str:
    if not module_name:
        return ROOT_PARENT_KEY
    parts = module_name.split(".")
    if len(parts) <= 1:
        return parts[0]
    return ".".join(parts[:-1])


def get_parent_lock(module_name: Optional[str]) -> RLock:
    key = parent_lock_key(module_name)
    with _PARENT_LOCKS_GUARD:
        lock = _PARENT_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _PARENT_LOCKS[key] = lock
    return lock


@contextlib.contextmanager
def parent_module_lock(module_name: Optional[str]) -> Iterator[None]:
    lock = get_parent_lock(module_name)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
