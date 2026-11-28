# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Containment checks for paths taken from untrusted checkpoints.

Checkpoint indexes, safetensors headers, and model configuration files are
untrusted input. Tensor names and ``weight_map`` shard names may contain
absolute paths, ``..`` components, backslashes, or NUL bytes. Joining such a
name to an output directory lets a checkpoint write outside that directory.

The helpers below are the single validation point for those names. They reject
anything that is not a plain relative POSIX path, and they verify that the
resolved path stays under the directory the caller intends to use.
"""

from __future__ import annotations

import os
from typing import Optional


class CheckpointPathError(ValueError):
    """A checkpoint-derived path is absolute, malformed, or escapes its root."""


def validate_relative_checkpoint_path(name: object, *, kind: str = "checkpoint path") -> str:
    """Return ``name`` if it is a safe relative path, else raise.

    A safe path has no NUL byte, no backslash (Windows separators are not a
    portable checkpoint convention), is not absolute or drive-qualified, and
    has no empty, ``.`` or ``..`` component. Nested directories are allowed
    because the per-layer reshard layout stores shards in subdirectories.
    """

    if not isinstance(name, str) or not name:
        raise CheckpointPathError(f"Invalid {kind}: expected a non-empty string, got {name!r}")
    if "\x00" in name or "\\" in name:
        raise CheckpointPathError(f"Unsafe {kind}: NUL bytes and backslashes are not allowed: {name!r}")
    if os.path.isabs(name) or os.path.splitdrive(name)[0]:
        raise CheckpointPathError(f"Unsafe {kind}: absolute paths are not allowed: {name!r}")
    if any(part in ("", ".", "..") for part in name.split("/")):
        raise CheckpointPathError(f"Unsafe {kind}: empty, `.` or `..` path components: {name!r}")
    return name


def _is_under(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        # Different drives on Windows cannot share a common path.
        return False


def resolve_checkpoint_path(
    root: str,
    name: object,
    *,
    kind: str = "checkpoint path",
    follow_symlinks: bool = True,
) -> str:
    """Join a validated checkpoint-relative ``name`` to ``root``.

    ``follow_symlinks=False`` checks only the lexical path. Source shards in a
    Hugging Face cache snapshot are symlinks into the sibling blob store, so a
    read path must accept them; callers that write must keep the default and
    also verify the resolved target stays under ``root``.
    """

    relative = validate_relative_checkpoint_path(name, kind=kind)
    root_abs = os.path.abspath(root)
    path = os.path.normpath(os.path.join(root_abs, relative))
    if not _is_under(root_abs, path):
        raise CheckpointPathError(f"Unsafe {kind}: {name!r} escapes {root!r}")
    if follow_symlinks and not _is_under(os.path.realpath(root_abs), os.path.realpath(path)):
        raise CheckpointPathError(f"Unsafe {kind}: {name!r} resolves outside {root!r}")
    return path


def _hf_repo_root(source_root: str) -> Optional[str]:
    """Return the ``models--X--Y`` / ``datasets--X--Y`` dir of an HF snapshot."""
    current = os.path.abspath(source_root)
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            return None
        if os.path.basename(current) == "snapshots":
            return parent
        current = parent


def _hf_blob_root(source_root: str) -> Optional[str]:
    """Return the sibling ``blobs`` directory of an HF ``snapshots`` root."""

    repo = _hf_repo_root(source_root)
    return None if repo is None else os.path.join(repo, "blobs")


def _hf_shared_blob_root(source_root: str) -> Optional[str]:
    """Return the cache-wide shared blob store of an HF snapshot root.

    Recent ``huggingface_hub`` releases store Xet-backed files once per cache:
    ``<cache>/models--X--Y/blobs/<etag>`` is itself a symlink to
    ``<cache>/blobs/<2-hex>/<xet-hash>``. That store lives one directory above
    the per-repo ``blobs`` directory, so the per-repo allowance misses it
    (#6150).
    """

    repo = _hf_repo_root(source_root)
    return None if repo is None else os.path.join(os.path.dirname(repo), "blobs")


def _one_symlink_hop(path: str) -> Optional[str]:
    """Return the normalized target of one symlink hop, or ``None``."""

    if not os.path.islink(path):
        return None
    target = os.readlink(path)
    if not os.path.isabs(target):
        target = os.path.join(os.path.dirname(path), target)
    return os.path.normpath(target)


def _is_hf_shared_blob_chain(source_root: str, source_path: str) -> bool:
    """Accept exactly ``snapshot -> repo/blobs/<etag> -> cache/blobs/...``.

    The shared blob store is allowed only through the two-hop chain that
    ``huggingface_hub`` builds. A direct link from the snapshot into the shared
    store, a third hop, or a link that leaves the store is rejected.
    """

    blob_root = _hf_blob_root(source_root)
    shared_root = _hf_shared_blob_root(source_root)
    if blob_root is None or shared_root is None:
        return False
    first = _one_symlink_hop(source_path)
    if first is None or not _is_under(os.path.abspath(blob_root), first):
        return False
    second = _one_symlink_hop(first)
    if second is None or os.path.islink(second):
        return False
    return _is_under(os.path.abspath(shared_root), second)


def _assert_real_target_allowed(source_root: str, source_path: str, *, kind: str) -> None:
    real_target = os.path.realpath(source_path)
    if _is_under(os.path.realpath(source_root), real_target):
        return
    blob_root = _hf_blob_root(source_root)
    if blob_root is not None and _is_under(os.path.realpath(blob_root), real_target):
        return
    if _is_hf_shared_blob_chain(source_root, source_path):
        return
    raise CheckpointPathError(
        f"Unsafe {kind}: {source_path!r} resolves outside the checkpoint tree: {real_target!r}"
    )


def assert_source_read_allowed(source_root: str, source_path: str, *, kind: str = "source file") -> None:
    """Reject a source path whose real target is outside the source tree.

    Hugging Face cache snapshots point their files at the sibling ``blobs``
    store, so that one layout is allowed. Intermediate symlinks are checked
    too, not only the final component.
    """

    _assert_real_target_allowed(source_root, source_path, kind=kind)


def assert_auxiliary_copy_allowed(source_root: str, source_path: str, *, kind: str = "checkpoint file") -> None:
    """Reject a symlink that copies content from outside the checkpoint tree.

    ``shutil.copy2`` follows symlinks, so a repository can ship
    ``README.md -> ~/.ssh/id_rsa`` and get the target copied into an output
    directory that is then pushed to the Hub. Links that resolve inside the
    source tree are fine, and so are the symlinks a Hugging Face cache uses to
    point a snapshot file at its sibling blob store.
    """

    if not os.path.islink(source_path):
        return
    _assert_real_target_allowed(source_root, source_path, kind=kind)


__all__ = [
    "CheckpointPathError",
    "assert_auxiliary_copy_allowed",
    "assert_source_read_allowed",
    "resolve_checkpoint_path",
    "validate_relative_checkpoint_path",
]
