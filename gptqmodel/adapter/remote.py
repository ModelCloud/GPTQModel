# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import hashlib
import ntpath
import os
import posixpath
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import unquote, urlparse

from ..utils.hub import hf_hub_download
from ..utils.logger import setup_logger


log = setup_logger()

if TYPE_CHECKING:
    from .adapter import Adapter

_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def _as_path_str(path) -> Optional[str]:
    """Coerce os.PathLike adapter locations to str so pathlib inputs keep working."""

    if isinstance(path, os.PathLike):
        return os.fspath(path)
    return path


def _check_remote_part(part: str, what: str) -> None:
    """Reject traversal or host-path fragments inside remote repo/file names."""

    if not isinstance(part, str):
        raise ValueError(f"Adapter: remote {what} must be a string.")
    segments = part.split("/")
    if (
        any(segment in ("", ".", "..") for segment in segments)
        or "\\" in part
        or "\0" in part
        or ntpath.splitdrive(part)[0]
    ):
        raise ValueError(f"Adapter: remote {what} contains invalid path segments: `{part}`")


def parse_url(url: str) -> list[str]:
    """Extract a pinned Hugging Face repo, revision, and filename from a blob URL."""

    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in {
        "huggingface.co",
        "hf.co",
    }:
        return []

    parts = unquote(parsed_url.path).lstrip("/").split("/")
    idx = next((index for index in (2, 1) if len(parts) > index and parts[index] == "blob"), None)
    if idx is None or len(parts) <= idx + 2:
        return []

    repo_id = "/".join(parts[:idx])
    revision = parts[idx + 1]
    filename = "/".join(parts[idx + 2 :])
    return [repo_id, revision, filename]


def _is_checkpoint_cache_blob(path: Path, checkpoint: Path) -> bool:
    """Allow the same repository's standard HF snapshot-to-blob file links."""

    if checkpoint.parent.name != "snapshots" or not _REVISION_RE.fullmatch(checkpoint.name):
        return False
    repo_cache = checkpoint.parent.parent
    if not repo_cache.name.startswith("models--"):
        return False
    blobs = repo_cache / "blobs"
    target = Path(os.path.abspath(path.parent / os.readlink(path)))
    if target.parent != blobs or not (
        _REVISION_RE.fullmatch(target.name) or _SHA256_RE.fullmatch(target.name)
    ):
        return False
    return not blobs.is_symlink() and not target.is_symlink() and target.is_file()


def resolve_checkpoint_adapter_path(
    path: Optional[str], checkpoint_path: str
) -> Optional[str]:
    """Resolve manifest adapter paths inside the checkpoint directory only."""

    if path in (None, ""):
        return path
    path = _as_path_str(path)
    if not isinstance(path, str):
        raise ValueError(
            "Adapter: checkpoint adapter path must be a local directory string."
        )

    checkpoint = Path(checkpoint_path).resolve(strict=True)
    candidate = Path(path)
    if candidate.is_absolute() or ntpath.isabs(path) or ntpath.splitdrive(path)[0]:
        raise ValueError(
            "Adapter: adapter paths from quantize_config.json must be relative paths "
            "inside the checkpoint; use the explicit adapter= argument for external adapters."
        )
    if "\\" in path or ".." in candidate.parts:
        raise ValueError("Adapter: manifest adapter paths must stay inside the checkpoint directory.")
    current = checkpoint
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Adapter: manifest adapter directories cannot contain symlinks.")
    candidate = checkpoint / candidate

    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(
            "Adapter: adapter paths from quantize_config.json must name an existing local directory "
            "inside the checkpoint; use the explicit adapter= argument for external adapters."
        ) from exc

    if not resolved.is_dir():
        raise ValueError(
            "Adapter: adapter paths from quantize_config.json must name a local directory."
        )

    try:
        resolved.relative_to(checkpoint)
    except ValueError as exc:
        raise ValueError(
            "Adapter: adapter paths from quantize_config.json must stay inside the checkpoint directory."
        ) from exc

    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        adapter_file = resolved / filename
        if adapter_file.is_symlink() and not _is_checkpoint_cache_blob(adapter_file, checkpoint):
            raise ValueError("Adapter: manifest adapter files cannot link outside the checkpoint storage.")
    return str(resolved)


def configure_explicit_remote_adapter(
    adapter: "Adapter",
    *,
    allow_remote: bool,
    revision: Optional[str] = None,
    sha256: Optional[str] = None,
) -> None:
    """Attach caller-supplied remote pinning to an explicitly passed adapter."""

    adapter._gptqmodel_remote_revision = None
    adapter._gptqmodel_remote_sha256 = None
    adapter._gptqmodel_checkpoint_source = None
    path = _as_path_str(getattr(adapter, "path", None))
    if path in (None, ""):
        if allow_remote or revision is not None or sha256 is not None:
            raise ValueError(
                "Adapter: remote adapter options require an explicit remote adapter path."
            )
        return
    if not isinstance(path, str):
        raise ValueError("Adapter: `path` must be a string.")
    adapter.path = path

    if os.path.isdir(path):
        if allow_remote or revision is not None or sha256 is not None:
            raise ValueError(
                "Adapter: remote adapter options cannot be used with a local adapter directory."
            )
        adapter.path = os.path.abspath(path)
        return

    if (
        os.path.isabs(path)
        or ntpath.isabs(path)
        or path.startswith(("./", "../", ".\\", "..\\"))
        or "\\" in path
    ):
        raise ValueError(f"Adapter: local adapter directory does not exist: `{path}`")

    if not allow_remote:
        raise ValueError(
            "Adapter: remote adapters are disabled by default. Pass allow_remote_adapter=True with "
            "adapter_revision and adapter_sha256 to load an explicitly selected Hub adapter."
        )
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError(
            "Adapter: remote adapters require an immutable 40-character Hub commit revision."
        )
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ValueError(
            "Adapter: remote adapters require the adapter_model.safetensors SHA-256 digest."
        )

    adapter._gptqmodel_remote_revision = revision.lower()
    adapter._gptqmodel_remote_sha256 = sha256.lower()


def _validate_remote_request(
    path: str,
    *,
    allow_remote: bool,
    revision: Optional[str],
) -> None:
    if not allow_remote:
        raise ValueError(
            f"Adapter: remote loading for `{path}` is disabled; provide an explicit adapter and opt in."
        )
    if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
        raise ValueError(
            "Adapter: remote files require an immutable 40-character Hub commit revision."
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_sha256(path: str, expected_sha256: Optional[str]) -> None:
    if expected_sha256 is None:
        return
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ValueError(
            "Adapter: expected SHA-256 digest must contain exactly 64 hexadecimal characters."
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Adapter: SHA-256 verification failed for `{os.path.basename(path)}`; "
            f"expected `{expected_sha256.lower()}`, got `{actual_sha256}`."
        )


def resolve_path(
    path: str,
    filename: str,
    *,
    allow_remote: bool = False,
    revision: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> str:
    """Resolve an adapter file locally or from an explicitly pinned Hub revision."""

    path = _as_path_str(path)
    if not isinstance(path, str) or not path:
        raise ValueError(
            "Adapter: `path` must be a local directory or a Hub repository id."
        )
    _check_remote_part(filename, "file name")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256)
    ):
        raise ValueError("Adapter: expected SHA-256 digest must contain exactly 64 hexadecimal characters.")

    if os.path.isdir(path):
        if allow_remote or revision is not None:
            raise ValueError("Adapter: remote adapter options cannot be used with a local adapter directory.")
        resolved_path = os.path.join(path, filename)
        log.info(f"Resolver: Local path: `{resolved_path}`")
        if not os.path.isfile(resolved_path):
            raise ValueError(f"Resolver: Cannot find file in path: `{resolved_path}`")
        _verify_sha256(resolved_path, expected_sha256)
        return resolved_path

    if (
        os.path.isabs(path)
        or ntpath.isabs(path)
        or path.startswith(("./", "../", ".\\", "..\\"))
        or "\\" in path
    ):
        raise ValueError(f"Resolver: Local adapter directory does not exist: `{path}`")

    if path.startswith("http://") or path.startswith("https://"):
        result = parse_url(path)
        if len(result) != 3:
            raise ValueError(
                "Resolver: Only HTTPS Hugging Face blob URLs with a pinned revision are supported."
            )
        repo_id, url_revision, remote_filename = result
        _validate_remote_request(path, allow_remote=allow_remote, revision=revision)
        if url_revision.lower() != revision.lower():
            raise ValueError(
                "Resolver: adapter URL revision must match the explicitly pinned commit revision."
            )
        _check_remote_part(repo_id, "repository id")
        _check_remote_part(remote_filename, "file name")
        filename = posixpath.join(posixpath.dirname(remote_filename), filename)
    else:
        _validate_remote_request(path, allow_remote=allow_remote, revision=revision)
        repo_id = path.rstrip("/")
        subfolder = None

        # Support repo ids with a nested adapter folder.
        if repo_id.count("/") > 1:
            path_split = repo_id.split("/")
            repo_id = f"{path_split[0]}/{path_split[1]}"
            subfolder = "/".join(path_split[2:])
            _check_remote_part(subfolder, "subfolder")
        _check_remote_part(repo_id, "repository id")

        log.warn(
            f"SECURITY: downloading explicitly enabled LoRA adapter from `{repo_id}` "
            f"at immutable revision `{revision.lower()}`."
        )
        resolved_path = hf_hub_download(
            repo_id=repo_id,
            revision=revision.lower(),
            filename=filename,
            subfolder=subfolder,
        )
        _verify_sha256(resolved_path, expected_sha256)
        return resolved_path

    log.warn(
        f"SECURITY: downloading explicitly enabled LoRA adapter from `{repo_id}` "
        f"at immutable revision `{revision.lower()}`."
    )
    resolved_path = hf_hub_download(
        repo_id=repo_id, revision=revision.lower(), filename=filename
    )
    _verify_sha256(resolved_path, expected_sha256)
    return resolved_path


def validate_remote_pin(revision: Optional[str], sha256: Optional[str]) -> None:
    """Require remote adapter pinning to be all-or-nothing.

    Remote downloads are only safe when both the immutable Hub commit and the
    adapter weights digest are recorded, so refuse half-configured adapters.
    """

    if revision is None and sha256 is None:
        return
    if revision is None or sha256 is None:
        raise ValueError(
            "Adapter: remote adapters require both an immutable Hub commit revision "
            "and the adapter_model.safetensors SHA-256 digest."
        )
    _validate_remote_request("adapter", allow_remote=True, revision=revision)
    if not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256):
        raise ValueError("Adapter: expected SHA-256 digest must contain exactly 64 hexadecimal characters.")
