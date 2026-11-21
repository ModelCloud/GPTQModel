# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Offline port inventories and pinned upstream assets for release fidelity tests."""

import fnmatch
import hashlib
import json
from functools import cache
from itertools import product
from pathlib import Path

import pytest
import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.constants import HF_HUB_OFFLINE
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, OfflineModeIsEnabled


PORTS = Path(__file__).resolve().parents[1] / "gptqmodel" / "models" / "ports"
DATA = Path(__file__).resolve().parent / "fixtures" / "ports"


@cache
def _release_manifest():
    return json.loads((DATA / "upstream.json").read_text(encoding="utf-8"))


def _range_values(expression):
    for part in expression.split(","):
        start, _, end = part.partition("-")
        yield from range(int(start), int(end or start) + 1)


def _unpack_inventory(value):
    if isinstance(value, list):
        return [_unpack_inventory(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "tensor_key_ranges" in value:
        result = {}
        for pattern, axes, entry in value["tensor_key_ranges"]:
            for coords in product(*map(_range_values, axes)):
                key = pattern.format(*coords)
                if key in result:
                    raise ValueError(f"Duplicate tensor key {key}")
                result[key] = entry
        return sorted(result) if value.get("keys_only") else result
    return {key: _unpack_inventory(item) for key, item in value.items()}


def _asset_key(path):
    return Path(path).resolve().relative_to(PORTS).as_posix()


def has_port_asset(path):
    key = _asset_key(path)
    return (DATA / key).is_file() or key in _release_manifest()


def iter_port_assets(directory, pattern):
    key = _asset_key(directory)
    local = (DATA / key).glob(pattern)
    names = {path.name for path in local}
    names.update(
        path.name
        for path in map(Path, _release_manifest())
        if path.parent.as_posix() == key and fnmatch.fnmatchcase(path.name, pattern)
    )
    return (directory / name for name in sorted(names))


def _fetch_range(url, start, length):
    if HF_HUB_OFFLINE:
        pytest.skip(f"Pinned release asset requires network access: {url}")
    end = start + length - 1
    with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=(5, 15), stream=True) as response:
        response.raise_for_status()
        if response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise ValueError(f"Upstream did not serve the requested byte range: {url}")
        # Read only the requested header and one byte to detect unexpected excess.
        return response.raw.read(length + 1)


def _download(entry):
    if "url" in entry:
        if HF_HUB_OFFLINE:
            pytest.skip(f"Pinned release asset requires network access: {entry['url']}")
        with requests.get(entry["url"], timeout=(5, 15), stream=True) as response:
            response.raise_for_status()
            return response.raw.read(entry["bytes"] + 1)
    if "header_bytes" in entry:
        url = hf_hub_url(entry["repo"], entry["file"], revision=entry["revision"])
        return _fetch_range(url, 8, entry["header_bytes"])
    path = hf_hub_download(entry["repo"], entry["file"], revision=entry["revision"], etag_timeout=5)
    return Path(path).read_bytes()


@cache
def _verified_release_bytes(key):
    entry = _release_manifest()[key]
    try:
        data = _download(entry)
    except (LocalEntryNotFoundError, OfflineModeIsEnabled):
        pytest.skip(f"Pinned release asset is unavailable from the cache or Hub: {key}")
    except (HfHubHTTPError, requests.RequestException) as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (400, 404, 416):
            raise
        pytest.skip(f"Pinned release asset {key} is unavailable ({status or type(exc).__name__})")
    if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Pinned release asset {key} does not match its size and SHA-256")
    return data


def read_port_asset(path):
    """Read an exact upstream file or the corresponding offline test fixture."""
    key = _asset_key(path)
    local = DATA / key
    if local.is_file():
        return local.read_bytes()
    return _verified_release_bytes(key)


def read_port_json(path):
    """Expand grouped tensor key ranges from an offline inventory."""
    return _unpack_inventory(json.loads(read_port_asset(path)))
