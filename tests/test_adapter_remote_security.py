# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.adapter.adapter import AdapterCache, Lora, normalize_adapter
from gptqmodel.adapter.remote import (
    configure_explicit_remote_adapter,
    resolve_checkpoint_adapter_path,
)
from gptqmodel.adapter import remote


def _write_adapter_files(adapter_dir):
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 2,
                "lora_alpha": 2,
                "target_modules": ["layer"],
                "rank_pattern": {"layer": 2},
            }
        ),
        encoding="utf-8",
    )
    weights_path = adapter_dir / "adapter_model.safetensors"
    save_file(
        {
            "base_model.model.layer.lora_A.weight": torch.tensor(
                [[1, 0], [1, 1]], dtype=torch.float16
            ),
            "base_model.model.layer.lora_B.weight": torch.tensor(
                [[1, 2], [0, 1]], dtype=torch.float16
            ),
        },
        str(weights_path),
    )
    return weights_path


def test_checkpoint_adapter_path_is_confined_to_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    adapter_dir = checkpoint / "local-lora-adapter"
    adapter_dir.mkdir(parents=True)

    assert resolve_checkpoint_adapter_path(
        "local-lora-adapter", str(checkpoint)
    ) == str(adapter_dir.resolve())
    with pytest.raises(ValueError, match="inside the checkpoint"):
        resolve_checkpoint_adapter_path("../outside", str(checkpoint))
    with pytest.raises(ValueError, match="existing local directory"):
        resolve_checkpoint_adapter_path("attacker/untrusted-adapter", str(checkpoint))

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="inside the checkpoint"):
        resolve_checkpoint_adapter_path(str(outside), str(checkpoint))

    # pathlib.Path adapter locations keep working.
    assert resolve_checkpoint_adapter_path(
        Path("local-lora-adapter"), str(checkpoint)
    ) == str(adapter_dir.resolve())


def test_explicit_remote_adapter_requires_opt_in_revision_and_digest():
    adapter = Lora(rank=2, path="attacker/untrusted-adapter")

    with pytest.raises(ValueError, match="disabled by default"):
        configure_explicit_remote_adapter(adapter, allow_remote=False)
    with pytest.raises(ValueError, match="40-character"):
        configure_explicit_remote_adapter(
            adapter, allow_remote=True, revision="main", sha256="0" * 64
        )
    with pytest.raises(ValueError, match="SHA-256"):
        configure_explicit_remote_adapter(
            adapter, allow_remote=True, revision="a" * 40, sha256="bad"
        )


def test_explicit_local_adapter_still_loads_without_remote_options(tmp_path):
    AdapterCache.reset()
    adapter_dir = tmp_path / "local"
    _write_adapter_files(adapter_dir)
    adapter = Lora(rank=2, path=adapter_dir)

    configure_explicit_remote_adapter(adapter, allow_remote=False)
    adapter.post_init(weight_key="layer", device=torch.device("cpu"))

    output = adapter.apply(
        torch.tensor([[1, 2]], dtype=torch.float16),
        torch.zeros((1, 2), dtype=torch.float16),
    )
    torch.testing.assert_close(output, torch.tensor([[7, 3]], dtype=torch.float16))


def test_remote_lora_is_pinned_hashed_and_applied_only_after_verification(
    tmp_path, monkeypatch
):
    AdapterCache.reset()
    adapter_dir = tmp_path / "hub"
    weights_path = _write_adapter_files(adapter_dir)
    weights_digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    calls = []

    def fake_hf_hub_download(*, repo_id, revision, filename, subfolder=None):
        calls.append((repo_id, revision, filename, subfolder))
        return str(adapter_dir / filename)

    monkeypatch.setattr(remote, "hf_hub_download", fake_hf_hub_download)
    adapter = Lora(rank=2, path="trusted/adapter")
    configure_explicit_remote_adapter(
        adapter,
        allow_remote=True,
        revision="a" * 40,
        sha256=weights_digest,
    )
    adapter.post_init(weight_key="layer", device=torch.device("cpu"))

    assert calls == [
        ("trusted/adapter", "a" * 40, "adapter_config.json", None),
        ("trusted/adapter", "a" * 40, "adapter_model.safetensors", None),
    ]
    output = adapter.apply(
        torch.tensor([[1, 2]], dtype=torch.float16),
        torch.zeros((1, 2), dtype=torch.float16),
    )
    torch.testing.assert_close(output, torch.tensor([[7, 3]], dtype=torch.float16))


def test_checkpoint_metadata_cannot_download_adapter(tmp_path, monkeypatch):
    AdapterCache.reset()
    calls = []
    monkeypatch.setattr(
        remote, "hf_hub_download", lambda **kwargs: calls.append(kwargs)
    )
    adapter = Lora(rank=2, path="attacker/untrusted-adapter")

    with pytest.raises(ValueError, match="disabled"):
        adapter.post_init(weight_key="layer", device=torch.device("cpu"))

    assert calls == []


def test_remote_adapter_requires_pin_and_digest_pair(monkeypatch):
    AdapterCache.reset()
    monkeypatch.setattr(
        remote,
        "hf_hub_download",
        lambda **kwargs: pytest.fail("unpinned remote adapter fetch attempted"),
    )
    for revision, sha256 in (("a" * 40, None), (None, "0" * 64)):
        adapter = Lora(rank=2, path="trusted/adapter")
        adapter._gptqmodel_remote_revision = revision
        adapter._gptqmodel_remote_sha256 = sha256
        with pytest.raises(ValueError, match="revision"):
            adapter.post_init(weight_key="layer", device=torch.device("cpu"))


def test_remote_repo_subfolder_rejects_traversal(tmp_path):
    AdapterCache.reset()
    adapter = Lora(rank=2, path="trusted/adapter/../../evil")
    configure_explicit_remote_adapter(
        adapter,
        allow_remote=True,
        revision="a" * 40,
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="invalid path segments"):
        adapter.post_init(weight_key="layer", device=torch.device("cpu"))


def test_remote_adapter_url_revision_must_match_pin(monkeypatch):
    AdapterCache.reset()
    adapter = Lora(
        rank=2,
        path="https://huggingface.co/trusted/adapter/blob/"
        + "b" * 40
        + "/adapter_model.safetensors",
    )
    configure_explicit_remote_adapter(
        adapter,
        allow_remote=True,
        revision="a" * 40,
        sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="must match the explicitly pinned"):
        adapter.post_init(weight_key="layer", device=torch.device("cpu"))


def test_serialized_adapter_cannot_inject_non_tensor_weights():
    with pytest.raises(ValueError, match="Invalid adapter config"):
        normalize_adapter(
            adapter={"name": "lora", "rank": 2, "lora_A": {"n": 1}}
        )
    with pytest.raises(ValueError, match="Invalid adapter config"):
        normalize_adapter(
            adapter={"name": "lora", "rank": 2, "lora_B": [[1, 0], [0, 1]]}
        )

    adapter = normalize_adapter(
        adapter={
            "name": "lora",
            "rank": 2,
            "lora_A": torch.zeros(1),
            "lora_B": torch.zeros(1),
        }
    )
    assert adapter.lora_A is not None


def test_remote_weight_digest_mismatch_fails_before_safetensors_load(
    tmp_path, monkeypatch
):
    AdapterCache.reset()
    adapter_dir = tmp_path / "hub"
    _write_adapter_files(adapter_dir)
    calls = []

    def fake_hf_hub_download(*, repo_id, revision, filename, subfolder=None):
        calls.append(filename)
        return str(adapter_dir / filename)

    monkeypatch.setattr(remote, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        "gptqmodel.adapter.adapter.safetensors.torch.load_file",
        lambda *_args, **_kwargs: pytest.fail("unverified adapter weights were loaded"),
    )
    adapter = Lora(rank=2, path="trusted/adapter")
    configure_explicit_remote_adapter(
        adapter,
        allow_remote=True,
        revision="a" * 40,
        sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        adapter.post_init(weight_key="layer", device=torch.device("cpu"))

    assert calls == ["adapter_config.json", "adapter_model.safetensors"]
