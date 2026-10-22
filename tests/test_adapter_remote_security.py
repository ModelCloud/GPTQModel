# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel import QuantizeConfig
from gptqmodel.adapter import remote
from gptqmodel.adapter.adapter import AdapterCache, Lora, normalize_adapter
from gptqmodel.adapter.remote import (
    configure_explicit_remote_adapter,
    resolve_checkpoint_adapter_path,
)


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


@pytest.mark.parametrize(
    "path",
    [
        "/outside/adapter",
        "C:\\outside\\adapter",
        "C:adapter",
        "\\\\server\\share\\adapter",
    ],
)
def test_checkpoint_rejects_absolute_manifest_paths_before_resolving(
    path, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    original_resolve = Path.resolve

    def resolve_checkpoint_only(candidate, *args, **kwargs):
        if candidate != checkpoint:
            pytest.fail(f"manifest path was resolved before rejection: {candidate}")
        return original_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_checkpoint_only)

    with pytest.raises(ValueError, match="relative paths inside the checkpoint"):
        resolve_checkpoint_adapter_path(path, str(checkpoint))


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


@pytest.mark.parametrize("filename", ["adapter_config.json", "adapter_model.safetensors"])
@pytest.mark.parametrize("repo_id", ["trusted/adapter", "gpt2", "trusted/blob", "blob"])
def test_pinned_blob_url_loads_config_and_weights_from_same_folder(
    tmp_path, monkeypatch, filename, repo_id
):
    AdapterCache.reset()
    adapter_dir = tmp_path / "hub"
    weights_path = _write_adapter_files(adapter_dir)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(adapter_dir / Path(kwargs["filename"]).name)

    monkeypatch.setattr(remote, "hf_hub_download", download)
    adapter = Lora(rank=2, path=f"https://huggingface.co/{repo_id}/blob/{'a' * 40}/nested/{filename}")
    configure_explicit_remote_adapter(
        adapter, allow_remote=True, revision="a" * 40,
        sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
    )
    adapter.post_init(weight_key="layer", device=torch.device("cpu"))

    assert calls == [
        {"repo_id": repo_id, "revision": "a" * 40, "filename": "nested/adapter_config.json"},
        {"repo_id": repo_id, "revision": "a" * 40, "filename": "nested/adapter_model.safetensors"},
    ]
    assert adapter.lora_A.shape == (2, 2)


@pytest.mark.parametrize(
    ("revision", "sha256"),
    [("main", "0" * 64), ("a" * 40, ""), ("a" * 40, "bad"), ("a" * 40, 123)],
)
def test_invalid_pin_pair_is_rejected_before_config_io(monkeypatch, revision, sha256):
    monkeypatch.setattr(
        "gptqmodel.adapter.adapter.LoraConfig.from_pretrained",
        lambda **_kwargs: pytest.fail("invalid pins reached adapter config I/O"),
    )
    adapter = Lora(rank=2, path="trusted/adapter")
    adapter._gptqmodel_remote_revision = revision
    adapter._gptqmodel_remote_sha256 = sha256
    with pytest.raises(ValueError):
        adapter.post_init(weight_key="layer", device=torch.device("cpu"))


def test_local_file_resolution_verifies_supplied_digest(tmp_path):
    weights = _write_adapter_files(tmp_path / "local")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert remote.resolve_path(weights.parent, weights.name, expected_sha256=digest) == str(weights)
    with pytest.raises(ValueError, match="SHA-256 verification failed"):
        remote.resolve_path(weights.parent, weights.name, expected_sha256="0" * 64)


def test_remote_resolution_does_not_fall_back_to_local_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "trusted" / "adapter").mkdir(parents=True)
    with pytest.raises(ValueError, match="local adapter directory"):
        remote.resolve_path(
            "trusted/adapter", "adapter_model.safetensors",
            allow_remote=True, revision="a" * 40, expected_sha256="0" * 64,
        )


def test_reconfiguring_adapter_resets_remote_authorization(tmp_path):
    adapter = Lora(rank=2, path="trusted/adapter")
    configure_explicit_remote_adapter(
        adapter, allow_remote=True, revision="a" * 40, sha256="0" * 64,
    )
    adapter.path = tmp_path
    configure_explicit_remote_adapter(adapter, allow_remote=False)
    assert adapter._gptqmodel_remote_revision is None
    assert adapter._gptqmodel_remote_sha256 is None
    assert adapter.path == str(tmp_path)


def test_explicit_relative_local_path_is_stable_across_working_directories(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    monkeypatch.chdir(tmp_path)
    adapter = Lora(rank=2, path="local")
    configure_explicit_remote_adapter(adapter, allow_remote=False)
    assert adapter.path == str(local)


@pytest.mark.parametrize("key", ["lora_A", "lora_B"])
def test_serialized_adapter_allows_optional_null_tensors(key):
    adapter = normalize_adapter({"name": "lora", "rank": 2, key: None})
    assert getattr(adapter, key) is None


@pytest.mark.parametrize("filename", ["quantize_config.json", "config.json"])
def test_checkpoint_adapter_provenance_survives_direct_config_loading(tmp_path, monkeypatch, filename):
    checkpoint = tmp_path / "checkpoint"
    _write_adapter_files(checkpoint / "adapter")
    payload = {"bits": 4, "adapter": {"name": "lora", "rank": 2, "path": "adapter"}}
    (checkpoint / filename).write_text(json.dumps(
        {"quantization_config": payload} if filename == "config.json" else payload
    ))
    monkeypatch.chdir(tmp_path)
    qcfg = QuantizeConfig.from_pretrained(checkpoint)
    assert qcfg.adapter._gptqmodel_checkpoint_source == (str(checkpoint), "adapter")
    qcfg.adapter.post_init(weight_key="layer", device=torch.device("cpu"))
    assert qcfg.adapter.path == str(checkpoint / "adapter")
    assert qcfg.adapter.to_dict()["path"] == "adapter"
    payload["adapter"] = qcfg.adapter.to_dict()
    (checkpoint / filename).write_text(json.dumps(
        {"quantization_config": payload} if filename == "config.json" else payload
    ))
    reloaded = QuantizeConfig.from_pretrained(checkpoint)
    reloaded.adapter.post_init(weight_key="layer", device=torch.device("cpu"))
    assert reloaded.adapter.path == qcfg.adapter.path


def test_checkpoint_adapter_files_reject_external_symlinks(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    adapter_dir = checkpoint / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").symlink_to(tmp_path / "not-read.json")
    monkeypatch.setattr(
        "gptqmodel.adapter.adapter.LoraConfig.from_pretrained",
        lambda **_kwargs: pytest.fail("manifest file confinement must precede config I/O"),
    )
    (checkpoint / "quantize_config.json").write_text(json.dumps({
        "bits": 4, "adapter": {"name": "lora", "rank": 2, "path": "adapter"},
    }))
    adapter = QuantizeConfig.from_pretrained(checkpoint).adapter
    with pytest.raises(ValueError, match="cannot link outside"):
        adapter.post_init(weight_key="layer", device=torch.device("cpu"))


def test_checkpoint_directory_symlinks_are_rejected_before_resolution(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter").symlink_to(tmp_path / "not-probed", target_is_directory=True)
    original_resolve = Path.resolve

    def resolve_checkpoint_only(candidate, *args, **kwargs):
        assert candidate == checkpoint
        return original_resolve(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve_checkpoint_only)
    with pytest.raises(ValueError, match="directories cannot contain symlinks"):
        resolve_checkpoint_adapter_path("adapter", str(checkpoint))


def test_checkpoint_adapter_preserves_standard_hf_cache_links(tmp_path):
    repo_cache = tmp_path / "models--trusted--model"
    checkpoint = repo_cache / "snapshots" / ("a" * 40)
    adapter_dir = checkpoint / "adapter"
    _write_adapter_files(adapter_dir)
    blobs = repo_cache / "blobs"
    blobs.mkdir()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter_dir / filename
        if filename.endswith(".json"):
            content = path.read_bytes()
            digest = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.rename(blobs / digest)
        path.symlink_to(f"../../../blobs/{digest}")
    assert resolve_checkpoint_adapter_path("adapter", str(checkpoint)) == str(adapter_dir)
    adapter = Lora(rank=2, path=str(adapter_dir))
    adapter._gptqmodel_checkpoint_source = (str(checkpoint), "adapter")
    adapter.post_init(weight_key="layer", device=torch.device("cpu"))
    assert adapter.lora_A.shape == (2, 2)


@pytest.mark.parametrize("filename", ["../weights", "/weights", "dir\\weights", "C:weights", "\0"])
def test_resolver_rejects_non_relative_filenames_before_io(filename, monkeypatch):
    monkeypatch.setattr(remote, "hf_hub_download", lambda **_kwargs: pytest.fail("unexpected I/O"))
    with pytest.raises(ValueError, match="invalid path segments"):
        remote.resolve_path("trusted/adapter", filename, allow_remote=True, revision="a" * 40)


def test_cached_adapter_rank_is_validated():
    from gptqmodel.adapter.peft import LoraConfig

    AdapterCache.reset()
    key = ("trusted/adapter", None, None)
    AdapterCache.add(key, LoraConfig(r=2), {})
    try:
        with pytest.raises(ValueError, match="rank.*must match"):
            Lora(rank=4, path="trusted/adapter").post_init("layer", torch.device("cpu"))
    finally:
        AdapterCache.reset()


def test_direct_local_adapter_cache_key_is_absolute(tmp_path, monkeypatch):
    adapter_dir = tmp_path / "adapter"
    _write_adapter_files(adapter_dir)
    monkeypatch.chdir(tmp_path)
    adapter = Lora(rank=2, path="adapter")
    adapter.post_init("layer", torch.device("cpu"))
    assert adapter.path == str(adapter_dir)


def test_explicit_override_clears_checkpoint_provenance(tmp_path):
    adapter = Lora(rank=2, path=tmp_path)
    adapter._gptqmodel_checkpoint_source = (str(tmp_path), "unused")
    configure_explicit_remote_adapter(adapter, allow_remote=False)
    assert adapter._gptqmodel_checkpoint_source is None


@pytest.mark.parametrize("path", [None, ""])
def test_merged_adapter_does_not_require_external_path(path):
    adapter = Lora(rank=2, path=path)
    configure_explicit_remote_adapter(adapter, allow_remote=False)
    assert adapter.path == path


@pytest.mark.parametrize("filename", ["adapter_config.json", "adapter_model.safetensors"])
def test_checkpoint_cache_links_cannot_leave_same_repository(tmp_path, filename):
    checkpoint = tmp_path / "models--trusted--model" / "snapshots" / ("a" * 40)
    adapter_dir = checkpoint / "adapter"
    adapter_dir.mkdir(parents=True)
    other_blob = tmp_path / "models--other--model" / "blobs" / ("b" * 64)
    (adapter_dir / filename).symlink_to(other_blob)
    with pytest.raises(ValueError, match="cannot link outside"):
        resolve_checkpoint_adapter_path("adapter", str(checkpoint))
