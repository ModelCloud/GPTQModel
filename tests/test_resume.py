# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import atexit
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from gptqmodel.looper.resume import (
    RESUME_ENV_FLAG,
    RESUME_STATE_FILENAME,
    _checkpoint_fingerprint,
    _offloaded_layer_modules,
    _resume_fingerprint,
    activation_cache_available,
    calibration_dataset_hash,
    load_activation_cache,
    read_resume_target,
    restore_completed_layer,
    save_activation_cache,
    write_resume_marker,
)
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from safetensors.torch import save_file as safetensors_save_file
import gptqmodel.looper.stage_layer as stage_layer_module
from gptqmodel.looper.stage_layer import (
    _is_resume_fastforward_candidate,
    _resume_replay_layer,
    _resume_restore_only_layer,
    _should_drain_finalize_futures_synchronously,
)


@contextmanager
def _resume_opted_in():
    """GPTQMODEL_RESUME=1 must be set on the run doing the writing too, not
    just on a later restart -- see resume_state_path's docstring."""
    previous = os.environ.get(RESUME_ENV_FLAG)
    os.environ[RESUME_ENV_FLAG] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(RESUME_ENV_FLAG, None)
        else:
            os.environ[RESUME_ENV_FLAG] = previous


def _make_qcfg(offload_to_disk_path=None, wait_for_submodule_finalizers=False):
    return SimpleNamespace(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        true_sequential=True,
        quant_method="gptq",
        format="gptq",
        dynamic=None,
        pack_dtype=torch.int32,
        offload_to_disk=offload_to_disk_path is not None,
        offload_to_disk_path=offload_to_disk_path,
        wait_for_submodule_finalizers=wait_for_submodule_finalizers,
    )


_MISSING = object()
_SHARED_CHECKPOINT_DIR = None


def _shared_checkpoint_dir() -> str:
    """A stable dummy checkpoint dir so _make_looper's default gives every
    test a real, non-empty checkpoint_fingerprint (same value every call --
    it's the same untouched file) without each test needing its own tmp_path.
    Not a tmp_path fixture itself since this is shared module-wide rather
    than per-test; registered for cleanup at interpreter exit instead."""
    global _SHARED_CHECKPOINT_DIR
    if _SHARED_CHECKPOINT_DIR is None:
        d = tempfile.mkdtemp(prefix="resume_test_checkpoint_")
        with open(os.path.join(d, "model.safetensors"), "wb") as fp:
            fp.write(b"fake weights")
        atexit.register(shutil.rmtree, d, ignore_errors=True)
        _SHARED_CHECKPOINT_DIR = d
    return _SHARED_CHECKPOINT_DIR


def _make_looper(qcfg, calibration_dataset, num_batches=1, total_calibration_tokens=4, model_local_path=_MISSING):
    # Mirrors module_looper.py: the hash is computed once and stashed before
    # release_calibration_dataset() frees the raw dataset, so the fingerprint
    # never reads `calibration_dataset` itself, only this stashed hash.
    if model_local_path is _MISSING:
        model_local_path = _shared_checkpoint_dir()
    processor = SimpleNamespace(
        num_batches=num_batches,
        total_calibration_tokens=total_calibration_tokens,
        calibration_dataset_hash=calibration_dataset_hash(calibration_dataset),
    )
    model_config = SimpleNamespace(_name_or_path="dummy", hidden_size=8, vocab_size=16)
    gptq_model = SimpleNamespace(
        quantize_config=qcfg, model=SimpleNamespace(config=model_config), model_local_path=model_local_path,
    )
    return SimpleNamespace(gptq_model=gptq_model, processors=[processor])


# -- calibration content fingerprinting --------------------------------------


def test_calibration_hash_is_deterministic():
    dataset = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7, 8]}]

    assert calibration_dataset_hash(dataset) == calibration_dataset_hash(dataset)


def test_calibration_hash_detects_content_change_with_same_token_count():
    dataset_a = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7, 8]}]
    dataset_b = [{"input_ids": [1, 2, 3, 4]}, {"input_ids": [5, 6, 7, 9]}]

    assert calibration_dataset_hash(dataset_a) != calibration_dataset_hash(dataset_b)


def test_calibration_hash_handles_tensor_and_dict_like_rows():
    """Real calibration rows can be plain dicts or dict-like (e.g. a
    tokenizer's BatchEncoding) holding a tensor, not a raw Python list."""
    from collections import UserDict

    row = UserDict({"input_ids": torch.tensor([[1, 2, 3, 4]])})

    assert calibration_dataset_hash([row]) != calibration_dataset_hash([])
    assert calibration_dataset_hash([row]) == calibration_dataset_hash([row])


def test_calibration_hash_distinguishes_sample_boundaries():
    """Same flat token stream, different split into samples, must not collide."""
    dataset_a = [{"input_ids": [1, 2]}, {"input_ids": [3]}]
    dataset_b = [{"input_ids": [1]}, {"input_ids": [2, 3]}]

    assert calibration_dataset_hash(dataset_a) != calibration_dataset_hash(dataset_b)


def test_calibration_hash_detects_attention_mask_difference_at_equal_input_ids():
    """Same tokens, different mask -- calibration consumes a different set
    of positions (loop_processor.py prefers attention_mask's sum over raw
    input_ids length), so this must not fingerprint as unchanged."""
    dataset_a = [{"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}]
    dataset_b = [{"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 1]}]

    assert calibration_dataset_hash(dataset_a) != calibration_dataset_hash(dataset_b)


def test_calibration_hash_detects_padding_position_difference():
    """Same tokens and same valid-token count, but padding on the other side."""
    dataset_a = [{"input_ids": [0, 1, 2, 3], "attention_mask": [0, 1, 1, 1]}]
    dataset_b = [{"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}]

    assert calibration_dataset_hash(dataset_a) != calibration_dataset_hash(dataset_b)


def test_calibration_hash_handles_gpu_tensor_calibration_data():
    if not torch.cuda.is_available():
        return
    row_cpu = {"input_ids": torch.tensor([1, 2, 3, 4])}
    row_gpu = {"input_ids": torch.tensor([1, 2, 3, 4], device="cuda")}

    assert calibration_dataset_hash([row_cpu]) == calibration_dataset_hash([row_gpu])


def test_calibration_hash_failure_does_not_produce_a_shared_sentinel():
    """A hash failure must never make two different (both-failed) datasets
    look identical -- that would let resume reuse mismatched state."""

    class _UnreadableRow:
        def __getitem__(self, key):
            raise RuntimeError("simulated unreadable calibration row")

    hash_a = calibration_dataset_hash([_UnreadableRow()])
    hash_b = calibration_dataset_hash([_UnreadableRow()])

    assert hash_a != hash_b
    assert hash_a != calibration_dataset_hash([{"input_ids": [1, 2, 3, 4]}])


def test_calibration_hash_survives_dataset_release():
    """The hash must be captured before release_calibration_dataset() frees
    the raw dataset -- see module_looper.py's release step."""
    dataset = [{"input_ids": [1, 2, 3, 4]}]
    processor = SimpleNamespace(calibration_dataset=dataset)

    processor.calibration_dataset_hash = calibration_dataset_hash(processor.calibration_dataset)
    del processor.calibration_dataset

    assert processor.calibration_dataset_hash == calibration_dataset_hash(dataset)


def test_checkpoint_fingerprint_empty_for_missing_path():
    assert _checkpoint_fingerprint(None) == ""
    assert _checkpoint_fingerprint("/no/such/directory") == ""


def test_checkpoint_fingerprint_stable_for_unchanged_files(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    assert _checkpoint_fingerprint(str(tmp_path)) == _checkpoint_fingerprint(str(tmp_path))


def test_checkpoint_fingerprint_changes_when_a_weight_file_is_replaced(tmp_path):
    """A same-path checkpoint swap (different revision, edited weights) must
    not be mistaken for the same on-disk state the marker was written for."""
    weight_file = tmp_path / "model.safetensors"
    weight_file.write_bytes(b"weights")
    before = _checkpoint_fingerprint(str(tmp_path))

    weight_file.write_bytes(b"different weights, different size")
    os.utime(weight_file, (0, 0))  # force an mtime change too, in case sizes ever collide
    after = _checkpoint_fingerprint(str(tmp_path))

    assert before != after


def test_fingerprint_differs_when_calibration_content_changes_at_equal_size():
    qcfg = _make_qcfg()
    dataset_a = [{"input_ids": [1, 2, 3, 4]}]
    dataset_b = [{"input_ids": [1, 2, 3, 9]}]

    fp_a = _resume_fingerprint(_make_looper(qcfg, dataset_a), layer_count=3)
    fp_b = _resume_fingerprint(_make_looper(qcfg, dataset_b), layer_count=3)

    assert fp_a["calibration_batches"] == fp_b["calibration_batches"]
    assert fp_a["calibration_tokens"] == fp_b["calibration_tokens"]
    assert fp_a["calibration_hash"] != fp_b["calibration_hash"]


# -- activation cache integrity -----------------------------------------------


def test_activation_cache_round_trips(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])
    layer_inputs = [[torch.randn(2, 3)]]

    with _resume_opted_in():
        save_activation_cache(looper, layer_index=0, layer_count=3, layer_inputs=layer_inputs)

        assert activation_cache_available(looper, layer_index=0, layer_count=3)
        cached = load_activation_cache(looper, layer_index=0, layer_count=3)
    assert cached is not None
    cached_inputs, shared_kv = cached
    assert torch.equal(cached_inputs[0][0], layer_inputs[0][0])
    assert shared_kv is None


def test_activation_cache_unavailable_when_data_file_is_corrupted(tmp_path):
    """Metadata alone must not be trusted: a truncated/corrupt safetensors
    file must make the cache unavailable, not just make load fail later
    after earlier layers already skipped their forward replay on its say."""
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        save_activation_cache(looper, layer_index=0, layer_count=3, layer_inputs=[[torch.randn(2, 3)]])

        data_path = tmp_path / "resume_activation_cache" / "activations.safetensors"
        data_path.write_bytes(b"not a real safetensors file")

        assert not activation_cache_available(looper, layer_index=0, layer_count=3)
        assert load_activation_cache(looper, layer_index=0, layer_count=3) is None


# -- write_resume_marker / read_resume_target round trip ----------------------


def test_write_resume_marker_noop_without_opt_in(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    os.environ.pop(RESUME_ENV_FLAG, None)
    write_resume_marker(looper, layer_index=0, layer_count=3, finalized_count=5)

    assert not (tmp_path / RESUME_STATE_FILENAME).exists()


def test_write_resume_marker_writes_fingerprinted_json(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        write_resume_marker(looper, layer_index=2, layer_count=6, finalized_count=31)

    payload = json.loads((tmp_path / RESUME_STATE_FILENAME).read_text())
    assert payload["last_completed_layer"] == 2
    assert payload["layer_finalized_counts"] == {"2": 31}
    assert payload["fingerprint"] == _resume_fingerprint(looper, layer_count=6)


def test_write_resume_marker_preserves_prior_counts_on_matching_fingerprint(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        write_resume_marker(looper, layer_index=0, layer_count=6, finalized_count=31)
        write_resume_marker(looper, layer_index=1, layer_count=6, finalized_count=31)

    payload = json.loads((tmp_path / RESUME_STATE_FILENAME).read_text())
    assert payload["last_completed_layer"] == 1
    assert payload["layer_finalized_counts"] == {"0": 31, "1": 31}


def test_write_resume_marker_resets_counts_when_fingerprint_changes(tmp_path):
    """A config/calibration change mid-marker-history must not let a stale
    per-layer count from a different run silently carry forward."""
    qcfg_a = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper_a = _make_looper(qcfg_a, [{"input_ids": [1, 2, 3, 4]}])
    qcfg_b = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper_b = _make_looper(qcfg_b, [{"input_ids": [9, 9, 9, 9]}])  # different calibration data

    with _resume_opted_in():
        write_resume_marker(looper_a, layer_index=0, layer_count=6, finalized_count=31)
        write_resume_marker(looper_b, layer_index=1, layer_count=6, finalized_count=31)

    payload = json.loads((tmp_path / RESUME_STATE_FILENAME).read_text())
    assert payload["layer_finalized_counts"] == {"1": 31}


def test_read_resume_target_returns_none_without_opt_in(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        write_resume_marker(looper, layer_index=2, layer_count=6, finalized_count=31)

    os.environ.pop(RESUME_ENV_FLAG, None)
    assert read_resume_target(looper, layer_count=6) is None


def test_read_resume_target_returns_none_without_a_marker_file(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        assert read_resume_target(looper, layer_count=6) is None


def test_read_resume_target_returns_none_on_corrupt_marker(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])
    (tmp_path / RESUME_STATE_FILENAME).write_text("{not valid json")

    with _resume_opted_in():
        assert read_resume_target(looper, layer_count=6) is None


def test_read_resume_target_returns_none_for_multi_processor_pipeline(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        write_resume_marker(looper, layer_index=2, layer_count=6, finalized_count=31)

    looper.processors.append(looper.processors[0])  # now 2 processors
    with _resume_opted_in():
        assert read_resume_target(looper, layer_count=6) is None


def test_read_resume_target_returns_none_on_fingerprint_mismatch(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper_original = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])
    looper_changed = _make_looper(qcfg, [{"input_ids": [9, 9, 9, 9]}])  # different calibration data

    with _resume_opted_in():
        write_resume_marker(looper_original, layer_index=2, layer_count=6, finalized_count=31)
        assert read_resume_target(looper_changed, layer_count=6) is None


def test_read_resume_target_returns_none_for_invalid_last_completed_layer(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        write_resume_marker(looper, layer_index=0, layer_count=6, finalized_count=31)
        payload = json.loads((tmp_path / RESUME_STATE_FILENAME).read_text())
        payload["last_completed_layer"] = 99  # out of range for layer_count=6
        (tmp_path / RESUME_STATE_FILENAME).write_text(json.dumps(payload))

        assert read_resume_target(looper, layer_count=6) is None


def test_read_resume_target_returns_last_completed_layer_on_matching_fingerprint(tmp_path):
    """The end-to-end happy path this whole fingerprint machinery exists for:
    an unchanged run's own marker must actually be accepted, not just
    correctly rejected when something changed."""
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        write_resume_marker(looper, layer_index=0, layer_count=6, finalized_count=31)
        write_resume_marker(looper, layer_index=3, layer_count=6, finalized_count=31)

        assert read_resume_target(looper, layer_count=6) == 3


def test_read_resume_target_returns_none_when_checkpoint_cannot_be_fingerprinted(tmp_path):
    """An empty checkpoint_fingerprint (no model_local_path, or no
    recognized weight file under it) can't distinguish this checkpoint from
    a different one that also failed to fingerprint -- must never be
    trusted, even if every other field in the marker matches exactly."""
    # model_local_path=None -> _checkpoint_fingerprint returns "" (see
    # test_checkpoint_fingerprint_empty_for_missing_path for that contract).
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}], model_local_path=None)

    with _resume_opted_in():
        write_resume_marker(looper, layer_index=3, layer_count=6, finalized_count=31)

        assert read_resume_target(looper, layer_count=6) is None


def test_read_resume_target_returns_none_on_checkpoint_fingerprint_mismatch(tmp_path):
    """Same calibration data and config, but the checkpoint at
    model_local_path changed (revision swap / edited weights) -- must be
    treated as a different run, not an unchanged one."""
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path))
    original_checkpoint = _shared_checkpoint_dir()
    looper_original = _make_looper(
        qcfg, [{"input_ids": [1, 2, 3, 4]}], model_local_path=original_checkpoint,
    )

    swapped_checkpoint = tmp_path / "swapped_checkpoint"
    swapped_checkpoint.mkdir()
    (swapped_checkpoint / "model.safetensors").write_bytes(b"different weights entirely")
    looper_swapped = _make_looper(
        qcfg, [{"input_ids": [1, 2, 3, 4]}], model_local_path=str(swapped_checkpoint),
    )

    with _resume_opted_in():
        write_resume_marker(looper_original, layer_index=3, layer_count=6, finalized_count=31)

        assert read_resume_target(looper_swapped, layer_count=6) is None


# -- _offloaded_layer_modules: atomic-offload .tmp/.old recovery --------------


def test_offloaded_layer_modules_finds_normal_module_dir(tmp_path):
    module_dir = tmp_path / "layers.0.self_attn.q_proj"
    module_dir.mkdir()
    (module_dir / "module.safetensors").write_bytes(b"x")

    found = _offloaded_layer_modules(str(tmp_path), "layers.0")

    assert found == [("layers.0.self_attn.q_proj", str(module_dir / "module.safetensors"))]


def test_offloaded_layer_modules_skips_in_flight_tmp_dir(tmp_path):
    """A `.tmp` dir is a write still in progress -- never a valid source,
    unlike `.old`, which can be a complete pre-swap backup."""
    tmp_dir = tmp_path / "layers.0.self_attn.q_proj.tmp"
    tmp_dir.mkdir()
    (tmp_dir / "module.safetensors").write_bytes(b"x")

    assert _offloaded_layer_modules(str(tmp_path), "layers.0") == []


def test_offloaded_layer_modules_recovers_lone_old_dir_from_interrupted_swap(tmp_path):
    """No final directory, only `.old` -- the crash landed between the
    rmtree-old and rename-into-place steps of the atomic swap."""
    old_dir = tmp_path / "layers.0.self_attn.q_proj.old"
    old_dir.mkdir()
    (old_dir / "module.safetensors").write_bytes(b"x")

    found = _offloaded_layer_modules(str(tmp_path), "layers.0")

    assert found == [("layers.0.self_attn.q_proj", str(old_dir / "module.safetensors"))]


def test_offloaded_layer_modules_ignores_stale_old_dir_when_live_copy_exists(tmp_path):
    """A finished swap can leave a stale `.old` behind (cleanup is
    best-effort) -- the live directory is authoritative and must not be
    shadowed or double-counted alongside its own leftover backup."""
    live_dir = tmp_path / "layers.0.self_attn.q_proj"
    live_dir.mkdir()
    (live_dir / "module.safetensors").write_bytes(b"live")
    old_dir = tmp_path / "layers.0.self_attn.q_proj.old"
    old_dir.mkdir()
    (old_dir / "module.safetensors").write_bytes(b"stale")

    found = _offloaded_layer_modules(str(tmp_path), "layers.0")

    assert found == [("layers.0.self_attn.q_proj", str(live_dir / "module.safetensors"))]


# -- restore_completed_layer: corrupt/incomplete offload safety net -----------


def _make_restore_looper(offload_root, get_submodule):
    qcfg = SimpleNamespace(offload_to_disk_path=str(offload_root))
    model_model = SimpleNamespace(get_submodule=get_submodule)
    gptq_model = SimpleNamespace(quantize_config=qcfg, model=model_model)
    return SimpleNamespace(gptq_model=gptq_model)


def test_restore_completed_layer_raises_on_missing_state_dict_keys(tmp_path):
    module_dir = tmp_path / "layers.0.self_attn.q_proj"
    module_dir.mkdir()
    safetensors_save_file({"qweight": torch.zeros(2, 2)}, str(module_dir / "module.safetensors"))

    quant_module = MagicMock(spec=BaseQuantLinear)
    quant_module.load_state_dict.return_value = (["scales", "qzeros"], [])  # missing keys
    looper = _make_restore_looper(tmp_path, get_submodule=lambda name: quant_module)

    with pytest.raises(RuntimeError, match="missing keys"):
        restore_completed_layer(looper, "layers.0")


def test_restore_completed_layer_warns_but_succeeds_on_unexpected_keys(tmp_path):
    module_dir = tmp_path / "layers.0.self_attn.q_proj"
    module_dir.mkdir()
    safetensors_save_file({"qweight": torch.zeros(2, 2)}, str(module_dir / "module.safetensors"))

    quant_module = MagicMock(spec=BaseQuantLinear)
    quant_module.load_state_dict.return_value = ([], ["some_extra_buffer"])
    looper = _make_restore_looper(tmp_path, get_submodule=lambda name: quant_module)

    restored = restore_completed_layer(looper, "layers.0")

    assert restored == ["layers.0.self_attn.q_proj"]


def test_restore_completed_layer_skips_module_missing_from_live_model(tmp_path):
    """Offload dirs can hold entries (renamed aliases) absent from the live
    module tree -- those must be skipped, not crash the whole restore."""
    module_dir = tmp_path / "layers.0.self_attn.q_proj"
    module_dir.mkdir()
    (module_dir / "module.safetensors").write_bytes(b"x")

    def get_submodule(name):
        raise AttributeError(name)

    looper = _make_restore_looper(tmp_path, get_submodule=get_submodule)

    assert restore_completed_layer(looper, "layers.0") == []


# -- synchronous drain gating --------------------------------------------------


def test_async_drain_default_when_resume_not_configured():
    qcfg = _make_qcfg(offload_to_disk_path=None, wait_for_submodule_finalizers=False)
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    os.environ.pop(RESUME_ENV_FLAG, None)

    assert not _should_drain_finalize_futures_synchronously(looper, finalize_tasks=[(object(),)])


def test_async_drain_unchanged_for_offload_users_who_did_not_opt_into_resume(tmp_path):
    """offload_to_disk alone must not force sync drain -- only opting in via
    GPTQMODEL_RESUME=1 should, so existing offload_to_disk users see no
    behavior change from this feature unless they ask for it."""
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path), wait_for_submodule_finalizers=False)
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    os.environ.pop(RESUME_ENV_FLAG, None)

    assert not _should_drain_finalize_futures_synchronously(looper, finalize_tasks=[(object(),)])


def test_sync_drain_forced_when_resume_marker_path_configured(tmp_path):
    qcfg = _make_qcfg(offload_to_disk_path=str(tmp_path), wait_for_submodule_finalizers=False)
    looper = _make_looper(qcfg, [{"input_ids": [1, 2, 3, 4]}])

    with _resume_opted_in():
        assert _should_drain_finalize_futures_synchronously(looper, finalize_tasks=[(object(),)])


# -- _resume_replay_layer's cache-hit/cache-miss branch -----------------------


class _FakeFluent:
    """Stands in for the progress-bar's title().subtitle().draw() chain."""

    def title(self, *args, **kwargs):
        return self

    def subtitle(self, *args, **kwargs):
        return self

    def draw(self, *args, **kwargs):
        return self


@contextmanager
def _stubbed_resume_replay_collaborators(
    *,
    load_activation_cache_impl=None,
    replay_layer_outputs_impl=None,
    restore_completed_layer_impl=None,
    restore_completed_layer_class_only_impl=None,
    marker_layer_finalized_count_impl=None,
):
    """Replaces _resume_replay_layer/_resume_restore_only_layer's heavy
    model/replay collaborators with lightweight fakes, so only the branching
    that is specific to each function is exercised -- the real model/forward
    path is already covered end-to-end by the dummy-model resume driver."""
    patched = {
        "materialize_model": lambda module: None,
        "get_device": lambda module: torch.device("cpu"),
        "MODULE_CONVERTER_MAP": {},
        "find_modules": lambda module, name="": {},
        "_replay_layer_outputs": replay_layer_outputs_impl or (lambda *a, **k: "REPLAYED_OUTPUTS"),
        "load_activation_cache": load_activation_cache_impl or (lambda looper, layer_index, layer_count: None),
        "restore_completed_layer": restore_completed_layer_impl or (lambda looper, layer_prefix: ["fake.module"]),
        "restore_completed_layer_class_only": (
            restore_completed_layer_class_only_impl or (lambda looper, layer_prefix: ["fake.module"])
        ),
        "marker_layer_finalized_count": marker_layer_finalized_count_impl or (lambda qcfg, layer_index: None),
    }
    originals = {name: getattr(stage_layer_module, name) for name in patched}
    for name, value in patched.items():
        setattr(stage_layer_module, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(stage_layer_module, name, value)


def _make_resume_replay_layer_args():
    fake_module = torch.nn.Module()

    class _FakeGptqModel:
        quantize_config = SimpleNamespace(
            calibration_data_device=None, offload_to_disk=False, offload_to_disk_path=None,
        )
        model = SimpleNamespace(config=SimpleNamespace(model_type="unit-test-fake"))

        def shell_module_materialize(self, target_submodule, device):
            return fake_module

        def extract_layers_node(self):
            return "layers"

        def pre_quantize(self, module):
            return module

        def post_quantize(self, module):
            return module

    received = []
    processor = SimpleNamespace(
        inputs_cache=SimpleNamespace(
            layer_inputs=[], layer_input_kwargs=[], position_ids=None, attention_masks=None,
        ),
        clear_cache_data=lambda: None,
        receive_layer_inputs=received.append,
    )
    looper = SimpleNamespace(gptq_model=_FakeGptqModel(), processors=[processor])

    kwargs = dict(
        looper=looper,
        layers=[fake_module],
        layer_index=0,
        layer_name="layers.0",
        layer_title="Quantizing layer 0",
        layer_count=3,
        shared_kv_cache_dict={},
        pb=_FakeFluent(),
        log=SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None),
        region_timer=None,
    )
    return kwargs, received


def test_resume_replay_layer_falls_back_to_full_replay_on_cache_miss():
    """When there is no usable activation cache for this layer (missing or
    failed to load), the layer must still get its input the normal way --
    a full forward replay -- not silently skip it."""
    calls = {"load_cache": 0, "replay": 0}

    def fake_load_activation_cache(looper, layer_index, layer_count):
        calls["load_cache"] += 1
        return None  # simulates a missing/corrupt cache

    def fake_replay_layer_outputs(*args, **kwargs):
        calls["replay"] += 1
        return "REPLAYED_OUTPUTS"

    kwargs, received = _make_resume_replay_layer_args()
    with _stubbed_resume_replay_collaborators(
        load_activation_cache_impl=fake_load_activation_cache,
        replay_layer_outputs_impl=fake_replay_layer_outputs,
    ):
        _resume_replay_layer(**kwargs, preloaded_cache=None)

    assert calls["load_cache"] == 1
    assert calls["replay"] == 1
    assert received == ["REPLAYED_OUTPUTS"]


def test_resume_replay_layer_uses_preloaded_cache_without_reloading_or_replaying():
    """When the caller already loaded this layer's cache (the common path,
    since run_layer_stage loads it once up front), _resume_replay_layer must
    reuse that result rather than re-reading the file or doing a full replay."""
    calls = {"load_cache": 0, "replay": 0}

    def fake_load_activation_cache(looper, layer_index, layer_count):
        calls["load_cache"] += 1
        return None

    def fake_replay_layer_outputs(*args, **kwargs):
        calls["replay"] += 1
        return "REPLAYED_OUTPUTS"

    cached_batch = [torch.randn(2, 3)]
    kwargs, received = _make_resume_replay_layer_args()
    with _stubbed_resume_replay_collaborators(
        load_activation_cache_impl=fake_load_activation_cache,
        replay_layer_outputs_impl=fake_replay_layer_outputs,
    ):
        _resume_replay_layer(**kwargs, preloaded_cache=([cached_batch], None))

    assert calls["load_cache"] == 0
    assert calls["replay"] == 0
    assert len(received) == 1
    assert torch.equal(received[0][0][0], cached_batch[0])


# -- _resume_restore_only_layer's class-restore + count validation -----------


def _make_resume_restore_only_layer_args():
    fake_module = torch.nn.Module()

    class _FakeGptqModel:
        quantize_config = SimpleNamespace()
        model = SimpleNamespace(config=SimpleNamespace(model_type="unit-test-fake"))

        def shell_module_materialize(self, target_submodule, device):
            return fake_module

        def extract_layers_node(self):
            return "layers"

    looper = SimpleNamespace(gptq_model=_FakeGptqModel(), processors=[])

    kwargs = dict(
        looper=looper,
        layers=[fake_module],
        layer_index=0,
        layer_name="layers.0",
        layer_title="Quantizing layer 0",
        shared_kv_cache_dict={},
        pb=_FakeFluent(),
        log=SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None),
    )
    return kwargs


def test_resume_restore_only_layer_raises_when_nothing_restored():
    """An empty restored set means the marker claims this layer is complete
    but the offload directory has nothing under its prefix -- fail loudly
    rather than silently save an unquantized layer."""
    kwargs = _make_resume_restore_only_layer_args()
    with _stubbed_resume_replay_collaborators(restore_completed_layer_class_only_impl=lambda looper, prefix: []):
        with pytest.raises(RuntimeError, match="no offloaded"):
            _resume_restore_only_layer(**kwargs)


def test_resume_restore_only_layer_raises_on_restored_count_mismatch():
    """Marker recorded 31 finalized modules for this layer but only 2 were
    actually restorable -- the offload directory is incomplete/damaged."""
    kwargs = _make_resume_restore_only_layer_args()
    with _stubbed_resume_replay_collaborators(
        restore_completed_layer_class_only_impl=lambda looper, prefix: ["a", "b"],
        marker_layer_finalized_count_impl=lambda qcfg, layer_index: 31,
    ):
        with pytest.raises(RuntimeError, match="incomplete or damaged"):
            _resume_restore_only_layer(**kwargs)


def test_resume_restore_only_layer_succeeds_when_count_matches():
    kwargs = _make_resume_restore_only_layer_args()
    with _stubbed_resume_replay_collaborators(
        restore_completed_layer_class_only_impl=lambda looper, prefix: ["a", "b"],
        marker_layer_finalized_count_impl=lambda qcfg, layer_index: 2,
    ):
        _resume_restore_only_layer(**kwargs)  # must not raise


def test_resume_restore_only_layer_succeeds_when_marker_predates_the_count_field():
    """An older marker with no layer_finalized_counts entry must not block
    resume -- just skip the count cross-check (a warning is still logged)."""
    kwargs = _make_resume_restore_only_layer_args()
    with _stubbed_resume_replay_collaborators(
        restore_completed_layer_class_only_impl=lambda looper, prefix: ["a", "b"],
        marker_layer_finalized_count_impl=lambda qcfg, layer_index: None,
    ):
        _resume_restore_only_layer(**kwargs)  # must not raise


# -- embeddings/lm_head excluded from the transformer-layer resume path -------


def test_embeddings_module_excluded_from_resume_fastforward():
    assert not _is_resume_fastforward_candidate(
        is_embeddings_module=True, layer_index=0, resume_target=0
    )


def test_real_layer_zero_is_a_resume_fastforward_candidate():
    assert _is_resume_fastforward_candidate(
        is_embeddings_module=False, layer_index=0, resume_target=0
    )


def test_no_fastforward_without_a_resume_target():
    assert not _is_resume_fastforward_candidate(
        is_embeddings_module=False, layer_index=0, resume_target=None
    )
