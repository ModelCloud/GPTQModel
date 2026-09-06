# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from gptqmodel.looper.resume import (
    RESUME_ENV_FLAG,
    _resume_fingerprint,
    activation_cache_available,
    calibration_dataset_hash,
    load_activation_cache,
    save_activation_cache,
)
from gptqmodel.looper.stage_layer import (
    _is_resume_fastforward_candidate,
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


def _make_looper(qcfg, calibration_dataset, num_batches=1, total_calibration_tokens=4):
    # Mirrors module_looper.py: the hash is computed once and stashed before
    # release_calibration_dataset() frees the raw dataset, so the fingerprint
    # never reads `calibration_dataset` itself, only this stashed hash.
    processor = SimpleNamespace(
        num_batches=num_batches,
        total_calibration_tokens=total_calibration_tokens,
        calibration_dataset_hash=calibration_dataset_hash(calibration_dataset),
    )
    model_config = SimpleNamespace(_name_or_path="dummy", hidden_size=8, vocab_size=16)
    gptq_model = SimpleNamespace(quantize_config=qcfg, model=SimpleNamespace(config=model_config))
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


def test_calibration_hash_survives_dataset_release():
    """The hash must be captured before release_calibration_dataset() frees
    the raw dataset -- see module_looper.py's release step."""
    dataset = [{"input_ids": [1, 2, 3, 4]}]
    processor = SimpleNamespace(calibration_dataset=dataset)

    processor.calibration_dataset_hash = calibration_dataset_hash(processor.calibration_dataset)
    del processor.calibration_dataset

    assert processor.calibration_dataset_hash == calibration_dataset_hash(dataset)


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
