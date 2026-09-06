# SPDX-License-Identifier: Apache-2.0
"""Layer-boundary snapshot markers and mid-quantization resume support.

Sequential GPTQ quantization of very large models can run for tens of hours.
This module makes a crash survivable by exploiting state the looper already
persists: every finalized module is packed into its quant-linear form and
disk-offloaded under ``offload_to_disk_path/<module full name>/``. All that is
missing for a restart is (1) a durable record of the last layer whose modules
are all finalized on disk, and (2) a fast-forward path that rebuilds the
calibration activations for the first unfinished layer.

Opt-in via env ``GPTQMODEL_RESUME=1`` on the run you want to be resumable
(with ``offload_to_disk`` configured) -- the marker (``quant_resume_state.json``)
is then written at each layer boundary, forcing submodule finalizers to drain
synchronously so it never lies about what's actually durable on disk. Runs
that don't set the flag pay none of this: no marker, no forced sync drain.

On restart with the same env var set, the layer stage replays completed
layers forward-only: the original (meta) submodules are swapped for quant
modules loaded from the offload directory, one whole-layer forward regenerates
the next layer's inputs, and quantization resumes at the first unfinished
layer. Replay runs through the original (pre-quantization) weights, so the
regenerated activations match the original run exactly, up to the same
floating-point reduction-order nondeterminism any rerun has.

The forward replay for the single most-recently-completed layer can itself be
skipped: ``save_activation_cache``/``load_activation_cache`` persist that
layer's output (and any paired ``shared_kv_cache_dict`` state) the first time
it is computed, so a *second* resume that fast-forwards through the same
layer again reuses the cached result instead of recomputing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import List, Optional, Tuple

import torch
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from ..nn_modules.qlinear import BaseQuantLinear
from ..quantization.config import resolve_quant_format
from ..utils.logger import setup_logger
from ..utils.model import create_quant_module

log = setup_logger()

RESUME_STATE_FILENAME = "quant_resume_state.json"
RESUME_ENV_FLAG = "GPTQMODEL_RESUME"


def calibration_dataset_hash(dataset) -> str:
    """SHA-256 over each sample's input_ids, so equal-length calibration data with different content is detected.

    Must be called before `LoopProcessor.release_calibration_dataset()` frees
    `dataset` -- by the time any layer's resume marker is written, it is
    already gone, so callers stash this on the processor up front (see
    `module_looper.py`) and `_resume_fingerprint` only ever reads that.
    """
    hasher = hashlib.sha256()
    try:
        for row in dataset:
            # Same access pattern as this row's other consumer (loop_processor.py's
            # avg-length check): works for a plain dict or a BatchEncoding alike.
            input_ids = row["input_ids"]
            ids = input_ids.reshape(-1).tolist() if torch.is_tensor(input_ids) else list(input_ids)
            hasher.update(torch.tensor(ids, dtype=torch.int64).numpy().tobytes())
    except Exception:  # pragma: no cover - hashing must never break the run
        return ""
    return hasher.hexdigest()


def _resume_fingerprint(looper, layer_count: int) -> dict:
    """Run parameters that must match for on-disk layer state to be reusable.

    Includes the model identity, quantization method/format, and calibration
    size: quantized layers from a different checkpoint, a different output
    format, or different calibration data load without shape errors, so
    accepting them would silently produce a hybrid model.
    """
    qcfg = looper.gptq_model.quantize_config
    model_config = getattr(looper.gptq_model.model, "config", None)
    try:
        # These are fixed at calibration load; inputs_cache would be unstable
        # here (it is empty at the final layer's marker write), and the raw
        # calibration_dataset itself is already freed by this point.
        processor = looper.processors[-1]
        calibration_batches = int(processor.num_batches or 0)
        calibration_tokens = int(getattr(processor, "total_calibration_tokens", 0) or 0)
        calibration_hash = getattr(processor, "calibration_dataset_hash", "") or ""
    except (AttributeError, IndexError, TypeError):
        calibration_batches = 0
        calibration_tokens = 0
        calibration_hash = ""
    return {
        "bits": int(qcfg.bits),
        "group_size": int(qcfg.group_size),
        "desc_act": bool(qcfg.desc_act),
        "sym": bool(qcfg.sym),
        "true_sequential": bool(qcfg.true_sequential),
        "quant_method": str(qcfg.quant_method),
        "format": str(qcfg.format),
        "dynamic": repr(qcfg.dynamic) if qcfg.dynamic else None,
        "pack_dtype": str(qcfg.pack_dtype),
        "layer_count": int(layer_count),
        "model_name_or_path": str(getattr(model_config, "_name_or_path", "") or ""),
        "hidden_size": int(getattr(model_config, "hidden_size", 0) or 0),
        "vocab_size": int(getattr(model_config, "vocab_size", 0) or 0),
        "calibration_batches": int(calibration_batches),
        "calibration_tokens": int(calibration_tokens),
        "calibration_hash": calibration_hash,
    }


def resume_state_path(qcfg) -> Optional[str]:
    """Marker path if this run opted into resume, else None.

    Gated on GPTQMODEL_RESUME=1 (not just offload_to_disk) so that offload
    users who never asked for crash-resume keep the plain async finalize
    drain -- opting in costs the same env var on the original run as on a
    later resume of it.
    """
    if os.environ.get(RESUME_ENV_FLAG) != "1":
        return None
    offload_path = getattr(qcfg, "offload_to_disk_path", None)
    if not getattr(qcfg, "offload_to_disk", False) or not offload_path:
        return None
    return os.path.join(offload_path, RESUME_STATE_FILENAME)


def _read_marker_payload(state_path: str) -> Optional[dict]:
    try:
        with open(state_path, encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return None


def write_resume_marker(looper, layer_index: int, layer_count: int, finalized_count: int) -> None:
    """Record that layers ``0..layer_index`` are fully finalized and offloaded.

    ``finalized_count`` is the number of modules this layer finalized; a
    resumed run cross-checks it against what it can actually restore from
    disk, so a partially-written offload directory fails loudly instead of
    silently keeping original weights.

    Callers must only invoke this after this layer's finalize futures were
    drained synchronously AND all of them succeeded; the sequential layer
    loop then guarantees the same for every earlier layer.
    """
    qcfg = looper.gptq_model.quantize_config
    state_path = resume_state_path(qcfg)
    if state_path is None:
        return

    fingerprint = _resume_fingerprint(looper, layer_count)
    layer_counts: dict = {}
    existing = _read_marker_payload(state_path)
    if existing and existing.get("fingerprint") == fingerprint:
        prior_counts = existing.get("layer_finalized_counts")
        if isinstance(prior_counts, dict):
            layer_counts = prior_counts
    layer_counts[str(int(layer_index))] = int(finalized_count)

    payload = {
        "last_completed_layer": int(layer_index),
        "fingerprint": fingerprint,
        "layer_finalized_counts": layer_counts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp_path = f"{state_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        os.replace(tmp_path, state_path)
    except OSError as exc:
        # Marker persistence is best-effort insurance; never fail the quant run.
        log.warn("Resume: failed to write marker %s: %s", state_path, exc)


def _activation_cache_dir(qcfg) -> Optional[str]:
    state_path = resume_state_path(qcfg)
    if state_path is None:
        return None
    return os.path.join(os.path.dirname(state_path), "resume_activation_cache")


def save_activation_cache(
    looper,
    layer_index: int,
    layer_count: int,
    layer_inputs,
    shared_kv_value: Optional[torch.Tensor] = None,
) -> None:
    """Persist this layer's forward output (the next layer's calibration input)
    so a future resume can skip re-replaying layers 0..layer_index. Only the
    last completed layer's cache is kept -- a resume only needs the input for
    the layer right after it.

    `shared_kv_value` is this layer's `shared_kv_cache_dict` entry (state some
    architectures thread into the next layer via `reuse_kv`). It must be
    cached alongside the activations: skipping this layer's forward also
    skips the only place that normally populates that entry.

    This does NOT let a resume skip restoring quant modules from the offload
    directory: that step is still required so the saved model contains the
    packed weights (see restore_completed_layer). It only skips the actual
    forward computation, which is the expensive part.
    """
    qcfg = looper.gptq_model.quantize_config
    cache_dir = _activation_cache_dir(qcfg)
    if cache_dir is None:
        return

    if shared_kv_value is not None and not torch.is_tensor(shared_kv_value):
        # `shared_kv_cache_dict` values can be arbitrary nested dict/list/tuple
        # structures for other model families (see forward_executor.py's use of
        # `nested_move_to`, e.g. Hymba's Mamba SSM state) -- only plain tensors
        # are handled below. Skip caching this layer entirely rather than
        # caching activations without the paired state: a resume that skips
        # this layer's forward but silently drops its kv/state contribution
        # would feed the next layer `prev_kv=None` instead of the real value,
        # which is a wrong-answer bug, not just a slower resume.
        log.warn(
            "Resume: layer %s's shared_kv_cache_dict entry is a %s, not a tensor; "
            "activation caching is not supported for it, skipping.",
            layer_index,
            type(shared_kv_value).__name__,
        )
        return

    tensors = {}
    for batch_idx, batch in enumerate(layer_inputs):
        for item_idx, item in enumerate(batch):
            if torch.is_tensor(item):
                tensors[f"batch_{batch_idx}_{item_idx}"] = item.detach().to("cpu").contiguous()
    if not tensors:
        return

    has_shared_kv = torch.is_tensor(shared_kv_value)
    if has_shared_kv:
        tensors["shared_kv"] = shared_kv_value.detach().to("cpu").contiguous()

    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp_path = os.path.join(cache_dir, "activations.safetensors.tmp")
        final_path = os.path.join(cache_dir, "activations.safetensors")
        safetensors_save_file(tensors, tmp_path)
        os.replace(tmp_path, final_path)

        meta = {
            "layer_index": int(layer_index),
            "num_batches": len(layer_inputs),
            "batch_lengths": [len(batch) for batch in layer_inputs],
            "has_shared_kv": has_shared_kv,
            "fingerprint": _resume_fingerprint(looper, layer_count),
        }
        meta_tmp = os.path.join(cache_dir, "activations.json.tmp")
        meta_final = os.path.join(cache_dir, "activations.json")
        with open(meta_tmp, "w", encoding="utf-8") as fp:
            json.dump(meta, fp)
        os.replace(meta_tmp, meta_final)
    except OSError as exc:
        # Best-effort: a missing/stale cache only costs a slower resume later.
        log.warn("Resume: failed to save activation cache for layer %s: %s", layer_index, exc)


def _read_valid_activation_cache_meta(looper, layer_index: int, layer_count: int) -> Optional[dict]:
    """Metadata-only check: does a valid, matching activation cache exist for `layer_index`?

    Shares the same layer/fingerprint validation as `load_activation_cache` but
    never touches the (potentially multi-GB) tensor payload, so callers that
    only need a yes/no answer -- e.g. deciding whether a whole chain of earlier
    resume-replay layers can skip forward computation entirely -- don't pay for
    a safetensors load just to check.
    """
    qcfg = looper.gptq_model.quantize_config
    cache_dir = _activation_cache_dir(qcfg)
    if cache_dir is None:
        return None

    meta_path = os.path.join(cache_dir, "activations.json")
    data_path = os.path.join(cache_dir, "activations.safetensors")
    if not os.path.isfile(meta_path) or not os.path.isfile(data_path):
        return None

    try:
        with open(meta_path, encoding="utf-8") as fp:
            meta = json.load(fp)
    except (OSError, ValueError):
        return None

    if meta.get("layer_index") != layer_index:
        return None
    if meta.get("fingerprint") != _resume_fingerprint(looper, layer_count):
        log.warn("Resume: activation cache for layer %s has a stale fingerprint; ignoring.", layer_index)
        return None

    return meta


def activation_cache_available(looper, layer_index: int, layer_count: int) -> bool:
    """Whether `load_activation_cache` would hit for `layer_index`.

    Must actually load the data, not just check the metadata: this result
    decides whether earlier layers skip forward replay entirely (see
    `_resume_restore_only_layer`), so a metadata-only check that later turns
    out to not match real, loadable tensors (a corrupt or truncated
    safetensors file) would strand `layer_index`'s replay with none of the
    upstream activations it needs.
    """

    return load_activation_cache(looper, layer_index, layer_count) is not None


def load_activation_cache(looper, layer_index: int, layer_count: int):
    """Load the cached forward output for `layer_index`, if present and valid.

    Returns None (triggering a full replay) when there is no cache, it belongs
    to a different layer, or its fingerprint no longer matches the current run.
    Otherwise returns `(layer_inputs, shared_kv_value)`, where `shared_kv_value`
    is the cached `shared_kv_cache_dict[layer_index]` entry (or None if this
    layer didn't write one) -- see `save_activation_cache` for why this must
    travel with the activations rather than being dropped.
    """
    meta = _read_valid_activation_cache_meta(looper, layer_index, layer_count)
    if meta is None:
        return None

    qcfg = looper.gptq_model.quantize_config
    cache_dir = _activation_cache_dir(qcfg)
    data_path = os.path.join(cache_dir, "activations.safetensors")

    try:
        tensors = safetensors_load_file(data_path)
    except Exception as exc:
        log.warn("Resume: activation cache for layer %s failed to load (%s); ignoring.", layer_index, exc)
        return None

    batch_lengths = meta.get("batch_lengths")
    num_batches = meta.get("num_batches")
    if not isinstance(batch_lengths, list) or not isinstance(num_batches, int):
        return None

    layer_inputs = []
    try:
        for batch_idx in range(num_batches):
            batch = [tensors[f"batch_{batch_idx}_{item_idx}"] for item_idx in range(batch_lengths[batch_idx])]
            layer_inputs.append(batch)
    except KeyError as exc:
        log.warn("Resume: activation cache for layer %s is missing tensor %s; ignoring.", layer_index, exc)
        return None

    shared_kv_value = None
    if meta.get("has_shared_kv"):
        shared_kv_value = tensors.get("shared_kv")
        if shared_kv_value is None:
            log.warn(
                "Resume: activation cache for layer %s is missing its shared_kv tensor; ignoring.",
                layer_index,
            )
            return None

    return layer_inputs, shared_kv_value


def marker_layer_finalized_count(qcfg, layer_index: int) -> Optional[int]:
    """Number of modules the original run finalized for this layer, if recorded."""
    state_path = resume_state_path(qcfg)
    if state_path is None:
        return None
    payload = _read_marker_payload(state_path)
    if not payload:
        return None
    counts = payload.get("layer_finalized_counts")
    if not isinstance(counts, dict):
        return None
    value = counts.get(str(int(layer_index)))
    return int(value) if isinstance(value, int) else None


def read_resume_target(looper, layer_count: int) -> Optional[int]:
    """Return the last fully-quantized layer index if a valid marker allows resume."""
    if os.environ.get(RESUME_ENV_FLAG) != "1":
        return None

    qcfg = looper.gptq_model.quantize_config
    state_path = resume_state_path(qcfg)
    if state_path is None:
        log.warn("Resume: %s=1 requires offload_to_disk with a path; ignoring.", RESUME_ENV_FLAG)
        return None
    if not os.path.isfile(state_path):
        log.warn("Resume: no marker at %s; starting from layer 0.", state_path)
        return None
    if len(looper.processors) != 1:
        log.warn("Resume: only single-processor pipelines are supported; ignoring marker.")
        return None

    try:
        with open(state_path, encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, ValueError) as exc:
        log.warn("Resume: unreadable marker %s (%s); starting from layer 0.", state_path, exc)
        return None

    expected = _resume_fingerprint(looper, layer_count)
    if payload.get("fingerprint") != expected:
        log.warn(
            "Resume: marker fingerprint %s does not match current run %s; starting from layer 0.",
            payload.get("fingerprint"),
            expected,
        )
        return None

    last_completed = payload.get("last_completed_layer")
    if not isinstance(last_completed, int) or not (0 <= last_completed < layer_count):
        log.warn("Resume: invalid last_completed_layer=%r; starting from layer 0.", last_completed)
        return None
    return last_completed


def _offloaded_layer_modules(offload_root: str, layer_prefix: str) -> List[Tuple[str, str]]:
    """List (module full name, safetensors bundle path) persisted for one layer.

    ``<name>.tmp`` / ``<name>.old`` directories are transient states of the
    atomic offload swap. A lone ``.old`` with no final directory means a crash
    hit the instant between the two renames — its bundle is still the complete
    previous state, so recover from it.
    """
    prefix = layer_prefix.rstrip(".") + "."
    entries = set(os.listdir(offload_root))
    found: List[Tuple[str, str]] = []
    for entry in sorted(entries):
        if not entry.startswith(prefix):
            continue
        if entry.endswith(".tmp") or entry.endswith(".old"):
            base = entry.rsplit(".", 1)[0]
            if entry.endswith(".old") and base not in entries:
                bundle = os.path.join(offload_root, entry, "module.safetensors")
                if os.path.isfile(bundle):
                    log.warn(
                        "Resume: recovering `%s` from interrupted offload swap (%s).",
                        base,
                        entry,
                    )
                    found.append((base, bundle))
            continue
        bundle = os.path.join(offload_root, entry, "module.safetensors")
        if os.path.isfile(bundle):
            found.append((entry, bundle))
    return found


def restore_completed_layer(looper, layer_prefix: str) -> List[str]:
    """Swap a completed layer's submodules for quant modules loaded from the offload dir.

    The layer (with original weights) must already be materialized on CPU:
    ``create_quant_module`` asserts CPU residency before replacing a submodule.
    Returns the full names of the restored modules.
    """
    model = looper.gptq_model
    qcfg = model.quantize_config
    offload_root = qcfg.offload_to_disk_path

    restored: List[str] = []
    for full_name, bundle in _offloaded_layer_modules(offload_root, layer_prefix):
        try:
            submodule = model.model.get_submodule(full_name)
        except AttributeError:
            # Offload dirs can hold entries (e.g. renamed aliases) that do not
            # exist in the live module tree; those are not resumable modules.
            log.warn("Resume: offloaded module `%s` not found in model; skipping.", full_name)
            continue

        if not isinstance(submodule, BaseQuantLinear):
            create_quant_module(
                name=full_name,
                linear_cls=model.qlinear_kernel,
                bits=getattr(qcfg, "runtime_bits", qcfg.bits),
                desc_act=qcfg.desc_act,
                dynamic=qcfg.dynamic,
                group_size=qcfg.group_size,
                module=model.model,
                submodule=submodule,
                sym=qcfg.sym,
                device=qcfg.device,
                lm_head_name=model.lm_head,
                pack_dtype=qcfg.pack_dtype,
                format=resolve_quant_format(qcfg.format, qcfg.method),
                register_buffers=True,
            )

        quant_module = model.model.get_submodule(full_name)
        state = safetensors_load_file(bundle)
        missing, unexpected = quant_module.load_state_dict(state, strict=False)
        # `missing` keys keep their freshly initialized values, which would
        # silently corrupt the restored weights — treat as fatal.
        if missing:
            raise RuntimeError(
                f"Resume: offloaded state for `{full_name}` is missing keys {missing}; "
                "the offload directory does not match this model/quant config."
            )
        if unexpected:
            log.warn("Resume: `%s` bundle had unexpected keys %s (ignored).", full_name, unexpected)
        # Deliberately skip post_init(): the normal quant-time finalize path
        # (gptq_processor.py submodule_finalize) never calls it either -- it's
        # reserved for gptqmodel_post_init() on a fully quantized model being
        # loaded for inference, not for a module that only needs to sit here
        # until save().
        #
        # This restored (packed) module must never be used for forward replay
        # to reconstruct a later layer's calibration input -- GPTQ requires
        # replaying with the pre-quantization checkpoint weights, not these
        # post-quantization ones. See `_resume_replay_layer`'s docstring.
        restored.append(full_name)

    return restored


def restore_completed_layer_class_only(looper, layer_prefix: str) -> List[str]:
    """Like `restore_completed_layer`, but skip reading/validating the packed
    tensor data (qweight/scales/qzeros/g_idx) entirely -- only swap the live
    module's class to `QuantLinear` (matching what's on disk), then move its
    parameters to the meta device.

    Safe because `save()`'s state-dict resolution reads any meta parameter
    straight from the offload directory by key name, regardless of whether
    resume ever materialized it -- this function never touches that bundle,
    it only defers reading it until save() time. That trades resume's own
    early corruption check for skipping the read/re-write on every layer
    strictly before the resume target; a corrupt bundle now surfaces as a
    save()-time failure instead of a resume-time one.

    Used when the resume target already has a valid activation cache (see
    `_resume_restore_only_layer`): these layers' forward outputs are never
    consumed by anything, so there is no reason to touch their data.
    """
    model = looper.gptq_model
    qcfg = model.quantize_config
    offload_root = qcfg.offload_to_disk_path

    restored: List[str] = []
    for full_name, _bundle in _offloaded_layer_modules(offload_root, layer_prefix):
        try:
            submodule = model.model.get_submodule(full_name)
        except AttributeError:
            log.warn("Resume: offloaded module `%s` not found in model; skipping.", full_name)
            continue

        if not isinstance(submodule, BaseQuantLinear):
            create_quant_module(
                name=full_name,
                linear_cls=model.qlinear_kernel,
                bits=getattr(qcfg, "runtime_bits", qcfg.bits),
                desc_act=qcfg.desc_act,
                dynamic=qcfg.dynamic,
                group_size=qcfg.group_size,
                module=model.model,
                submodule=submodule,
                sym=qcfg.sym,
                device=qcfg.device,
                lm_head_name=model.lm_head,
                pack_dtype=qcfg.pack_dtype,
                format=resolve_quant_format(qcfg.format, qcfg.method),
                register_buffers=True,
            )

        quant_module = model.model.get_submodule(full_name)
        # No load_state_dict: leave qweight/scales/qzeros/g_idx exactly as
        # create_quant_module allocated them, then evict to meta immediately
        # so save() knows to fetch the real values from the offload bundle.
        quant_module.to(device=torch.device("meta"))
        restored.append(full_name)

    return restored


__all__ = [
    "RESUME_ENV_FLAG",
    "RESUME_STATE_FILENAME",
    "read_resume_target",
    "restore_completed_layer",
    "restore_completed_layer_class_only",
    "resume_state_path",
    "write_resume_marker",
]
