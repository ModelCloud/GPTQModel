# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Bounded, document-separated dense-teacher capture for the P32 quantization job."""

from dataclasses import dataclass

import torch

from .qvq_rank8 import Rank8Calibration, _check_documents, _digest


@dataclass(frozen=True)
class Rank8Document:
    document_id: str
    inputs: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Rank8Capture:
    module_names: tuple[str, ...]
    train: tuple[Rank8Document, ...]
    heldout: tuple[Rank8Document, ...]
    rows_per_document: int = 128
    max_bytes: int = 512 * 1024 * 1024
    minimum_improvement: float = 0.01
    source_kind: str = "calibration"
    max_solver_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if self.source_kind != "calibration":
            raise ValueError("rank8 capture accepts calibration documents only")
        if not self.module_names or len(set(self.module_names)) != len(self.module_names):
            raise ValueError("rank8 capture requires unique module names")
        if any(
            type(v) is not int or v < 1
            for v in (self.rows_per_document, self.max_bytes, self.max_solver_bytes)
        ):
            raise ValueError("rank8 capture bounds must be positive integers")
        if not 0 <= self.minimum_improvement < 1:
            raise ValueError("minimum_improvement must be in [0,1)")
        ids = [tuple(d.document_id for d in split) for split in (self.train, self.heldout)]
        _check_documents(*ids)
        hashes = []
        for split, names in zip((self.train, self.heldout), ids):
            if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
                raise ValueError("each calibration document needs a unique nonempty ID")
            current = set()
            for document in split:
                inputs = document.inputs
                if not inputs or any(not isinstance(v, torch.Tensor) for v in inputs.values()):
                    raise ValueError("document inputs must be a nonempty tensor dictionary")
                tokens = inputs.get("input_ids")
                if (
                    tokens is None or tokens.ndim != 2 or tokens.shape[0] != 1
                    or tokens.dtype not in (torch.int32, torch.int64)
                ):
                    raise ValueError("capture requires one tokenized document per forward")
                mask = inputs.get("attention_mask")
                if mask is not None:
                    if mask.shape != tokens.shape or not torch.all((mask == 0) | (mask == 1)):
                        raise ValueError("capture requires a binary document attention mask")
                    tokens = tokens[mask.to(tokens.device).bool()]
                current.add(_digest({"input_ids": tokens.flatten().long()}, {}))
            hashes.append(current)
        if hashes[0] & hashes[1]:
            raise ValueError("train and held-out token documents overlap despite their IDs")


@torch.no_grad()
def capture_rank8_calibration(model, request, *, materialize_teacher=None):
    """Capture original inputs before any quantized replay; leave no hooks behind.

    Documents retain their boundaries and are never concatenated or sorted with
    ordinary quantizer calibration. Input tensors must already be on the model's
    execution device. A lazy teacher may provide ``materialize_teacher``; it is
    called at most once, before hooks or forwards, and must materialize the
    requested eval-mode Linear modules in place. Loading is intentionally kept
    outside CUDA graph capture and the callback cannot replace the model object.
    """
    if not isinstance(request, Rank8Capture):
        raise TypeError("rank8_capture must be Rank8Capture")
    request.__post_init__()  # Tensor dictionaries can change after construction.
    if model.training:
        raise ValueError("rank8 capture requires an eval-mode dense teacher")
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise ValueError("rank8 capture requires explicit teacher precision, without autocast")
    if materialize_teacher is not None and not callable(materialize_teacher):
        raise TypeError("materialize_teacher must be callable")
    modules = dict(model.named_modules())
    missing = [
        name
        for name in request.module_names
        if not isinstance(modules.get(name), torch.nn.Linear)
        or modules[name].weight.is_meta
    ]
    if missing and materialize_teacher is not None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("teacher materialization must happen before CUDA graph capture")
        returned = materialize_teacher(model)
        if returned is not None and returned is not model:
            raise ValueError("materialize_teacher must materialize the supplied model in place")
        if model.training:
            raise ValueError("materialize_teacher changed the teacher to training mode")
        modules = dict(model.named_modules())
    for name in request.module_names:
        child = modules.get(name)
        if not isinstance(child, torch.nn.Linear) or child.weight.is_meta or child.training:
            raise ValueError(f"rank8 capture requires a materialized eval Linear: {name}")
    samples = {name: [[], []] for name in request.module_names}
    teacher_hashes = {
        name: _digest(dict(modules[name].named_parameters()), {}) for name in request.module_names
    }
    handles = []
    used_bytes = 0
    split_index = 0
    current_document = None
    seen = set()

    def hook(name):
        def collect(module, args):
            nonlocal used_bytes
            if name in seen:
                raise ValueError(f"rank8 module executed more than once per document: {name}")
            seen.add(name)
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError(f"rank8 module requires positional activation input: {name}")
            x = args[0].detach()
            if x.ndim != 3 or x.shape[:2] != current_document.inputs['input_ids'].shape:
                raise ValueError(f"rank8 capture requires unmodified document token rows: {name}")
            mask = current_document.inputs.get("attention_mask")
            if mask is None:
                rows = torch.arange(x.shape[1], device=x.device)
            else:
                if mask.shape != x.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
                    raise ValueError("capture requires a binary document attention mask")
                rows = mask.flatten().to(x.device).nonzero().flatten()
            if rows.numel() == 0:
                raise ValueError("rank8 document has no unmasked tokens")
            # Evenly spaced token positions, independent of values or teacher error.
            take = min(request.rows_per_document, rows.numel())
            positions = torch.arange(take, device=x.device) * rows.numel() // take
            size = take * x.shape[-1] * x.element_size()
            # Reserve a second copy for final concatenation below.
            if 2 * (used_bytes + size) > request.max_bytes:
                raise ValueError("rank8 activation capture exceeds max_bytes")
            sample = x.reshape(-1, x.shape[-1]).index_select(0, rows[positions])
            if not torch.isfinite(sample).all():
                raise ValueError("rank8 capture contains non-finite activations")
            samples[name][split_index].append(sample.to(device="cpu", copy=True))
            used_bytes += size
        return collect

    try:
        for name in request.module_names:
            handles.append(modules[name].register_forward_pre_hook(hook(name)))
        for split_index, documents in enumerate((request.train, request.heldout)):
            for current_document in documents:
                seen.clear()
                model(**current_document.inputs)
                if seen != set(request.module_names):
                    raise ValueError("some requested rank8 modules did not execute")
        if any(
            _digest(dict(modules[name].named_parameters()), {}) != teacher_hashes[name]
            for name in request.module_names
        ):
            raise ValueError("rank8 teacher state changed during activation capture")
        return {
            name: Rank8Calibration(
                torch.cat(values[0]), torch.cat(values[1]),
                tuple(d.document_id for d in request.train),
                tuple(d.document_id for d in request.heldout),
                minimum_improvement=request.minimum_improvement,
                teacher_hash=teacher_hashes[name],
                max_solver_bytes=request.max_solver_bytes,
            )
            for name, values in samples.items()
        }
    finally:
        for handle in handles:
            handle.remove()
