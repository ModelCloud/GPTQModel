# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native sibling inputs share one masked copy per calibration batch.

``NativeProcessor`` captures one masked copy per sample for GPTAQ/FOEM. The
sibling modules fed by one source activation (q/k/v, gate/up) used to build
bitwise-identical copies, so the native cache held 3x/2x the required storage.
These tests pin the sharing, the unchanged ``batch_size=1`` alias path, the
per-batch tagging, and that consumers still read the right values without
mutating the shared entry.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
import torch.nn as nn
from torch.multiprocessing.reductions import StorageWeakRef

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY, NativeProcessor
from gptqmodel.quantization.config import CalibrationConfig, FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


_CLASSES = {"gptaq": GPTAQ, "foem": FOEM}


def _qcfg(kind: str, *, device: str = "cpu") -> QuantizeConfig:
    if kind == "gptaq":
        return QuantizeConfig(bits=4, group_size=8, gptaq=GPTAQConfig(device=device))
    return QuantizeConfig(bits=4, group_size=8, foem=FOEMConfig(alpha=0.25, device=device))


def _processor(kind: str, *, batch_size: int = 4) -> NativeProcessor:
    return NativeProcessor(
        tokenizer=None,
        qcfg=_qcfg(kind),
        calibration=CalibrationConfig(dataset=None, batch_size=batch_size),
    )


def _named(module: nn.Module, name: str) -> NamedModule:
    return NamedModule(module, name=name, full_name=f"model.layers.0.{name}", layer_index=0)


def _bind_batch(processor: NativeProcessor, mask, batch_index: int) -> None:
    processor._mask_tls = threading.local()
    processor._mask_tls.value = mask
    processor._set_current_batch_index(batch_index)


def _capture(processor: NativeProcessor, modules, x, mask, *, batch_index: int = 0):
    """Register and fire one native capture pass over sibling modules."""

    named = {}
    for name, module in modules.items():
        named[name] = _named(module, name)
        processor.preprocess(named[name])
    _bind_batch(processor, mask, batch_index)
    for name, module in modules.items():
        processor.pre_process_fwd_hook(name)(module, (x,), None)
    return named


def _finish(processor: NativeProcessor, named) -> None:
    for module in named.values():
        processor.process(module)


def _entries(named: NamedModule) -> list:
    return [tensor for _batch_index, tensor in named.state[NATIVE_INPUTS_STATE_KEY]]


def _sibling_modules(width: int = 8) -> dict:
    return {
        "q_proj": nn.Linear(width, width, bias=False),
        "k_proj": nn.Linear(width, width, bias=False),
        "v_proj": nn.Linear(width, width, bias=False),
    }


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_siblings_share_masked_storage_at_batch_gt_one(kind):
    torch.manual_seed(0)
    x = torch.randn(4, 5, 8)
    mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, False, False],
            [False, False, False, False, False],
            [True, False, True, False, True],
        ]
    )
    modules = _sibling_modules()
    processor = _processor(kind)

    named = _capture(processor, modules, x, mask)
    _finish(processor, named)

    entries = {name: _entries(named[name]) for name in modules}
    kept_samples = int(mask.any(dim=1).sum())
    # Same number of add_batch entries as kept samples for every sibling.
    assert all(len(values) == kept_samples for values in entries.values())

    for position, sample_index in enumerate(mask.any(dim=1).nonzero().flatten().tolist()):
        expected = x[sample_index][mask[sample_index]]
        reference = entries["q_proj"][position]
        for name in modules:
            entry = entries[name][position]
            # q/k/v all reuse the one masked copy ...
            assert entry.untyped_storage().data_ptr() == reference.untyped_storage().data_ptr()
            assert entry.data_ptr() == reference.data_ptr()
            # ... which is still a masked copy, not an alias of the full batch.
            assert entry.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
            torch.testing.assert_close(entry, expected)


def test_batch_size_one_without_mask_still_aliases_input():
    """The historical bs=1 path (no attention mask) must not change."""

    torch.manual_seed(1)
    x = torch.randn(1, 4, 8)
    modules = {"gate_proj": nn.Linear(8, 8, bias=False)}
    processor = _processor("gptaq", batch_size=1)

    named = _capture(processor, modules, x, None)
    _finish(processor, named)

    entries = _entries(named["gate_proj"])
    assert len(entries) == 1
    entry = entries[0]
    # One entry per batch, aliasing the shared source storage (no copy).
    assert entry.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert entry.data_ptr() == x[0].data_ptr()
    torch.testing.assert_close(entry, x)


def test_gate_up_siblings_share_masked_storage_at_batch_gt_one():
    torch.manual_seed(7)
    x = torch.randn(3, 4, 8)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, False, True, False],
        ]
    )
    modules = {
        "gate_proj": nn.Linear(8, 8, bias=False),
        "up_proj": nn.Linear(8, 8, bias=False),
    }
    processor = _processor("gptaq")

    named = _capture(processor, modules, x, mask)
    _finish(processor, named)

    gate = _entries(named["gate_proj"])
    up = _entries(named["up_proj"])
    assert len(gate) == len(up) == int(mask.any(dim=1).sum())
    for sample_index, (gate_entry, up_entry) in enumerate(zip(gate, up)):
        assert gate_entry.untyped_storage().data_ptr() == up_entry.untyped_storage().data_ptr()
        torch.testing.assert_close(gate_entry, x[sample_index][mask[sample_index]])
        torch.testing.assert_close(up_entry, gate_entry)


def test_distinct_sources_and_batches_do_not_share():
    """Only the same source tensor in the same batch may share an entry."""

    torch.manual_seed(2)
    first = torch.randn(2, 4, 8)
    second = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, True], [True, False, True, False]])
    modules = _sibling_modules()
    processor = _processor("gptaq")

    named_first = _capture(processor, modules, first, mask, batch_index=0)
    _finish(processor, named_first)

    # The same storage replayed as a *different* calibration batch must not
    # reuse the previous batch's entries; the per-batch tagging stays exact.
    named_second = _capture(processor, modules, first, mask, batch_index=1)
    _finish(processor, named_second)

    named_other = _capture(processor, modules, second, mask, batch_index=0)
    _finish(processor, named_other)

    first_entries = _entries(named_first["q_proj"])
    second_entries = _entries(named_second["q_proj"])
    other_entries = _entries(named_other["q_proj"])

    assert [batch for batch, _ in named_second["q_proj"].state[NATIVE_INPUTS_STATE_KEY]] == [1, 1]
    for before, after in zip(first_entries, second_entries):
        assert before.untyped_storage().data_ptr() != after.untyped_storage().data_ptr()
    for first_entry, other_entry in zip(first_entries, other_entries):
        assert first_entry.untyped_storage().data_ptr() != other_entry.untyped_storage().data_ptr()
    torch.testing.assert_close(other_entries[0], second[0][mask[0]])


def test_new_forward_pass_does_not_reuse_cached_entries():
    """The same batch index, mask and source object in a later layer are new."""

    torch.manual_seed(5)
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    modules = _sibling_modules()
    processor = _processor("gptaq")

    first = _capture(processor, modules, x, mask, batch_index=0)
    _finish(processor, first)
    second = _capture(processor, modules, x, mask, batch_index=0)
    _finish(processor, second)

    first_entry = _entries(first["q_proj"])[0]
    second_entry = _entries(second["q_proj"])[0]
    # ``process()`` ends the capture pass, so a recycled source address in the
    # next layer must not hand back the previous layer's cached copy.
    assert first_entry.untyped_storage().data_ptr() != second_entry.untyped_storage().data_ptr()
    torch.testing.assert_close(first_entry, second_entry)


def test_batches_in_one_layer_do_not_share_entries():
    """Two calibration batches of the same layer keep separate entries."""

    torch.manual_seed(6)
    batches = [torch.randn(2, 4, 8), torch.randn(2, 4, 8)]
    mask = torch.ones(2, 4, dtype=torch.bool)
    modules = _sibling_modules()
    processor = _processor("gptaq")

    named = {}
    for name, module in modules.items():
        named[name] = _named(module, name)
        processor.preprocess(named[name])
    for batch_index, x in enumerate(batches):
        _bind_batch(processor, mask, batch_index)
        for name, module in modules.items():
            processor.pre_process_fwd_hook(name)(module, (x,), None)
    _finish(processor, named)

    # Two kept samples per batch, tagged with their own batch index.
    assert [batch for batch, _ in named["q_proj"].state[NATIVE_INPUTS_STATE_KEY]] == [0, 0, 1, 1]
    q_entries = _entries(named["q_proj"])
    k_entries = _entries(named["k_proj"])
    assert q_entries[0].untyped_storage().data_ptr() == k_entries[0].untyped_storage().data_ptr()
    assert q_entries[2].untyped_storage().data_ptr() == k_entries[2].untyped_storage().data_ptr()
    assert q_entries[0].untyped_storage().data_ptr() != q_entries[2].untyped_storage().data_ptr()
    torch.testing.assert_close(q_entries[0], batches[0][0][mask[0]])
    torch.testing.assert_close(q_entries[2], batches[1][0][mask[0]])


@pytest.mark.parametrize("kind", ["gptaq", "foem"])
def test_consumers_read_shared_entries_without_mutating(kind):
    torch.manual_seed(3)
    x = torch.randn(3, 4, 8)
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [False, True, False, True],
        ]
    )
    modules = _sibling_modules()
    processor = _processor(kind)

    named = _capture(processor, modules, x, mask)
    _finish(processor, named)

    shared = {name: _entries(named[name]) for name in modules}
    before = {name: [entry.clone() for entry in values] for name, values in shared.items()}
    samples = [
        x[sample_index : sample_index + 1][:, mask[sample_index]]
        for sample_index in range(x.shape[0])
        if mask[sample_index].any()
    ]

    def run(name, entries):
        module = _named(modules[name], name)
        module.state[NATIVE_INPUTS_STATE_KEY] = [(0, entry) for entry in entries]
        task = _CLASSES[kind](module, _qcfg(kind))
        for sample in samples:
            task.add_batch(sample, None)
        return task

    # Both siblings consume the *same* storage, exactly as they would in the
    # quantization loop. Independent clones stand in for the old per-module
    # copies, so the two runs are the bitwise-identical reference pair.
    shared_q = run("q_proj", shared["q_proj"])
    shared_k = run("k_proj", shared["k_proj"])
    cloned_q = run("q_proj", [entry.clone() for entry in shared["q_proj"]])
    cloned_k = run("k_proj", [entry.clone() for entry in shared["k_proj"]])

    for shared_task, cloned_task in ((shared_q, cloned_q), (shared_k, cloned_k)):
        assert torch.equal(shared_task.H, cloned_task.H)
        if shared_task.dXXT is not None:
            assert torch.equal(shared_task.dXXT, cloned_task.dXXT)

    # Reading the entries through the quantizer must not alter them.
    for name in modules:
        for entry, expected in zip(shared[name], before[name]):
            assert torch.equal(entry, expected)



def _capture_on_worker(processor, modules, mask, worker, *, batch_index: int = 0):
    """Fire one capture pass on a long-lived worker thread, like the forward pool.

    Returns weak references to the source activation and keep-mask storages;
    the caller drops every strong reference it holds.
    """

    named = {}
    for name, module in modules.items():
        named[name] = _named(module, name)
        processor.preprocess(named[name])

    def forward():
        x = torch.randn(2, 4, 8)
        _bind_batch(processor, mask, batch_index)
        for name, module in modules.items():
            processor.pre_process_fwd_hook(name)(module, (x,), None)
        # ModuleLooper clears the active keep-mask after the forward pass.
        processor._mask_tls.value = None
        return StorageWeakRef(x.untyped_storage())

    source_ref = worker.submit(forward).result()
    return named, source_ref


@pytest.mark.parametrize("release", ["process", "finalize"])
def test_share_cache_does_not_pin_source_after_layer(release):
    """The sibling-share cache must not keep the last batch's activation alive (#4463).

    Capture hooks run on long-lived forward workers while ``process()`` and
    ``finalize()`` run elsewhere. A per-thread cache that is only invalidated
    lazily kept ``(source, mask, entries)`` alive on every worker until its
    next capture, i.e. through the rest of the layer and after ``quantize()``.
    """

    torch.manual_seed(8)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    mask_ref = StorageWeakRef(mask.untyped_storage())
    modules = _sibling_modules()
    processor = _processor("gptaq")

    with ThreadPoolExecutor(max_workers=1) as worker:
        named, source_ref = _capture_on_worker(processor, modules, mask, worker)
        del mask
        if release == "process":
            # ``process()`` runs on another thread than the capture worker.
            _finish(processor, named)
        else:
            processor.finalize(model=None)

        # The product releases the cache when the layer finishes; assert that
        # explicitly rather than inferring it from storage liveness alone.
        assert processor._native_share_caches == {}

        # CPython 3.14t keeps the last returned frame's data stack cached on the
        # thread state, so the capture frame's locals stay reachable until that
        # thread runs another Python call. A no-op call cannot release a real
        # thread-local cache (that would live in the thread's local dict), so it
        # only drops the interpreter's frame cache and keeps the leak assertion
        # meaningful.
        worker.submit(lambda: None).result()

        # The worker thread is still alive, so a thread-local cache would
        # still be reachable here.
        assert source_ref.expired()
        assert mask_ref.expired()

    if release == "process":
        # The consumer-owned per-sample copies are unaffected.
        entries = _entries(named["q_proj"])
        assert len(entries) == 2
        assert entries[0].untyped_storage().data_ptr() == _entries(named["k_proj"])[0].untyped_storage().data_ptr()
