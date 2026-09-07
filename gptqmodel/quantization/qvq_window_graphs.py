# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Request-owned CUDA graphs for the existing unified P32 window operator."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from threading import Lock
from weakref import WeakKeyDictionary, ref

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from .qvq_rank8 import RANK8_BUFFERS, P32WindowConfig, prepare_rank8

_MODES = ("fast", "balanced", "quality")
_POLICY = ("_p32_window_config", "_p32_rank8_enabled", "_p32_rank8_versions")
_MISSING = object()
_OWNERS = WeakKeyDictionary()
_OWNER_LOCK = Lock()


def _resident_tensors(values):
    """Own tensors in eager caches whose pointers were embedded in a graph."""
    tensors, seen = [], set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))

    visit(values)
    return tensors


class P32WindowGraphs:
    """Capture one stateless model call per shape and quality mode.

    Inputs are tensor keyword arguments; fixed keyword arguments (for example
    ``use_cache=False``) belong to capture. Outputs are tensor pytrees. Mutable
    KV state must be explicit tensor inputs, not hidden model/Python state.
    Capture is transactional and restores the original eager module policies.
    Replay never changes Python policies or branches on correction inside a
    graph. Returned tensors own their storage, so later requests cannot replace
    an earlier result. One owner must have exclusive use of the model while
    capturing/replaying. Parameter/buffer mutation invalidates captured graphs;
    untracked writes through .data or external pointers are unsupported. The
    ``max_graphs`` limit bounds retained executable/pool resources; least
    recently used graphs are synchronized and retired before a new capture is
    installed.
    """

    def __init__(self, model, *, max_graphs=8):
        from ..nn_modules.qlinear.qvq import QVQLinear

        if type(max_graphs) is not int or max_graphs < 1:
            raise ValueError("max_graphs must be a positive integer")
        self.model = model
        self.max_graphs = max_graphs
        self.layers = {
            name: child for name, child in model.named_modules()
            if isinstance(child, QVQLinear) and child.v2b2_p32
        }
        if not self.layers:
            raise ValueError("model has no P32 window modules")
        self._lock = Lock()
        # Ordered by most-recent capture/replay.  Graphs retain CUDA pools and
        # native payloads, so a request owner must have a finite residency
        # bound instead of accumulating one executable per shape forever.
        self._graphs = OrderedDict()
        self._event = None
        self._device = None
        self._closed = False
        with _OWNER_LOCK:
            owner = _OWNERS.get(model)
            if owner is not None and owner() is not None:
                raise RuntimeError("model already has a window graph owner; close it first")
            _OWNERS[model] = ref(self)

    def __enter__(self):
        """Return this request-owned graph manager as a context manager.

        Graphs retain CUDA allocations and native handles until they are
        explicitly closed.  The context-manager form makes that lifetime
        boundary unambiguous for callers that capture short-lived request
        graphs while preserving the existing explicit ``close`` API.
        """
        if self._closed:
            raise RuntimeError("window graph owner is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @contextmanager
    def _exclusive(self, *, allow_closed=False):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("a window graph request or capture is already active")
        try:
            if self._closed and not allow_closed:
                raise RuntimeError("window graph owner is closed")
            yield
        finally:
            self._lock.release()

    def _state(self, mode):
        if any(self.model.get_submodule(name) is not child for name, child in self.layers.items()):
            raise RuntimeError("model structure changed; create a new window graph owner")
        modules = tuple(
            (name, id(child), child.training)
            for name, child in self.model.named_modules()
        )
        if any(training for _, _, training in modules):
            raise ValueError("window graphs require the entire model in eval mode")
        values = []
        for name, value in (*self.model.named_parameters(), *self.model.named_buffers()):
            if mode == "fast" and name.rsplit(".", 1)[-1] in RANK8_BUFFERS:
                continue
            # Inference tensors cannot track mutation. Refuse them rather than
            # replay stale device pointers under an unverifiable state guard.
            if value.is_inference():
                raise ValueError("graph model tensors must support mutation version tracking")
            values.append((name, id(value), value._version, value.data_ptr(),
                           tuple(value.shape), tuple(value.stride()), value.dtype, value.device))
        semantics = tuple(
            (name, child.bits, child.codebook_version, child.input_hadamard,
             child.output_hadamard, id(child.activation), child.in_features, child.out_features,
             id(getattr(child, "_gptqmodel_qvq_grouped_runtime", None)),
             id(getattr(child, "_qvq_grouped_p32_delegate", None)))
            for name, child in self.layers.items()
        )
        return modules, tuple(values), semantics

    def _idle_groups(self):
        groups = {
            id(group): group for child in self.layers.values()
            if (group := getattr(child, "_gptqmodel_qvq_grouped_runtime", None)) is not None
        }
        if any(group._outputs is not None for group in groups.values()):
            raise RuntimeError("window policy cannot change during a sibling projection cycle")
        return groups.values()

    def capture(self, key, inputs, *, configs=None, static_kwargs=None, warmup=3):
        """Atomically install fast/balanced/quality graphs for one input shape.

        ``configs[mode][module_name]`` supplies externally tuned geometry and
        implementation. Recovery enablement is always resolved from that mode
        and validated quantizer metadata. Missing entries use the reference
        implementation. No fitting or latency selection occurs here.
        """
        with self._exclusive(), torch.no_grad():
            if not inputs or any(not isinstance(x, torch.Tensor) for x in inputs.values()):
                raise ValueError("graph inputs must be a nonempty mapping of tensors")
            device = next(iter(inputs.values())).device
            if device.type != "cuda" or any(x.device != device for x in inputs.values()):
                raise ValueError("window graphs require inputs on one CUDA device")
            if self._device is not None and device != self._device:
                raise ValueError("one graph owner cannot span CUDA devices")
            hash(key)
            if type(warmup) is not int or warmup < 1:
                raise ValueError("at least one warmup call is required")
            static_kwargs = dict(static_kwargs or {})
            if inputs.keys() & static_kwargs.keys():
                raise ValueError("static and tensor keyword arguments must be disjoint")
            if any(type(v) not in (bool, int, float, str, type(None)) for v in static_kwargs.values()):
                raise ValueError("static keyword values must be immutable scalars; tensors belong in inputs")
            configs = configs or {}
            if set(configs) - set(_MODES):
                raise ValueError("unknown quality mode")
            for choices in configs.values():
                if set(choices) - self.layers.keys():
                    raise ValueError("kernel config names must identify P32 modules")
                if any(not isinstance(c, P32WindowConfig) for c in choices.values()):
                    raise TypeError("kernel configs must be P32WindowConfig")
            groups = tuple(self._idle_groups())
            for mode in _MODES:
                self._state(mode)
            original = {
                name: tuple(getattr(child, attr, _MISSING) for attr in _POLICY)
                for name, child in self.layers.items()
            }
            pending = {}
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("prepare window graphs outside CUDA capture")
                from ..utils.qvq_cuda import prewarm_qvq_cuda

                if not prewarm_qvq_cuda():
                    raise RuntimeError("QVQ CUDA extension is unavailable for graph capture")
                if self._event is not None:
                    self._event.synchronize()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                try:
                    with torch.cuda.stream(stream):
                        for mode in _MODES:
                            for name, child in self.layers.items():
                                config = configs.get(mode, {}).get(name, P32WindowConfig())
                                prepare_rank8(child, replace(config, recovery_mode="auto", quality_mode=mode))
                                child._prepare_cuda_graph_auxiliary_caches()
                            buffers = {name: x.detach().clone() for name, x in inputs.items()}
                            for _ in range(warmup):
                                self.model(**buffers, **static_kwargs)
                            self._idle_groups()
                            stream.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                result = self.model(**buffers, **static_kwargs)
                            outputs, spec = tree_flatten(result)
                            if not outputs or any(not isinstance(x, torch.Tensor) for x in outputs):
                                raise ValueError("graph outputs must be a tensor pytree")
                            self._idle_groups()
                            keepalive = _resident_tensors(
                                [child.__dict__ for child in self.model.modules()]
                                + [group.__dict__ for group in groups]
                            )
                            pending[mode] = (graph, buffers, outputs, spec, keepalive)
                finally:
                    stream.synchronize()
                    for group in groups:
                        group.invalidate()
                    for name, child in self.layers.items():
                        for attr, value in zip(_POLICY, original[name], strict=True):
                            if value is _MISSING:
                                child.__dict__.pop(attr, None)
                            else:
                                setattr(child, attr, value)
                states = {mode: self._state(mode) for mode in _MODES}
                self._retire_for_insert(key)
                self._graphs[key] = (pending, states)
                self._device = device

    def replay(self, key, mode, **inputs):
        """Execute one complete request with a fixed captured correction mode."""
        with self._exclusive(), torch.no_grad():
            if mode not in _MODES:
                raise ValueError("unknown quality mode")
            graphs, states = self._graphs[key]
            # Replay is a use of the executable and therefore refreshes its
            # residency without changing any captured policy or buffers.
            self._graphs.move_to_end(key)
            self._idle_groups()
            if self._state(mode) != states[mode]:
                raise RuntimeError("model state changed; invalidate and recapture window graphs")
            graph, buffers, outputs, spec, _keepalive = graphs[mode]
            if inputs.keys() != buffers.keys() or any(
                not isinstance(inputs[name], torch.Tensor)
                or inputs[name].shape != target.shape
                or inputs[name].dtype != target.dtype
                or inputs[name].device != target.device
                for name, target in buffers.items()
            ):
                raise ValueError("request tensors do not match the captured input signature")
            with torch.cuda.device(self._device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("window graph requests cannot nest inside CUDA capture")
                stream = torch.cuda.current_stream(self._device)
                if self._event is not None:
                    stream.wait_event(self._event)
                for name, target in buffers.items():
                    target.copy_(inputs[name])
                graph.replay()
                result = tree_unflatten([output.clone() for output in outputs], spec)
                self._event = torch.cuda.Event()
                self._event.record(stream)
                return result

    def invalidate(self):
        """Retire graphs after all submitted requests finish; weights are retained."""
        with self._exclusive():
            if self._event is not None:
                self._event.synchronize()
            self._graphs.clear()
            self._event = None

    def close(self):
        """Finish queued requests and release exclusive graph ownership."""
        with self._exclusive(allow_closed=True):
            if self._closed:
                return
            if self._event is not None:
                self._event.synchronize()
            self._graphs.clear()
            self._event = None
            self._closed = True
            with _OWNER_LOCK:
                # Weak-key cleanup or an earlier owner teardown may already
                # have removed this entry.  Cleanup must remain idempotent so
                # an exception path cannot strand captured graph resources.
                owner = _OWNERS.get(self.model)
                if owner is not None and owner() is self:
                    del _OWNERS[self.model]

    def _retire_for_insert(self, key):
        """Synchronize and retire least-recent graphs before installing one."""
        if key in self._graphs:
            self._graphs.pop(key)
        while len(self._graphs) >= self.max_graphs:
            # A replay may have submitted work on the current stream.  The
            # event records the latest request and must complete before its
            # graph/pool references are released.
            if self._event is not None:
                self._event.synchronize()
                self._event = None
            self._graphs.popitem(last=False)
