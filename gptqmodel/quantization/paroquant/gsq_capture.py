"""Transactional module-input capture for paired ParoQuant reconstruction."""

from contextlib import contextmanager
from threading import Lock

import torch


@contextmanager
def capture_module_inputs(modules, batch_index):
    """Yield captures keyed by module, then (batch, invocation) in replay order.

    The caller supplies the exact selected module mapping and replay batch ID.
    Captures commit only when the context exits successfully. Hooks and partial
    data are removed on failure; existing hooks remain owned by their callers.
    This helper does not itself enable paired GSQ in the processor.
    """
    captures = {name: [] for name in modules}
    counts = {}
    handles = []
    lock = Lock()
    succeeded = False

    def hook(name):
        def record(module, args):
            index = batch_index()
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("Paired GSQ capture requires an explicit nonnegative batch index")
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError("Paired GSQ capture requires a tensor first argument")
            tensor = args[0]
            if tensor.ndim < 2 or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
                raise ValueError("Paired GSQ capture requires finite floating module activations")
            copied = tensor.detach().to(device='cpu', copy=True)
            with lock:
                key = (name, index)
                invocation = counts.get(key, 0)
                counts[key] = invocation + 1
                captures[name].append((index, invocation, copied))
        return record

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(hook(name)))
        yield captures
        succeeded = True
    finally:
        for handle in handles:
            handle.remove()
        if not succeeded:
            for values in captures.values():
                values.clear()


def align_module_inputs(clean, noisy):
    """Pair captures by explicit batch/call IDs, never by truncating zip."""
    def indexed(records):
        result = {}
        for batch, invocation, tensor in records:
            key = (batch, invocation)
            if key in result:
                raise ValueError("Paired GSQ capture has duplicate batch/invocation IDs")
            result[key] = tensor
        return result

    clean_map, noisy_map = indexed(clean), indexed(noisy)
    if clean_map.keys() != noisy_map.keys():
        raise ValueError("Paired GSQ clean/noisy invocation IDs do not match")
    pairs = []
    for key in sorted(clean_map):
        teacher, runtime = clean_map[key], noisy_map[key]
        if teacher.shape != runtime.shape:
            raise ValueError("Paired GSQ clean/noisy activation shapes do not match")
        pairs.append((key, teacher, runtime))
    return pairs
