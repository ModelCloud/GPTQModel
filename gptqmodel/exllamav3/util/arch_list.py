import os
import threading
from contextlib import contextmanager

import torch


# Since Torch 2.3.0 an annoying warning is printed every time the C++ extension is loaded, unless the
# TORCH_CUDA_ARCH_LIST variable is set. The default behavior from pytorch/torch/utils/cpp_extension.py
# is copied in the helper below, but without the warning.
#
# The override is process-global, so it must not outlive the EXL3 JIT build that
# needs it: every other JIT extension keys its cache on the environment value
# (see TorchOpsJitExtension._cuda_cache_fingerprint_payload). A process-wide
# mutation would make every later Marlin/ExLlamaV2/AWQ build miss its cache.

_ARCH_LIST_ENV_LOCK = threading.Lock()


def _build_arch_list() -> str | None:
    """Return the arch list for the visible devices, or ``None`` when unavailable."""

    if not torch.version.cuda:
        return None

    supported_sm = []
    for arch in torch.cuda.get_arch_list():
        if "sm_" not in arch:
            continue
        try:
            # Strip known NVIDIA suffixes: 'a' (accelerated) or 'f' (family)
            supported_sm.append(int(arch.split("_")[1].rstrip("af")))
        except (IndexError, ValueError):
            continue
    if not supported_sm:
        return None

    max_supported_sm = max((sm // 10, sm % 10) for sm in supported_sm)
    arch_pairs = set()
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        # Capability of the device may be higher than what's supported by the user's
        # NVCC, causing compilation error. User's NVCC is expected to match the one
        # used to build pytorch, so we use the maximum supported capability of pytorch
        # to clamp the capability.
        arch_pairs.add(min(max_supported_sm, capability))
    if not arch_pairs:
        return None

    # Sort numerically, not lexicographically: "8.9" must precede "12.0" and
    # the "+PTX" suffix belongs on the highest architecture.
    arch_list = [f"{major}.{minor}" for major, minor in sorted(arch_pairs)]
    arch_list[-1] += "+PTX"
    return ";".join(arch_list)


@contextmanager
def scoped_arch_list_env():
    """Temporarily set ``TORCH_CUDA_ARCH_LIST`` for one EXL3 JIT build.

    The previous value (or its absence) is restored on every exit path,
    including an exception. The restore is compare-and-set: if another thread
    installed a different value while the build ran, that value is left in
    place instead of being clobbered.
    """

    had_previous = "TORCH_CUDA_ARCH_LIST" in os.environ
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if previous:
        yield
        return

    arch_list = _build_arch_list()
    if arch_list is None:
        yield
        return

    installed = False
    with _ARCH_LIST_ENV_LOCK:
        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
            installed = True
    if not installed:
        # Another thread installed its own override while we were resolving
        # the visible capabilities; honor it rather than replacing it.
        yield
        return

    try:
        yield
    finally:
        with _ARCH_LIST_ENV_LOCK:
            # A concurrent caller may have replaced the value; leave its value
            # alone instead of clobbering it.
            if os.environ.get("TORCH_CUDA_ARCH_LIST") == arch_list:
                if had_previous:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = previous
                else:
                    os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
