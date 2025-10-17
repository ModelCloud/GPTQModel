# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import time
from contextlib import contextmanager
from enum import Enum
from typing import Callable, List, Union

import torch
from packaging import version
from torch.cpu import StreamContext

from ..utils.logger import setup_logger
from ..utils.safe import GC
from . import gte_python_3_13_3, gte_python_3_14, has_gil_disabled, log_gil_requirements_for


# pytorch 2.6.0 fixes many compilation errors
TORCH_HAS_COMPILE = version.parse(torch.__version__).release >= version.Version('2.6').release
TORCH_GTE_28 = version.parse(torch.__version__).release >= version.Version('2.8').release
TORCH_GTE_210 = version.parse(torch.__version__).release >= version.Version('2.10').release

TORCH_HAS_FUSED_OPS = version.parse(torch.__version__).release >= version.Version('2.8').release

HAS_CUDA = False
HAS_XPU = False
HAS_MPS = False
HAS_MLX = False
HAS_NPU = False

CPU = torch.device("cpu")
META = torch.device("meta")

class BalanceStrategy(str, Enum):
    MEMORY = "memory", # make vram more spread out
    GPU = "gpu" # vram is less balanced (gpu0) but gpu0 is also used for quantization

DEFAULT_BALANCE_STRATEGY = BalanceStrategy.GPU

# TODO FIX ME...this should be removed
STREAM = None # cache

log = setup_logger()


def timed_gc_collect() -> int:
    """Run ``gc.collect`` and log the elapsed time along with reclaimed object count."""
    start = time.perf_counter()

    # Python 3.14 removed gen1 so there is only gen0 and gen2
    collected = GC.collect()

    duration = time.perf_counter() - start
    log.info(f"gc.collect() reclaimed {collected} objects in {duration:.3f}s")
    return collected

# reset dynamo cache on each model load since during ci loop model inference may exhuast cache
try:
    torch._dynamo.reset()
    # Increase the dynamo cache size limit, default of 8 is too low
    if torch._dynamo.config.cache_size_limit < 128:
        torch._dynamo.config.cache_size_limit = 128
except BaseException:
    # triton built from source maybe incompatible with _dynamo private api
    pass

if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
    HAS_CUDA = True

if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
    HAS_XPU = True

if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
    HAS_MPS = True

if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
    HAS_NPU = True


# mlx check
try:
    import mlx.core.metal
    HAS_MLX = True
except BaseException:
    pass

BACKENDS_HAS_FP32_PRECISION = hasattr(torch.backends, "fp32_precision")


def _set_tf32_state(enabled: bool) -> None:
    if BACKENDS_HAS_FP32_PRECISION:
        mode = "tf32" if enabled else "ieee"
        torch.backends.fp32_precision = mode
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.fp32_precision = mode
        torch.backends.cudnn.conv.fp32_precision = mode
        torch.backends.cudnn.rnn.fp32_precision = mode
        return

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _snapshot_tf32_state():
    if BACKENDS_HAS_FP32_PRECISION:
        return (
            torch.backends.fp32_precision,
            torch.backends.cuda.matmul.fp32_precision,
            torch.backends.cudnn.fp32_precision,
            torch.backends.cudnn.conv.fp32_precision,
            torch.backends.cudnn.rnn.fp32_precision,
        )

    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def _restore_tf32_state(state) -> None:
    if BACKENDS_HAS_FP32_PRECISION:
        torch.backends.fp32_precision = state[0]
        torch.backends.cuda.matmul.fp32_precision = state[1]
        torch.backends.cudnn.fp32_precision = state[2]
        torch.backends.cudnn.conv.fp32_precision = state[3]
        torch.backends.cudnn.rnn.fp32_precision = state[4]
        return

    torch.backends.cuda.matmul.allow_tf32 = state[0]
    torch.backends.cudnn.allow_tf32 = state[1]


def torch_compile(module: Union[torch.nn.Module, Callable], backend:str ="inductor", mode: str = None, fullgraph=False):
    # requires torch >2.8 for proper torch.compile + Python 3.13.3t (freethreading)
    if has_gil_disabled() and not gte_python_3_13_3():
        log_gil_requirements_for("Torch Compile")
        return module

    if gte_python_3_14() and not TORCH_GTE_210:
        log_gil_requirements_for("Torch Compile")
        return module

    if not TORCH_HAS_COMPILE:
        return module
    if HAS_MPS and not TORCH_GTE_28:
        if not torch._dynamo.config.suppress_errors:
            log.warn("To use compile() with MPS, you need to have torch version >= 2.8.0, "
                     "please upgrade it by `pip install -U torch torchaudio torchvision`")
            torch._dynamo.config.suppress_errors = True
        return module
    try:
        return torch.compile(module, backend=backend, mode=mode, fullgraph=fullgraph)
    except BaseException as e:
        log.warn.once(f"Failed to compile `{module}`, {e}")
        return module

def torch_new_stream():
    global STREAM
    if STREAM is None:
        return STREAM

    if HAS_CUDA:
        STREAM = torch.cuda.Stream()
        return STREAM
    if HAS_XPU:
        STREAM = torch.xpu.Stream()
        return STREAM
    return None

def torch_new_stream_ctx():
    if HAS_CUDA:
        return torch.cuda.stream(torch_new_stream())
    if HAS_XPU:
        return torch.xpu.Stream(torch_new_stream())
    return contextlib.nullcontext()

def torch_sync(device: torch.device = None):
    """Synchronize accelerator queues.

    When no device is provided we synchronize every detected accelerator index so
    replication work staged on multiple GPUs/NPUs completes before issuing more
    kernels.
    """

    if device is None:
        synchronized_any = False

        if HAS_CUDA:
            dev_count = torch.cuda.device_count()
            if dev_count:
                synchronized_any = True
                for idx in range(dev_count):
                    torch.cuda.synchronize(idx)

        if HAS_XPU and hasattr(torch.xpu, "device_count"):
            dev_count = torch.xpu.device_count()
            if dev_count:
                synchronized_any = True
                for idx in range(dev_count):
                    torch.xpu.synchronize(idx)

        if HAS_MPS:
            synchronized_any = True
            torch.mps.synchronize()

        if HAS_NPU and hasattr(torch.npu, "device_count"):
            dev_count = torch.npu.device_count()
            if dev_count:
                synchronized_any = True
                for idx in range(dev_count):
                    torch.npu.synchronize(idx)

        if not synchronized_any:
            torch.cpu.synchronize()
        return

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize(device=device)
    elif device.type == "cpu":
        torch.cpu.synchronize()

def torch_empty_cache(device: torch.device = None, gc: bool = True):
    if gc:
        timed_gc_collect()

    # check all backends
    if device is None:
        if HAS_CUDA:
            # torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if HAS_XPU:
            # torch.xpu.synchronize()
            torch.xpu.empty_cache()
        if HAS_MPS:
            torch.mps.empty_cache()
        if HAS_MLX:
            mlx.core.clear_cache()
        return

    # if device passed, only execute for device backend
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

        # mlx is detached from pytorch
        if HAS_MLX:
            mlx.core.clear_cache()

def auto_select_torch_device(index: int = 0):
    assert index >= 0, f"device index should be a positive integer: actual = `{index}`"

    if HAS_CUDA:
        # defensive check
        if index > 0 and torch.cuda.device_count() <= index :
            index = 0
        device = torch.device(f"cuda:{index}")
    elif HAS_XPU:
        # defensive check
        if index > 0 and torch.xpu.device_count() <= index:
            index = 0
        device = torch.device(f"xpu:{index}")
    elif HAS_MPS:
        device = torch.device("mps") # mps has no index
    else:
        device = CPU # cpu has no index

    return device

# some device types can have multiple gpus cuda/rocm + xpu
def torch_devices() -> List[torch.device]:
    if HAS_CUDA:
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    elif HAS_XPU:
        return [torch.device(f"xpu:{i}") for i in range(torch.xpu.device_count())]
    elif HAS_MPS:
        return [torch.device("mps")]
    else:
        return [CPU]

ALL_DEVICES = torch_devices()

if HAS_CUDA:
    ALL_STREAMS = [torch.cuda.Stream(device=device) for device in ALL_DEVICES]
elif HAS_XPU:
    ALL_STREAMS = [torch.xpu.Stream(device=device) for device in ALL_DEVICES]
else:
    ALL_STREAMS = [contextlib.nullcontext()]

DEVICE_0 = auto_select_torch_device(index=0)
# device_1 may be same as device_0 if there is only 1 visible/active device
DEVICE_1 = auto_select_torch_device(index=1)

DEVICE_0_STREAM = ALL_STREAMS[0]

NEXT_DEVICE_INDEX = 0

# def device_next_reset():
#     global NEXT_DEVICE_INDEX
#     NEXT_DEVICE_INDEX = 0
#
# def device_next(balance_strategy: BalanceStrategy = DEFAULT_BALANCE_STRATEGY) -> torch.device:
#     global NEXT_DEVICE_INDEX
#
#     if len(ALL_DEVICES) <= 1:
#         return ALL_DEVICES[0]
#
#     device = ALL_DEVICES[NEXT_DEVICE_INDEX]
#     if NEXT_DEVICE_INDEX < len(ALL_DEVICES) - 1:
#         NEXT_DEVICE_INDEX += 1
#     else:
#         if balance_strategy == BalanceStrategy.MEMORY:
#             NEXT_DEVICE_INDEX = 1
#         else:
#             NEXT_DEVICE_INDEX = 0
#
#     return device

def torch_streamCtx(stream: Union[torch.cuda.Stream, torch.xpu.Stream]) -> StreamContext:
    return torch.cuda.stream(stream) if HAS_CUDA else torch.xpu.stream(stream)


@contextmanager
def tf32_high_precision_guard():
    if not HAS_CUDA:
        yield
        return

    previous_state = _snapshot_tf32_state()
    _set_tf32_state(False)
    try:
        yield
    finally:
        _restore_tf32_state(previous_state)


@contextmanager
def tf32_disable_guard():
    if not HAS_CUDA:
        yield
        return

    previous_state = _snapshot_tf32_state()
    _set_tf32_state(False)
    try:
        yield
    finally:
        _restore_tf32_state(previous_state)


@contextmanager
def tf32_enable_guard():
    if not HAS_CUDA:
        yield
        return

    previous_state = _snapshot_tf32_state()
    _set_tf32_state(True)
    try:
        yield
    finally:
        _restore_tf32_state(previous_state)
