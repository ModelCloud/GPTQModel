# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc as py_gc
from typing import Callable, Union

import torch
from packaging.version import Version

from ..utils.logger import setup_logger

HAS_CUDA = False
HAS_XPU = False
HAS_MPS = False
HAS_MLX = False

STREAM = None # cache

log = setup_logger()

# reset dynamo cache on each model load since during ci loop model inference may exhuast cache
torch._dynamo.reset()

# Increase the dynamo cache size limit, default of 8 is too low
if torch._dynamo.config.cache_size_limit < 128:
    torch._dynamo.config.cache_size_limit = 128

if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
    HAS_CUDA = True

if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
    HAS_XPU = True

if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
    HAS_MPS = True

# mlx check
try:
    import mlx.core.metal
    HAS_MLX = True
except BaseException:
    pass

def torch_compile(module: Union[torch.nn.Module, Callable], backend:str ="inductor", mode: str = None, fullgraph=False):
    from ..models.base import PYTORCH_MIN_VERSION_WITH_COMPILE

    if Version(torch.__version__) < PYTORCH_MIN_VERSION_WITH_COMPILE:
        return module
    try:
        return torch.compile(module, backend=backend, mode=mode, fullgraph=fullgraph)
    except BaseException:
        log.warn(f"Failed to compile `{module}`")
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
    return None

def torch_sync(device: torch.device = None):
    # check all backends
    if device is None:
        if HAS_CUDA:
            torch.cuda.synchronize()
        if HAS_XPU:
            torch.xpu.synchronize()
        if HAS_MPS:
            torch.mps.synchronize()
        return

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def torch_empty_cache(device: torch.device = None, gc: bool = True):
    if gc:
        py_gc.collect()

    # check all backends
    if device is None:
        if HAS_CUDA:
            torch.cuda.empty_cache()
        if HAS_XPU:
            torch.xpu.empty_cache()
        if HAS_MPS:
            torch.mps.empty_cache()
        if HAS_MLX:
            mlx.core.metal.clear_cache()
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
            mlx.core.metal.clear_cache()
