# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


DEBUG_ON = str(os.environ.get("DEBUG", "")).lower() in ("1", "true", "yes", "on")

from .utils.threadx import DeviceThreadPool


DEVICE_THREAD_POOL = DeviceThreadPool(
    inference_mode=True,
    workers={
        "cuda:per": 4,
        "xpu:per": 1,
        "mps": 4,
        "cpu": 4,
    },
    empty_cache_every_n=1024,
)

from .models import GPTQModel, get_best_device
from .quantization import BaseQuantizeConfig, QuantizeConfig
from .utils import BACKEND
from .utils.exllama import exllama_set_max_input_length
from .version import __version__


if os.getenv('GPTQMODEL_USE_MODELSCOPE', 'False').lower() in ['true', '1']:
    try:
        from modelscope.utils.hf_util.patcher import patch_hub
        patch_hub()
    except Exception:
        raise ModuleNotFoundError("you have set GPTQMODEL_USE_MODELSCOPE env, but doesn't have modelscope? install it with `pip install modelscope`")
