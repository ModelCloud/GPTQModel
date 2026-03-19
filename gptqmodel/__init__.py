# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


# isort: off
from ._banner import get_startup_banner  # noqa: E402
from .utils.nogil_patcher import TritonPatch, patch_safetensors_save_file  # noqa: E402
# isort: on

patch_safetensors_save_file()

# TODO: waiting for official fix from triton
#  monkeypatching triton threading issues is fragile
try:
    TritonPatch.apply()
except Exception:
    pass


def _patch_transformers_gptq_device_map_compat():
    """Preserve concrete single-device GPTQ maps for Optimum's later packing step."""
    try:
        from functools import wraps

        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
    except Exception:
        return

    original_process = GptqHfQuantizer._process_model_before_weight_loading
    if getattr(original_process, "_gptqmodel_device_map_compat", False):
        return

    @wraps(original_process)
    def _process_model_before_weight_loading_with_device_map(self, model, **kwargs):
        """Backfill `hf_device_map` when GPTQ uses a single concrete device."""
        device_map = kwargs.get("device_map")
        if (
            isinstance(device_map, dict)
            and device_map
            and len(set(device_map.values())) == 1
            and not hasattr(model, "hf_device_map")
        ):
            model.hf_device_map = dict(device_map)
        return original_process(self, model, **kwargs)

    _process_model_before_weight_loading_with_device_map._gptqmodel_device_map_compat = True
    GptqHfQuantizer._process_model_before_weight_loading = _process_model_before_weight_loading_with_device_map

from .utils.env import env_flag
from .utils.logger import setup_logger
from .utils.modelscope import ensure_modelscope_available


DEBUG_ON = env_flag("DEBUG")

from .utils.linalg_warmup import run_torch_linalg_warmup
from .utils.threadx import DeviceThreadPool


DEVICE_THREAD_POOL = DeviceThreadPool(
    inference_mode=True,
    warmups={
        "cuda": run_torch_linalg_warmup,
        "xpu": run_torch_linalg_warmup,
        "mps": run_torch_linalg_warmup,
        "cpu": run_torch_linalg_warmup,
    },
    workers={
        "cuda:per": 4,
        "xpu:per": 1,
        "mps": 8,
        "cpu": min(12, max(1, (os.cpu_count() or 1) + 1 // 2)), # count + 1, fixed pool size > 1 check when count=3
        "model_loader:cpu": 2,
    },
    empty_cache_every_n=512,
)


_patch_transformers_gptq_device_map_compat()


def exllama_set_max_input_length(model, max_input_length: int):
    """Resize exllama scratch buffers through the legacy package-root API."""
    from .utils.model import hf_gptqmodel_post_init

    quantize_config = getattr(model, "quantize_config", None)
    use_act_order = bool(getattr(quantize_config, "desc_act", False))
    return hf_gptqmodel_post_init(
        model,
        use_act_order=use_act_order,
        quantize_config=quantize_config,
        max_input_length=max_input_length,
    )


from .models import GPTQModel, get_best_device
from .models.auto import ASCII_LOGO, TRANSFORMERS_VERSION
from .quantization import BaseQuantizeConfig, GPTAQConfig, QuantizeConfig
from .utils import BACKEND
from .version import __version__
import torch


setup_logger().info(
    "\n%s",
    get_startup_banner(
        ASCII_LOGO,
        gptqmodel_version=__version__,
        transformers_version=TRANSFORMERS_VERSION,
        torch_version=torch.__version__,
    ),
)


if ensure_modelscope_available():
    try:
        from modelscope.utils.hf_util.patcher import patch_hub
        patch_hub()
    except Exception as exc:
        raise ModuleNotFoundError(
            "env `GPTQMODEL_USE_MODELSCOPE` used but modelscope pkg is not found: please install with `pip install modelscope`."
        ) from exc
