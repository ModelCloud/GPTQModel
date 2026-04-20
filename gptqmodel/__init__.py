# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os


# isort: off
from ._banner import get_startup_banner  # noqa: E402
from .utils import _MONKEY_PATCH_LOCK  # noqa: E402
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

    with _MONKEY_PATCH_LOCK:
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


def _patch_transformers_paroquant_quantizer_compat():
    """Teach transformers to treat ParoQuant checkpoints as GPTQ-backed configs.

    Upstream transformers currently rejects `quant_method="paroquant"` before
    the GPT-QModel loader path gets a chance to handle the checkpoint. ParoQuant
    artifacts reuse GPT-QModel/Optimum loading semantics, so register the method
    alongside GPTQ only when upstream has not provided native support yet.
    """
    try:
        from transformers.quantizers import auto as hf_quant_auto
        from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
        from transformers.utils.quantization_config import GPTQConfig
    except Exception:
        return

    with _MONKEY_PATCH_LOCK:
        if getattr(hf_quant_auto, "_gptqmodel_paroquant_quantizer_compat", False):
            return

        hf_quant_auto.AUTO_QUANTIZATION_CONFIG_MAPPING.setdefault("paroquant", GPTQConfig)
        hf_quant_auto.AUTO_QUANTIZER_MAPPING.setdefault("paroquant", GptqHfQuantizer)
        hf_quant_auto._gptqmodel_paroquant_quantizer_compat = True


def _patch_openvino_gptqmodel_compat():
    """Extend OpenVINO's GPTQ patcher to understand GPTQModel new kernels."""
    try:
        from openvino.frontend.pytorch import gptq as ov_gptq
    except Exception:
        return

    with _MONKEY_PATCH_LOCK:
        if getattr(ov_gptq, "_gptqmodel_torch_quant_compat", False):
            return

        class MatchAll(list):
            def __contains__(self, item):
                return True

        ov_gptq.supported_quant_types = MatchAll()
        ov_gptq._gptqmodel_torch_quant_compat = True


def _patch_transformers_causal_conv1d_hub_kernel_compat():
    """Prioritize local kernel ports before falling back to Transformers hub kernels."""
    try:
        import importlib
        from pathlib import Path

        import transformers.integrations as hf_integrations
        from transformers.integrations import hub_kernels
    except Exception:
        return

    with _MONKEY_PATCH_LOCK:
        if getattr(hub_kernels, "_gptqmodel_local_causal_conv1d_kernel", False):
            if hasattr(hf_integrations, "lazy_load_kernel"):
                hf_integrations.lazy_load_kernel = hub_kernels.lazy_load_kernel
            return

        original_lazy_load_kernel = hub_kernels.lazy_load_kernel
        project_root = Path(__file__).resolve().parent.parent

        def _resolve_local_kernel_module(kernel_name):
            module_name = kernel_name.replace("-", "_")
            namespaced_module = f"gptqmodel.hf_kernels.{module_name}"
            if (project_root / "gptqmodel" / "hf_kernels" / module_name / "__init__.py").exists() or (
                project_root / "gptqmodel" / "hf_kernels" / f"{module_name}.py"
            ).exists():
                return importlib.import_module(namespaced_module)
            package_path = project_root / module_name / "__init__.py"
            module_path = project_root / f"{module_name}.py"
            if not package_path.exists() and not module_path.exists():
                return None
            return importlib.import_module(module_name)

        def _lazy_load_kernel_with_local_causal_conv1d(kernel_name, mapping=hub_kernels._KERNEL_MODULE_MAPPING):
            local_kernel = _resolve_local_kernel_module(kernel_name)
            if local_kernel is not None:
                mapping[kernel_name] = local_kernel
                return local_kernel
            return original_lazy_load_kernel(kernel_name, mapping)

        hub_kernels.lazy_load_kernel = _lazy_load_kernel_with_local_causal_conv1d
        local_causal_conv1d = _resolve_local_kernel_module("causal-conv1d")
        if local_causal_conv1d is not None:
            hub_kernels._KERNEL_MODULE_MAPPING["causal-conv1d"] = local_causal_conv1d
        if hasattr(hf_integrations, "lazy_load_kernel"):
            hf_integrations.lazy_load_kernel = _lazy_load_kernel_with_local_causal_conv1d
        hub_kernels._gptqmodel_local_causal_conv1d_kernel = True


from .utils.env import env_flag
from .utils.logger import setup_logger
from .utils.modelscope import ensure_modelscope_available


DEBUG_ON = env_flag("DEBUG")

from .utils.linalg_warmup import run_torch_linalg_warmup
from .utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask


_DEVICE_THREAD_POOL = None


def _build_device_thread_pool():
    return DeviceThreadPool(
        inference_mode=True,
        warmups={
            "cuda": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "xpu": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "mps": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "cpu": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
        },
        workers={
            "cuda:per": 4,
            "xpu:per": 1,
            "mps": 8,
            "cpu": min(12, max(1, (os.cpu_count() or 1) + 1 // 2)),  # count + 1, fixed pool size > 1 check when count=3
            "model_loader:cpu": 2,
        },
        empty_cache_every_n=512,
    )


def get_device_thread_pool():
    global _DEVICE_THREAD_POOL
    if _DEVICE_THREAD_POOL is None:
        _DEVICE_THREAD_POOL = _build_device_thread_pool()
    return _DEVICE_THREAD_POOL


class _LazyDeviceThreadPoolProxy:
    def __init__(self):
        object.__setattr__(self, "_overrides", {})

    def __getattribute__(self, name):
        if name in {
            "_overrides",
            "__class__",
            "__dict__",
            "__getattribute__",
            "__setattr__",
            "__delattr__",
            "__repr__",
            "__dir__",
            "_get_pool",
        }:
            return object.__getattribute__(self, name)

        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]

        return getattr(self._get_pool(), name)

    def __setattr__(self, name, value):
        if name == "_overrides":
            object.__setattr__(self, name, value)
            return
        self._overrides[name] = value

    def __delattr__(self, name):
        overrides = self._overrides
        if name in overrides:
            del overrides[name]
            return
        delattr(self._get_pool(), name)

    def __repr__(self):
        pool = _DEVICE_THREAD_POOL
        if pool is None:
            return "<LazyDeviceThreadPoolProxy uninitialized>"
        return repr(pool)

    def __dir__(self):
        attrs = set(self._overrides.keys())
        pool = _DEVICE_THREAD_POOL
        if pool is not None:
            attrs.update(dir(pool))
        return sorted(attrs)

    @staticmethod
    def _get_pool():
        return get_device_thread_pool()


DEVICE_THREAD_POOL = _LazyDeviceThreadPoolProxy()

_patch_transformers_gptq_device_map_compat()
_patch_transformers_paroquant_quantizer_compat()
_patch_openvino_gptqmodel_compat()
_patch_transformers_causal_conv1d_hub_kernel_compat()


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


import torch

from . import extension
from .models import GPTQModel, get_best_device
from .models.auto import ASCII_LOGO, TRANSFORMERS_VERSION
from .quantization import (
    AWQConfig,
    BaseQuantizeConfig,
    FOEMConfig,
    GGUFConfig,
    GPTAQConfig,
    GPTQConfig,
    QuantizeConfig,
    RTNConfig,
    WeightOnlyConfig,
)
from .utils import BACKEND, PROFILE
from .version import __version__


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
