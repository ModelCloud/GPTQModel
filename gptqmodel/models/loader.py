# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import copy
import os
import time
from importlib.metadata import PackageNotFoundError, version
from itertools import chain
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from ..utils.modelscope import ensure_modelscope_available
from ..utils.structure import LazyTurtle, print_module_tree


if ensure_modelscope_available():
    from modelscope import snapshot_download
else:
    from huggingface_hub import snapshot_download

import defuser
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.utils import is_flash_attn_2_available

from ..adapter.adapter import Adapter
from ..nn_modules.exllamav3 import ExllamaV3Linear
from ..nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from ..nn_modules.qlinear.exllamav2 import ExllamaV2Linear
from ..nn_modules.qlinear.gguf import GGUFTorchLinear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, METHOD, MIN_VERSION_WITH_V2, BaseQuantizeConfig, resolve_quant_format
from ..utils import internal_gguf
from ..utils.backend import BACKEND, PROFILE, normalize_backend, normalize_profile
from ..utils.exllamav3 import replace_exllamav3_placeholders
from ..utils.hf import (
    INTERNAL_HF_GGUF_FILE_KWARG,
    get_hf_config_dtype,
    get_hf_gguf_load_kwargs,
    has_native_transformers_causallm_support,
    normalize_hf_config_compat,
    normalize_model_id_or_path_for_hf_gguf,
    normalize_torch_dtype_kwarg,
    prepare_remote_model_init_compat,
    resolve_trust_remote_code,
    set_hf_config_dtype,
    suspend_hf_weight_init,
)
from ..utils.importer import (
    auto_select_device,
    get_kernel_for_backend,
    normalize_device_device_map,
    select_quant_linear,
)
from ..utils.inspect import safe_kwargs_call
from ..utils.logger import setup_logger
from ..utils.machete import _validate_machete_device_support
from ..utils.marlin import _marlin_capability_supported, _validate_marlin_device_support
from ..utils.model import (
    auto_dtype,
    convert_gptq_v1_to_v2_format,
    find_config_seq_len,
    find_modules,
    get_checkpoints,
    get_module_by_name_prefix,
    gptqmodel_post_init,
    load_checkpoint_in_model_then_tie_weights,
    make_quant,
    simple_dispatch_model,
)
from ._const import DEVICE, normalize_device


log = setup_logger()

ATTN_IMPLEMENTATION = "attn_implementation"


def _should_print_module_tree() -> bool:
    """Keep expensive module-tree dumps opt-in during model loading."""

    raw = os.environ.get("GPTQMODEL_PRINT_MODULE_TREE")
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _maybe_print_module_tree(model) -> None:
    """Print the module tree only when explicitly requested for debugging."""

    if _should_print_module_tree():
        print_module_tree(model=model)


def _supports_flash_attn_2(config: PretrainedConfig) -> bool:
    """Detect whether the resolved HF architecture exposes FA2 kernels."""

    if not getattr(config, "architectures", None):
        return False

    model_class = getattr(transformers, config.architectures[0], None)
    if model_class is None:
        return False

    if hasattr(model_class, "_supports_flash_attn_2"):
        return bool(getattr(model_class, "_supports_flash_attn_2"))
    if hasattr(model_class, "_supports_flash_attn"):
        return bool(getattr(model_class, "_supports_flash_attn"))
    return False


def _is_accelerated_attention_device(device: object) -> bool:
    """Return True when the selected device can run CUDA/ROCm flash attention."""

    if isinstance(device, torch.device):
        return device.type in {"cuda", "hip"}
    if isinstance(device, DEVICE):
        return device in {DEVICE.CUDA, DEVICE.ROCM}
    if isinstance(device, str):
        return device in {"cuda", "rocm", "hip"}
    return False


def _resolve_native_gguf_profile(
    *,
    native_gguf_qspec: Optional["internal_gguf.GGUFQuantizedCheckpointSpec"],
    profile: PROFILE,
) -> PROFILE:
    """Resolve user profile intent for native GGUF checkpoints."""

    if (
        native_gguf_qspec is not None
        and native_gguf_qspec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0_g128
        and profile == PROFILE.AUTO
    ):
        log.info("Loader: Bonsai/Prism Q1_0_g128 PROFILE.AUTO resolved to PROFILE.FAST.")
        return PROFILE.FAST
    return profile


def _should_use_dense_native_gguf_path(
    *,
    native_gguf_qspec: Optional["internal_gguf.GGUFQuantizedCheckpointSpec"],
    profile: PROFILE,
) -> bool:
    """Fast Bonsai mode stays on the dense HF GGUF import path."""

    return (
        native_gguf_qspec is not None
        and native_gguf_qspec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0_g128
        and profile == PROFILE.FAST
    )


def parse_version_string(version_str: str):
    try:
        return Version(version_str)
    except InvalidVersion:
        raise ValueError(f"Invalid version format: {version_str}")


def parse_requirement(req):
    for op in [">=", "<=", ">", "<", "=="]:
        if op in req:
            pkg, version_required = req.split(op, 1)
            return pkg.strip(), op, version_required.strip()
    raise ValueError(f"Unsupported version constraint in: {req}")


def compare_versions(installed_version, required_version, operator):
    installed = parse_version_string(installed_version)
    required = parse_version_string(required_version)
    if operator == ">":
        return installed > required
    elif operator == ">=":
        return installed >= required
    elif operator == "<":
        return installed < required
    elif operator == "<=":
        return installed <= required
    elif operator == "==":
        return installed == required
    else:
        raise ValueError(f"Unsupported operator: {operator}")


def _is_meta_shell_build_error(exc: Exception) -> bool:
    # Some trust_remote_code model constructors call int()/item() on tensors
    # during __init__, which breaks when the shell is built on the meta device.
    message = str(exc)
    return "cannot be called on meta tensors" in message and ".item()" in message


def _coerce_quantized_awq_dtype(*, backend: BACKEND, qcfg: QuantizeConfig, dtype):
    if qcfg.quant_method not in (METHOD.AWQ, METHOD.PARO):
        return dtype
    if backend in (None, BACKEND.AUTO, BACKEND.AUTO_TRAINABLE):
        return dtype
    if not isinstance(dtype, torch.dtype):
        return dtype

    try:
        qlinear = get_kernel_for_backend(backend, qcfg.quant_method, qcfg.format)
    except ValueError:
        return dtype

    supported_dtypes = getattr(qlinear, "SUPPORTS_DTYPES", None) or []
    if dtype in supported_dtypes or torch.float16 not in supported_dtypes:
        return dtype

    log.info(f"Loading Quantized Model: Auto fix `dtype` to `torch.float16` for `{qlinear.__name__}`")
    return torch.float16


def check_versions(model_class, requirements: List[str]):
    if requirements is None:
        return
    for req in requirements:
        pkg, operator, version_required = parse_requirement(req)
        try:
            installed_version = version(pkg)
            if not compare_versions(installed_version, version_required, operator):
                raise ValueError(f"{model_class} requires version {req}, but current {pkg} version is {installed_version} ")
        except PackageNotFoundError:
            raise ValueError(f"{model_class} requires version {req}, but {pkg} not installed.")


def set_dtype_compat(model_init_kwargs: dict, torch_dtype):
    """
    Set dtype argument in a version-compatible way for Transformers.
    See: https://github.com/huggingface/transformers/releases/tag/v4.56.0

    Args:
        model_init_kwargs (dict): kwargs used to initialize model
        torch_dtype: torch dtype (e.g. torch.float16)
    """
    if Version(transformers.__version__) >= Version("4.56.0"):
        model_init_kwargs["dtype"] = torch_dtype
    else:
        model_init_kwargs["torch_dtype"] = torch_dtype

def get_model_local_path(pretrained_model_id_or_path, **kwargs):
    is_local = os.path.isdir(pretrained_model_id_or_path)
    if is_local or os.path.isabs(pretrained_model_id_or_path):
        return os.path.normpath(pretrained_model_id_or_path)
    kwargs.pop(INTERNAL_HF_GGUF_FILE_KWARG, None)
    def _log_removed(removed: list[str]):
        log.debug("Loader: dropping unsupported snapshot_download kwargs: %s", ", ".join(removed))

    return safe_kwargs_call(
        snapshot_download,
        pretrained_model_id_or_path,
        kwargs=kwargs,
        on_removed=_log_removed,
    )


def _get_tokenizer_load_kwargs(model_init_kwargs: Dict) -> Dict:
    return get_hf_gguf_load_kwargs(model_init_kwargs)


def _resolve_local_gguf_checkpoint_path(model_local_path: str, hf_gguf_load_kwargs: Dict[str, str]) -> Optional[str]:
    gguf_file = hf_gguf_load_kwargs.get("gguf_file")
    if not gguf_file:
        return None

    checkpoint_path = os.path.join(str(model_local_path), gguf_file)
    if not os.path.isfile(checkpoint_path):
        return None
    return checkpoint_path


def _resolve_native_quantized_gguf_checkpoint(
    model_local_path: str,
    hf_gguf_load_kwargs: Dict[str, str],
) -> tuple[Optional[str], Optional[internal_gguf.GGUFQuantizedCheckpointSpec]]:
    if not internal_gguf.native_quantized_loader_enabled():
        return None, None

    gguf_checkpoint_path = _resolve_local_gguf_checkpoint_path(model_local_path, hf_gguf_load_kwargs)
    if gguf_checkpoint_path is None:
        return None, None

    try:
        spec = internal_gguf.inspect_quantized_checkpoint(gguf_checkpoint_path)
    except Exception as exc:
        log.debug("Loader: failed to inspect GGUF checkpoint `%s`: %s", gguf_checkpoint_path, exc)
        return None, None

    if spec is None:
        return None, None
    return gguf_checkpoint_path, spec


def _resolve_model_slot(model: torch.nn.Module, name: str) -> tuple[torch.nn.Module, str]:
    module_name, _, attr_name = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    return module, attr_name


def _lookup_model_slot_tensor(model: torch.nn.Module, name: str) -> torch.Tensor:
    module, attr_name = _resolve_model_slot(model, name)
    if attr_name in module._parameters:
        return module._parameters[attr_name]
    if attr_name in module._buffers:
        return module._buffers[attr_name]
    raise KeyError(f"Loader: model slot `{name}` does not exist.")


def _assign_model_slot_tensor(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    module, attr_name = _resolve_model_slot(model, name)
    tensor = tensor.contiguous()

    if attr_name in module._parameters:
        current = module._parameters[attr_name]
        if current is not None and (tensor.device != current.device or tensor.dtype != current.dtype):
            tensor = tensor.to(device=current.device, dtype=current.dtype)
        requires_grad = current.requires_grad if isinstance(current, torch.nn.Parameter) else False
        module._parameters[attr_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return

    if attr_name in module._buffers:
        current = module._buffers[attr_name]
        if current is not None and (tensor.device != current.device or tensor.dtype != current.dtype):
            tensor = tensor.to(device=current.device, dtype=current.dtype)
        module._buffers[attr_name] = tensor
        return

    raise KeyError(f"Loader: model slot `{name}` does not exist.")


def _build_gguf_tensor_key_mapping(model: torch.nn.Module, config: PretrainedConfig) -> dict[str, str]:
    import transformers.modeling_gguf_pytorch_utils as gguf_utils

    processor_cls = gguf_utils.TENSOR_PROCESSORS.get(config.model_type, gguf_utils.TensorProcessor)
    if processor_cls is not gguf_utils.TensorProcessor:
        raise NotImplementedError(
            f"Loader: native quantized GGUF loading only supports the default tensor processor. "
            f"Actual processor for `{config.model_type}`: `{processor_cls.__name__}`."
        )

    processor = processor_cls(config=config.to_dict())
    return gguf_utils.get_gguf_hf_weights_map(model, processor)


def _load_quantized_gguf_checkpoint_into_model(
    *,
    model: torch.nn.Module,
    gguf_checkpoint_path: str,
    tensor_key_mapping: dict[str, str],
) -> None:
    reader = internal_gguf.GGUFReader(gguf_checkpoint_path)
    loaded: set[str] = set()

    for tensor in reader.tensors:
        target_name = tensor_key_mapping.get(tensor.name)
        if target_name is None:
            continue

        module_name, _, attr_name = target_name.rpartition(".")
        target_module = model.get_submodule(module_name) if module_name else model
        resolved_target_name = target_name

        if isinstance(target_module, GGUFTorchLinear) and attr_name == "weight":
            resolved_target_name = f"{module_name}.qweight" if module_name else "qweight"
            packed = torch.from_numpy(np.array(tensor.data, dtype=np.uint8, copy=True, order="C"))
            expected = _lookup_model_slot_tensor(model, resolved_target_name)
            if tuple(packed.shape) != tuple(expected.shape):
                raise RuntimeError(
                    f"Loader: GGUF qweight shape mismatch for `{resolved_target_name}`. "
                    f"Expected {tuple(expected.shape)}, got {tuple(packed.shape)}."
                )
            _assign_model_slot_tensor(model, resolved_target_name, packed)
            loaded.add(resolved_target_name)
            continue

        reference = _lookup_model_slot_tensor(model, resolved_target_name)
        weights = internal_gguf.dequantize_to_torch(
            tensor.data,
            tensor.tensor_type,
            device=reference.device,
            dtype=reference.dtype,
        )
        _assign_model_slot_tensor(model, resolved_target_name, weights)
        loaded.add(resolved_target_name)

    missing_qweights = []
    for module_name, module in model.named_modules():
        if not isinstance(module, GGUFTorchLinear):
            continue
        qweight_name = f"{module_name}.qweight" if module_name else "qweight"
        if qweight_name not in loaded:
            missing_qweights.append(qweight_name)
    if missing_qweights:
        raise RuntimeError(
            "Loader: GGUF checkpoint did not populate required quantized weights: "
            + ", ".join(sorted(missing_qweights))
        )

    model.tie_weights()


def ModelLoader(cls):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_id_or_path: str,
            quantize_config: BaseQuantizeConfig,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            profile: Union[str, int, PROFILE] = PROFILE.AUTO,
            trust_remote_code: bool = False,
            dtype: [str | torch.dtype] = "auto",
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            **model_init_kwargs,
    ):
        # quantization is unsafe with GIL=0 and torch.compile/graphs
        import torch._dynamo
        torch._dynamo.disable()

        pretrained_model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
            pretrained_model_id_or_path,
            model_init_kwargs,
            api_name=f"{cls.__name__}.from_pretrained",
        )

        dtype = normalize_torch_dtype_kwarg(
            model_init_kwargs,
            api_name=f"{cls.__name__}.from_pretrained",
            explicit_dtype=dtype,
        )
        backend = normalize_backend(backend)
        profile = normalize_profile(profile)
        hf_gguf_load_kwargs = get_hf_gguf_load_kwargs(model_init_kwargs)
        model_init_kwargs_without_internal = dict(model_init_kwargs)
        model_init_kwargs_without_internal.pop(INTERNAL_HF_GGUF_FILE_KWARG, None)

        tokenizer_trust_remote_code = model_init_kwargs_without_internal.pop("tokenizer_trust_remote_code", trust_remote_code)
        model_local_path = get_model_local_path(pretrained_model_id_or_path, **model_init_kwargs_without_internal)
        trust_remote_code = resolve_trust_remote_code(model_local_path, trust_remote_code=trust_remote_code)

        model_init_kwargs_without_internal["trust_remote_code"] = trust_remote_code

        config = AutoConfig.from_pretrained(model_local_path, **model_init_kwargs_without_internal, **hf_gguf_load_kwargs)

        defuser.replace_fused_blocks(config.model_type)

        normalize_hf_config_compat(config, trust_remote_code=trust_remote_code)
        prepare_remote_model_init_compat(model_local_path, config)

        atten_impl = model_init_kwargs.get("attn_implementation", None)

        if atten_impl is not None and atten_impl != "auto":
            log.info(f"Loader: overriding attn_implementation in config to `{atten_impl}`")
            config._attn_implementation = atten_impl

        resolved_device = normalize_device_device_map(device, device_map)
        resolved_device = auto_select_device(resolved_device, backend)

        if cls.require_dtype:
            dtype = cls.require_dtype
        elif dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype):
            dtype = auto_dtype(config=config, device=resolved_device, quant_inference=False)

        if isinstance(dtype, torch.dtype) and get_hf_config_dtype(config) != dtype:
            # Align config metadata with the dtype we will materialize weights in.
            set_hf_config_dtype(config, dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            model_local_path,
            trust_remote_code=tokenizer_trust_remote_code,
            **_get_tokenizer_load_kwargs(model_init_kwargs),
        )

        gguf_checkpoint_path, native_gguf_qspec = _resolve_native_quantized_gguf_checkpoint(
            model_local_path,
            hf_gguf_load_kwargs,
        )
        effective_profile = _resolve_native_gguf_profile(
            native_gguf_qspec=native_gguf_qspec,
            profile=profile,
        )

        if quantize_config is None:
            if native_gguf_qspec is not None:
                if _should_use_dense_native_gguf_path(
                    native_gguf_qspec=native_gguf_qspec,
                    profile=effective_profile,
                ):
                    if backend != BACKEND.AUTO:
                        log.info(
                            "Loader: PROFILE.%s uses dense GGUF import for `%s`; backend `%s` is ignored.",
                            effective_profile.name,
                            gguf_checkpoint_path,
                            backend.value,
                        )
                else:
                    redirect_kwargs = dict(model_init_kwargs)
                    redirect_kwargs.pop("tokenizer_trust_remote_code", None)
                    log.info(
                        "Loader: detected native quantized GGUF checkpoint `%s`; redirecting `%s` to from_quantized() with PROFILE.%s.",
                        gguf_checkpoint_path,
                        cls.__name__,
                        effective_profile.name,
                    )
                    return cls.from_quantized(
                        model_id_or_path=pretrained_model_id_or_path,
                        device_map=device_map,
                        device=device,
                        backend=backend,
                        dtype=dtype,
                        trust_remote_code=trust_remote_code,
                        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
                        **redirect_kwargs,
                    )

            hf_model_init_kwargs = dict(model_init_kwargs_without_internal)
            hf_model_init_kwargs["device_map"] = device_map if device_map else "auto"
            set_dtype_compat(hf_model_init_kwargs, dtype)
            hf_model_init_kwargs.update(hf_gguf_load_kwargs)
            if (
                native_gguf_qspec is not None
                and native_gguf_qspec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0_g128
                and atten_impl in {None, "auto"}
                and _is_accelerated_attention_device(resolved_device)
                and (config.model_type == "qwen3" or _supports_flash_attn_2(config))
                and is_flash_attn_2_available()
            ):
                hf_model_init_kwargs[ATTN_IMPLEMENTATION] = "flash_attention_2"
                log.info("Loader: Auto enabling flash_attention_2 for dense Bonsai PROFILE.%s.", effective_profile.name)
            # Load a non-quantized model, but do not perform quantization. For example, for evaluation.
            model = cls.loader.from_pretrained(model_local_path, config=config, **hf_model_init_kwargs)
            model._model_init_kwargs = hf_model_init_kwargs
            _maybe_print_module_tree(model=model)

            turtle_model = None

            instance = cls(
                model,
                turtle_model=turtle_model,
                quantized=False,
                quantize_config=quantize_config,
                tokenizer=tokenizer,
                trust_remote_code=trust_remote_code,
                model_local_path=model_local_path,
            )

            return instance

        load_start = time.perf_counter()

        # non-quantized models are always loaded into cpu
        cpu_device_map = {"": "cpu"}

        if quantize_config is None or not isinstance(quantize_config, BaseQuantizeConfig):
            raise AttributeError("`quantize_config` must be passed and be an instance of BaseQuantizeConfig.")

        quantize_config.calculate_bits_per_weight()

        if quantize_config.device is not None:
            if device is not None or device_map is not None:
                raise AttributeError("Passing device and device_map is not allowed when QuantizeConfig.device is set. Non-quantized model is always loaded as cpu. Please set QuantizeConfig.device for accelerator used in quantization or do not set for auto-selection.")

        if quantize_config.desc_act not in cls.supports_desc_act:
            raise ValueError(f"{cls} only supports desc_act={cls.supports_desc_act}, "
                             f"but quantize_config.desc_act is {quantize_config.desc_act}.")

        native_support = has_native_transformers_causallm_support(model_local_path)
        if cls.require_trust_remote_code and not trust_remote_code and not native_support:
            raise ValueError(
                f"{pretrained_model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs)

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        # normalize and auto select quantization device is not passed
        if quantize_config.device is None:
            quantize_config.device = auto_select_device(None, None)
        else:
            quantize_config.device = normalize_device(quantize_config.device)

        if dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype):
            # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
            dtype = auto_dtype(config=config, device=quantize_config.device, quant_inference=False)

        # enforce some values despite user specified
        # non-quantized models are always loaded into cpu
        model_init_kwargs_without_internal["device_map"] = cpu_device_map
        set_dtype_compat(model_init_kwargs_without_internal, dtype)
        model_init_kwargs_without_internal["_fast_init"] = cls.require_fast_init
        #model_init_kwargs["low_cpu_mem_usage"] = True

        cls.before_model_load(cls, model_local_path=model_local_path, load_quantized_model=False)
        from ..utils.hf import build_shell_model

        # XIELUActivation will use some weights when activation init, so can't use init_empty_weights
        if hasattr(config, "hidden_act") and config.hidden_act == "xielu":
            quantize_config.offload_to_disk = False

        # some models need convert moe-experts after model loaded, like GPTOSS and Llama4
        # so offload_to_disk is not supported for them.
        if not cls.support_offload_to_disk:
            quantize_config.offload_to_disk = False
            log.warn(f"{cls} doesn't support offload_to_disk, set quantize_config.offload_to_disk to False.")

        if quantize_config.offload_to_disk:
            shell_config = copy.deepcopy(config)
            try:
                model = build_shell_model(cls.loader, config=shell_config, **model_init_kwargs_without_internal)
            except RuntimeError as exc:
                if not _is_meta_shell_build_error(exc):
                    raise

                log.warn(
                    "Loader: meta-device shell build failed for `%s`; falling back to direct CPU load without turtle_model: %s",
                    model_local_path,
                    exc,
                )
                log.info("Loader: loading model directly to CPU (meta shell unsupported; turtle_model disabled)")
                fallback_init_kwargs = model_init_kwargs_without_internal.copy()
                fallback_init_kwargs.pop("device_map", None)
                fallback_init_kwargs["low_cpu_mem_usage"] = False
                model = cls.loader.from_pretrained(
                    model_local_path,
                    config=config,
                    **fallback_init_kwargs,
                    **hf_gguf_load_kwargs,
                )
                if getattr(model, "config", None) is config:
                    model.config = copy.deepcopy(config)
                defuser.convert_model(model, cleanup_original=False)
                model._model_init_kwargs = fallback_init_kwargs
                _maybe_print_module_tree(model=model)
                turtle_model = None
            else:
                defuser.convert_model(model, cleanup_original=False)
                shell_model_init_kwargs = dict(model_init_kwargs_without_internal)
                shell_model_init_kwargs.update(hf_gguf_load_kwargs)
                model._model_init_kwargs = shell_model_init_kwargs
                _maybe_print_module_tree(model=model)
                turtle_model = LazyTurtle.maybe_create(
                    model_local_path=model_local_path,
                    config=model.config,
                    model_init_kwargs=shell_model_init_kwargs,
                    module_tree=copy.deepcopy(getattr(cls, "module_tree", None)),
                    hf_conversion_map_reversed=copy.deepcopy(
                        cls.resolve_hf_conversion_map_reversed(target_model=model)
                    ),
                    target_model=model,
                )

                if turtle_model is None:
                    raise RuntimeError(
                        f"Loader: can't open model path `{model_local_path}` for offload_to_disk."
                    )

                log.info(
                    "Loader: using checkpoint-backed lazy turtle source for `%s`",
                    model_local_path,
                )
        else:
            log.info("Loader: loading model directly to CPU (not using meta device or turtle_model)")
            model = cls.loader.from_pretrained(
                model_local_path,
                config=config,
                **model_init_kwargs_without_internal,
                **hf_gguf_load_kwargs,
            )
            if getattr(model, "config", None) is config:
                model.config = copy.deepcopy(config)
            defuser.convert_model(model, cleanup_original=False)
            direct_model_init_kwargs = dict(model_init_kwargs_without_internal)
            direct_model_init_kwargs.update(hf_gguf_load_kwargs)
            model._model_init_kwargs = direct_model_init_kwargs
            _maybe_print_module_tree(model=model)

            turtle_model = None

        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
        config_seq_len = find_config_seq_len(model_config, seq_len_keys)
        if config_seq_len is not None:
            model.seqlen = config_seq_len
        else:
            log.warn("Model: can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        model.eval()
        turtle_model.eval() if turtle_model is not None else None

        instance = cls(
            model,
            turtle_model=turtle_model,
            quantized=False,
            quantize_config=quantize_config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
        )

        timer = getattr(instance, "quant_region_timer", None)
        if timer is not None:
            source_label = getattr(instance, "model_local_path", None) or str(pretrained_model_id_or_path)
            timer.record("model_load", time.perf_counter() - load_start, source=source_label)

        return instance

    cls.from_pretrained = from_pretrained

    @classmethod
    def from_quantized(
            cls,
            model_id_or_path: Optional[str],
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            backend: Union[str, BACKEND] = BACKEND.AUTO,
            adapter: Optional[Adapter] = None,
            dtype: [str | torch.dtype] = "auto",
            trust_remote_code: bool = False,
            **kwargs,
    ):

        # post-quant is safe with GIL=0 and torch.compile/graphs

        import torch._dynamo
        torch._dynamo.reset()
        model_id_or_path = normalize_model_id_or_path_for_hf_gguf(
            model_id_or_path,
            kwargs,
            api_name=f"{cls.__name__}.from_quantized",
        )
        dtype = normalize_torch_dtype_kwarg(
            kwargs,
            api_name=f"{cls.__name__}.from_quantized",
            explicit_dtype=dtype,
        )
        hf_gguf_load_kwargs = get_hf_gguf_load_kwargs(kwargs)
        kwargs_without_internal = dict(kwargs)
        kwargs_without_internal.pop(INTERNAL_HF_GGUF_FILE_KWARG, None)
        tokenizer_trust_remote_code = kwargs_without_internal.pop("tokenizer_trust_remote_code", trust_remote_code)
        requested_device_map = device_map
        explicit_device_map = requested_device_map if isinstance(requested_device_map, dict) else None

        if requested_device_map is None:
            explicit_device = None
            if isinstance(device, str) and ":" in device:
                explicit_device = device
            elif isinstance(device, torch.device) and device.index is not None:
                explicit_device = str(device)

            if explicit_device is not None:
                explicit_device_map = {"": explicit_device}
                requested_device_map = explicit_device_map

        # normalized device + device_map into single device
        normalized_device = device if requested_device_map is None else None  # let device_map dictate placement when present
        device = normalize_device_device_map(normalized_device, requested_device_map)

        # Keep string inputs compatible while allowing canonical method-prefixed names.
        backend = normalize_backend(backend)
        device = auto_select_device(device, backend)

        if backend == BACKEND.VLLM:
            import os

            # to optimize vllm inference, set an environment variable 'VLLM_ATTENTION_BACKEND' to 'FLASHINFER'.
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

        model_local_path = get_model_local_path(model_id_or_path, **kwargs_without_internal)
        trust_remote_code = resolve_trust_remote_code(model_local_path, trust_remote_code=trust_remote_code)
        native_support = has_native_transformers_causallm_support(model_local_path)

        """load quantized model from local disk"""
        if cls.require_trust_remote_code and not trust_remote_code and not native_support:
            raise ValueError(
                f"{model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs)

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs_without_internal.pop("cache_dir", None)
        force_download = kwargs_without_internal.pop("force_download", False)
        resume_download = kwargs_without_internal.pop("resume_download", False)
        proxies = kwargs_without_internal.pop("proxies", None)
        local_files_only = kwargs_without_internal.pop("local_files_only", False)
        use_auth_token = kwargs_without_internal.pop("use_auth_token", None)
        revision = kwargs_without_internal.pop("revision", None)
        subfolder = kwargs_without_internal.pop("subfolder", "")
        commit_hash = kwargs_without_internal.pop("_commit_hash", None)
        attn_implementation = kwargs_without_internal.pop("attn_implementation", None)

        cached_file_kwargs = {
            "cache_dir": cache_dir,
            "force_download": force_download,
            "proxies": proxies,
            "resume_download": resume_download,
            "local_files_only": local_files_only,
            "use_auth_token": use_auth_token,
            "revision": revision,
            "subfolder": subfolder,
            "_raise_exceptions_for_missing_entries": False,
            "_commit_hash": commit_hash,
            "attn_implementation": attn_implementation,
        }

        # == step1: prepare configs and file names == #
        config: PretrainedConfig = AutoConfig.from_pretrained(
            model_local_path,
            trust_remote_code=trust_remote_code,
            **cached_file_kwargs,
            **hf_gguf_load_kwargs,
        )

        defuser.replace_fused_blocks(config.model_type)

        normalize_hf_config_compat(config, trust_remote_code=trust_remote_code)
        prepare_remote_model_init_compat(model_local_path, config)

        if cls.require_dtype:
            dtype = cls.require_dtype

        if dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype) :
            # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
            dtype = auto_dtype(config=config, device=device, quant_inference=True)

        if isinstance(dtype, torch.dtype) and get_hf_config_dtype(config) != dtype:
            # Ensure flash attention kernels see an explicit dtype instead of relying on defaults.
            set_hf_config_dtype(config, dtype)

        gguf_checkpoint_path, native_gguf_qspec = _resolve_native_quantized_gguf_checkpoint(
            model_local_path,
            hf_gguf_load_kwargs,
        )
        if native_gguf_qspec is not None:
            qcfg = QuantizeConfig(
                bits=native_gguf_qspec.bits_alias,
                method=METHOD.GGUF,
                lm_head=native_gguf_qspec.lm_head_quantized,
            )
        else:
            qcfg = QuantizeConfig.from_pretrained(model_local_path, **cached_file_kwargs, **kwargs_without_internal)
        export_quant_method = qcfg.export_quant_method()
        format_code = resolve_quant_format(qcfg.format, qcfg.method)
        backend = normalize_backend(backend, quant_method=export_quant_method)

        # Prism/Bonsai sign-only GGUF tensors only have a torch runtime today.
        # Bypass higher-priority GGUF backends that either do not support 1-bit
        # formats or depend on optional external runtimes.
        if (
            native_gguf_qspec is not None
            and native_gguf_qspec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0
        ):
            if backend == BACKEND.AUTO:
                backend = BACKEND.GGUF_TORCH
            elif backend != BACKEND.GGUF_TORCH:
                raise ValueError(
                    "Native Q1_0 GGUF checkpoints currently require BACKEND.GGUF_TORCH. "
                    f"Actual backend: `{backend}`."
                )
        elif (
            native_gguf_qspec is not None
            and native_gguf_qspec.tensor_qtype == internal_gguf.GGMLQuantizationType.Q1_0_g128
            and backend not in {BACKEND.AUTO, BACKEND.GGUF_TORCH, BACKEND.GGUF_TRITON}
        ):
            raise ValueError(
                "Native Q1_0_g128 GGUF checkpoints support BACKEND.AUTO, BACKEND.GGUF_TORCH, or BACKEND.GGUF_TRITON. "
                f"Actual backend: `{backend}`."
            )

        if format_code == FORMAT.EXL3:
            if backend not in (BACKEND.AUTO, BACKEND.EXL3_EXLLAMA_V3, BACKEND.EXL3_TORCH):
                raise TypeError("FORMAT.EXL3 requires BACKEND.AUTO, BACKEND.EXL3_EXLLAMA_V3, or BACKEND.EXL3_TORCH.")
            if backend == BACKEND.AUTO:
                if torch.cuda.is_available() and device in (DEVICE.CUDA, DEVICE.ROCM):
                    backend = BACKEND.EXL3_EXLLAMA_V3
                else:
                    backend = BACKEND.EXL3_TORCH
            if backend == BACKEND.EXL3_EXLLAMA_V3:
                if not torch.cuda.is_available():
                    raise ValueError("EXL3 CUDA loading requires CUDA/HIP.")
                if device not in (DEVICE.CUDA, DEVICE.ROCM):
                    raise ValueError("EXL3 CUDA loading requires a CUDA/HIP device.")
        elif format_code == FORMAT.BITSANDBYTES:
            if backend not in (BACKEND.AUTO, BACKEND.BITSANDBYTES):
                raise TypeError("FORMAT.BITSANDBYTES requires BACKEND.AUTO or BACKEND.BITSANDBYTES.")
            backend = BACKEND.BITSANDBYTES

        if export_quant_method == METHOD.AWQ and format_code in [FORMAT.GEMV_FAST, FORMAT.LLM_AWQ]:
            # GEMV_FAST and LLM_AWQ only supports torch.float16
            log.info("Loading Quantized Model: Auto fix `dtype` to `torch.float16`")
            dtype = torch.float16

        dtype = _coerce_quantized_awq_dtype(backend=backend, qcfg=qcfg, dtype=dtype)

        # inject adapter into qcfg
        if adapter is not None:
            qcfg.adapter = adapter

        qcfg.calculate_bits_per_weight()

        tokenizer = AutoTokenizer.from_pretrained(
            model_local_path,
            trust_remote_code=tokenizer_trust_remote_code,
            **hf_gguf_load_kwargs,
        )

        if backend == BACKEND.VLLM or backend == BACKEND.SGLANG:
            runtime_generate = None
            if backend == BACKEND.VLLM:
                if format_code not in [FORMAT.GPTQ, FORMAT.GEMM]:
                    raise ValueError(f"{backend} backend only supports FORMAT.GPTQ or FORMAT.GEMM: actual = {qcfg.format}")
            elif backend == BACKEND.SGLANG:
                if format_code != FORMAT.GPTQ:
                    raise ValueError(f"{backend} backend only supports FORMAT.GPTQ: actual = {qcfg.format}")

            if backend == BACKEND.VLLM:
                from ..utils.vllm import load_model_by_vllm, vllm_generate

                model = load_model_by_vllm(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs_without_internal,
                )

                model.config = model.llm_engine.model_config
                model.device = model.llm_engine.vllm_config.device_config.device
                runtime_generate = vllm_generate

            elif backend == BACKEND.SGLANG:
                from ..utils.sglang import load_model_by_sglang, sglang_generate

                model, hf_config = load_model_by_sglang(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    dtype=torch.float16,
                    **kwargs_without_internal,
                )
                model.config = hf_config
                runtime_generate = sglang_generate
            instance = cls(
                model,
                quantized=True,
                quantize_config=qcfg,
                tokenizer=tokenizer,
                qlinear_kernel=None,
                load_quantized_model=True,
                trust_remote_code=trust_remote_code,
                model_local_path=model_local_path,
            )
            instance._runtime_generate = runtime_generate
            return instance

        if format_code == FORMAT.MARLIN:
            # format marlin requires marlin kernel
            expected_marlin_backend = BACKEND.AWQ_MARLIN if qcfg.quant_method == METHOD.AWQ else BACKEND.GPTQ_MARLIN
            expected_marlin_backends = [expected_marlin_backend]
            if backend not in expected_marlin_backends and backend != BACKEND.AUTO:
                raise TypeError(
                    f"FORMAT.MARLIN requires BACKEND.AUTO or BACKEND.{expected_marlin_backend.name}: actual = `{backend}`."
                )
            backend = expected_marlin_backend

        # marlin_compatible = False if backend == BACKEND.IPEX else _validate_marlin_device_support()
        # check for marlin compat for cuda device only
        # if backend not in [BACKEND.GPTQ_MARLIN, BACKEND.AWQ_MARLIN] and device == DEVICE.CUDA:
        #     unsupported = _validate_marlin_compatibility(qcfg)
        #     if unsupported is None and marlin_compatible:
        #         logger.info(
        #             "Hint: Model is compatible with the Marlin kernel. Use the canonical Marlin BACKEND enum."
        #         )

        if format_code == FORMAT.BITBLAS:
            # format bitblas requires bitblas kernel
            expected_backend = BACKEND.AWQ_BITBLAS if qcfg.quant_method == METHOD.AWQ else BACKEND.GPTQ_BITBLAS
            if backend != expected_backend and backend != BACKEND.AUTO:
                raise TypeError(
                    f"FORMAT.BITBLAS requires BACKEND.AUTO or BACKEND.{expected_backend.name}: actual = `{backend}`."
                )
            backend = expected_backend

        if backend in [BACKEND.GPTQ_BITBLAS, BACKEND.AWQ_BITBLAS]:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        model_local_path = str(model_local_path)
        if native_gguf_qspec is not None:
            is_sharded = False
            model_save_name = gguf_checkpoint_path
        else:
            if format_code == FORMAT.EXL3:
                possible_model_basenames = ["model"]
            else:
                possible_model_basenames = [
                    f"gptq_model-{qcfg.bits}bit-{qcfg.group_size}g",
                    "model",
                ]

            extensions = [".safetensors"]

            # Retrieve (and if necessary download) the quantized checkpoint(s).
            is_sharded, resolved_archive_file, true_model_basename = get_checkpoints(
                model_id_or_path=model_local_path,
                extensions=extensions,
                possible_model_basenames=possible_model_basenames,
                **cached_file_kwargs,
            )

            # bin files have security issues: disable loading by default
            if ".bin" in resolved_archive_file:
                raise ValueError(
                    "Loading of .bin files are not allowed due to safety. Please convert your model to safetensor or pytorch format."
                )

            model_save_name = resolved_archive_file  # In case a model is sharded, this would be `model.safetensors.index.json` which may later break.

        qcfg.runtime_format = format_code

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        gguf_tensor_key_mapping = None
        with suspend_hf_weight_init():
            cls.before_model_load(cls, model_local_path=model_local_path, load_quantized_model=True)

            if config.architectures:
                model_class = getattr(transformers, config.architectures[0], None)
                if model_class is not None:
                    # backward-compatible fallback for "_supports_flash_attn" field
                    if hasattr(model_class, "_supports_flash_attn_2"):
                        supports_flash_attn = getattr(model_class, "_supports_flash_attn_2")
                    elif hasattr(model_class, "_supports_flash_attn"):
                        supports_flash_attn = getattr(model_class, "_supports_flash_attn")
                else:
                    supports_flash_attn = None
            else:
                supports_flash_attn = None

            args = {}
            if supports_flash_attn and device in [DEVICE.CUDA, DEVICE.ROCM]:
                if attn_implementation is not None:
                    args[ATTN_IMPLEMENTATION] = attn_implementation
                elif is_flash_attn_2_available():
                    args = {ATTN_IMPLEMENTATION: "flash_attention_2"}
                    log.info("Loader: Auto enabling flash attention2")
            set_dtype_compat(args, dtype)

            model = cls.loader.from_config(
                config, trust_remote_code=trust_remote_code, **args
            )
            defuser.convert_model(model, cleanup_original=True)
            model.checkpoint_file_name = model_save_name
            if native_gguf_qspec is not None:
                gguf_tensor_key_mapping = _build_gguf_tensor_key_mapping(model, config)

            extract_layers_node = cls.extract_layers_node()
            # Get the first layer to determine layer type
            layers, _ = get_module_by_name_prefix(model, extract_layers_node)

            modules = find_modules(model)
            ignore_modules = [cls.lm_head] + cls.get_base_modules(model)

            simple_layer_modules = cls.simple_layer_modules(config, qcfg)
            for name in list(modules.keys()):
                # allow loading of quantized lm_head
                if qcfg.lm_head and name == cls.lm_head:
                    continue

                if not any(name.startswith(prefix) for prefix in extract_layers_node) or any(name.startswith(ignore_module) for ignore_module in ignore_modules) or all(
                        not name.endswith(ignore_module) for sublist in simple_layer_modules for ignore_module in sublist
                ):
                    # log non-lm-head quantized modules only
                    if name is not cls.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]

            if format_code == FORMAT.EXL3:
                if not isinstance(qcfg.tensor_storage, dict) or not qcfg.tensor_storage:
                    raise ValueError("EXL3 checkpoints require `quantization_config.tensor_storage` metadata.")

                exl3_module_cls = ExllamaV3TorchLinear if backend == BACKEND.EXL3_TORCH else ExllamaV3Linear
                replace_exllamav3_placeholders(
                    model=model,
                    module_names=list(qcfg.tensor_storage.keys()),
                    tensor_storage=qcfg.tensor_storage,
                    module_cls=exl3_module_cls,
                )
                preload_qlinear_kernel = exl3_module_cls
            else:
                preload_qlinear_kernel = make_quant(
                    model,
                    qcfg=qcfg,
                    quant_result=modules,
                    backend=backend,
                    lm_head_name=cls.lm_head,
                    device=device,
                    dtype=dtype,
                )

        if isinstance(requested_device_map, str) and requested_device_map not in [
                "auto",
                "balanced",
                "balanced_low_0",
                "sequential",
            ]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )


        def build_layerwise_device_map(
                model,
                device,
                layers: List[torch.nn.Module],
                ignore_modules: List[torch.nn.Module],
                num_gpus: Optional[int] = None,
        ) -> Dict[str, str]:
            """
            Build a deterministic alternating device_map for multi-GPU loading.
            Designed for quantized GPTQ models.

            Rules:
              • Input embedding(s) → GPU 0
              • Each repeating layer → alternate across GPUs (round-robin)
              • Output head (lm_head / embed_out):
                  – If weight-tied with input embedding → GPU 0
                  – Else → last GPU
              • Ignore modules (e.g., norms, projections) → co-located with last layer’s GPU
            """

            if num_gpus is None:
                num_gpus = torch.cuda.device_count()
            if num_gpus < 1:
                raise RuntimeError("No CUDA devices detected")

            device_ids = list(range(num_gpus))
            device_map: Dict[str, str] = {}
            mod2name = {m: n for n, m in model.named_modules()}

            if device == DEVICE.CUDA:
                if torch.cuda.is_available():
                    device_strs = [f"cuda:{i}" for i in range(num_gpus)]
                else:
                    raise RuntimeError("CUDA is not available")
            elif device == DEVICE.XPU:
                if hasattr(torch, "xpu") and torch.xpu.is_available():
                    device_strs = [f"xpu:{i}" for i in range(num_gpus)]
                else:
                    raise RuntimeError("XPU is not available")
            else:
                device_strs = ["cpu"] * num_gpus

            def assign(mod, device_id):
                if mod is None:
                    return
                name = mod2name.get(mod)
                if name is not None:
                    device_map[name] = device_strs[device_id]

            # -------------------------------------------------------------
            # 1–3. Assign input embeddings, layers, and ignored modules
            # -------------------------------------------------------------
            # Input embeddings → GPU 0
            try:
                in_emb = model.get_input_embeddings()
            except NotImplementedError:
                log.warning("Model does not implement get_input_embeddings. Skipping input embeddings assignment.")
                in_emb = None
            assign(in_emb, device_ids[0])

            # Alternating layers
            layer_name2devid: Dict[str, int] = {}

            for i, layer in enumerate(layers):
                gpu = device_ids[i % num_gpus]
                assign(layer, gpu)
                lname = mod2name.get(layer)
                if lname is not None:
                    layer_name2devid[lname] = gpu

            # Ignored modules - skip input embeddings to avoid overriding GPU 0 assignment
            # Iterate over modules that should be ignored during default layer-wise mapping
            for mod in ignore_modules:
                # Preserve GPU-0 placement for the input embedding module if it exists
                if in_emb is not None and mod is in_emb:
                    continue  # Skip input embedding to preserve GPU 0 assignment
                # Retrieve the module’s fully-qualified name
                name = mod2name.get(mod)
                if name is None:
                    continue
                # Walk up the module hierarchy to find the closest ancestor that already has a device assignment
                owner = name
                dev_id = None
                while owner:
                    if owner in layer_name2devid:
                        dev_id = layer_name2devid[owner]
                        break
                    if "." not in owner:
                        break
                    owner = owner.rsplit(".", 1)[0]
                # If no ancestor is found, fall back to the last GPU
                if dev_id is None:
                    dev_id = device_ids[-1]
                # Assign the current module to the determined device
                assign(mod, dev_id)
            # -------------------------------------------------------------
            # 4. Handle lm_head / output projection explicitly
            # -------------------------------------------------------------
            # Look for lm_head or similar projection
            head = getattr(model, "lm_head", None)
            if head is None:
                for cand in ("embed_out", "output_projection", "output_head"):
                    if hasattr(model, cand):
                        head = getattr(model, cand)
                        break

            if head is not None:
                assign(head, device_ids[-1])
                # If weight-tied, co-locate on GPU 0
                tie_cfg = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
                if (
                    in_emb is not None
                    and tie_cfg
                ):
                    assign(head, device_ids[0])

            # -------------------------------------------------------------
            # 5. Safety check: ensure all params are covered
            # -------------------------------------------------------------
            missing = [
                n for n, _ in chain(model.named_parameters(), model.named_buffers())
                if not any(n == k or n.startswith(k + ".") for k in device_map)
            ]
            module_names = set(mod2name.values())
            if missing:
                # map any leftover params (rare) to last GPU
                fallback_device = device_ids[-1]
                for param_name in missing:
                    owner = param_name
                    while owner and owner not in module_names:
                        if "." not in owner:
                            owner = ""
                        else:
                            owner = owner.rsplit(".", 1)[0]
                    if owner:
                        device_map.setdefault(owner, device_strs[fallback_device])
                    else:
                        log.info(f"Loader: unable to map param '{param_name}' to a module; skipping fallback assignment.")

            # -------------------------------------------------------------
            # 6. Prune parent assignments that would override child devices
            # -------------------------------------------------------------
            for name, device_str in list(device_map.items()):
                if not name:
                    continue
                child_devices = {
                    device_map[child_name]
                    for child_name in device_map
                    if child_name != name and child_name.startswith(f"{name}.")
                }
                if child_devices and (len(child_devices) > 1 or device_str not in child_devices):
                    log.info(f"Loader: dropping parent '{name}' from device_map to preserve child placements.")
                    device_map.pop(name, None)

            # Collect parameters/buffers that were not assigned to any device in the current device_map
            missing_after_prune = [
                n for n, _ in chain(model.named_parameters(), model.named_buffers())
                if not any(n == k or n.startswith(k + ".") for k in device_map)
            ]
            # If any tensors remain unmapped, assign them to the last GPU as a fallback
            if missing_after_prune:
                fallback_device = device_ids[-1]
                for param_name in missing_after_prune:
                    # Walk up the module tree until we find a module name that exists in module_names
                    owner = param_name
                    while owner and owner not in module_names:
                        if "." not in owner:
                            owner = ""
                        else:
                            owner = owner.rsplit(".", 1)[0]
                    # Map the closest owning module to the fallback device
                    if owner:
                        device_map.setdefault(owner, device_strs[fallback_device])
                    else:
                        log.info(f"Loader: unable to map param '{param_name}' to a module; skipping fallback assignment.")
            # optional logging for debug
            log.info(f"Loader: Built map across {num_gpus} GPU(s), "
                  f"{len(device_map)} entries. First 8: {list(device_map.items())[:8]}")

            return device_map

        log.info(f"Loader: device = {device}")
        if explicit_device_map is None:
            layers, _ = get_module_by_name_prefix(model, extract_layers_node)
            num_gpus = 1
            if device is DEVICE.CUDA:
                num_gpus = torch.cuda.device_count()
            elif device is DEVICE.XPU:
                num_gpus = torch.xpu.device_count()
            device_map = build_layerwise_device_map(model, device, layers, ignore_modules, num_gpus)
        else:
            device_map = dict(explicit_device_map)
            log.info(f"Loader: honoring explicit device_map request: {device_map}")
        log.info(f"Loader: device_map = {device_map}")

        load_checkpoint_in_model = native_gguf_qspec is None
        # compat: runtime convert checkpoint gptq(v1) to gptq_v2 format
        if format_code in [FORMAT.GPTQ, FORMAT.GEMM, FORMAT.PAROQUANT]:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=dtype,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                offload_state_dict=True,
                offload_buffers=True,
            )

            load_checkpoint_in_model = False

            if format_code == FORMAT.GPTQ:
                # validate sym=False v1 loading needs to be protected for models produced with new v2 format codebase
                if not qcfg.sym and not qcfg.is_quantized_by_gptaq() and not qcfg.is_quantized_by_foem():
                    raise ValueError(
                        f"Format: Loading of a sym=False model with format={FORMAT.GPTQ} is only supported if produced by gptqmodel version >= {MIN_VERSION_WITH_V2}"
                    )

                if preload_qlinear_kernel.REQUIRES_FORMAT_V2:
                    model = convert_gptq_v1_to_v2_format(
                        model,
                        cfg=qcfg,
                        qlinear_kernel=preload_qlinear_kernel,
                    )

                    qcfg.runtime_format = FORMAT.GPTQ_V2

        if backend in (BACKEND.GPTQ_MACHETE, BACKEND.AWQ_MACHETE):
            if is_sharded:
                raise ValueError(
                    "Format: The loading of sharded checkpoints with Machete is currently not supported."
                )
            if not _validate_machete_device_support():
                raise ValueError(
                    f"Kernel: Machete kernel requires compute capability >= 9.0. Detected capability: {torch.cuda.get_device_capability()}"
                )

        if backend in [BACKEND.GPTQ_MARLIN, BACKEND.AWQ_MARLIN] and (
                preload_qlinear_kernel == ExllamaV2Linear or format_code == FORMAT.MARLIN):
            if is_sharded:
                raise ValueError(
                    "Format: The loading of sharded checkpoints with Marlin is currently not supported."
                )
            device_capability = torch.cuda.get_device_capability()
            if backend == BACKEND.GPTQ_MARLIN:
                if not _validate_marlin_device_support():
                    raise ValueError(
                        "Kernel: Marlin kernel requires compute capability >= 7.5 for the "
                        f"GPTQ Marlin backend. Detected capability: `{device_capability}`."
                    )
                if device_capability == (7, 5) and dtype == torch.bfloat16:
                    raise ValueError(
                        "Kernel: GPTQ Marlin on Turing (compute capability 7.5) supports "
                        "dtype=torch.float16 only."
                    )
            elif backend == BACKEND.AWQ_MARLIN:
                if not _marlin_capability_supported(*device_capability) or device_capability[0] < 8:
                    raise ValueError(
                        "Kernel: AWQ Marlin requires compute capability >= 8.0. "
                        f"Detected capability: `{device_capability}`."
                    )

            # GPTQ Marlin and AWQ Marlin support fp16 and bf16 compute on Ampere+.
            if backend == BACKEND.GPTQ_MARLIN and dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Marlin kernel requires dtype=torch.float16 or dtype=torch.bfloat16.")
            if backend == BACKEND.AWQ_MARLIN and dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("AWQ Marlin kernel requires dtype=torch.float16 or dtype=torch.bfloat16.")


        if backend in [BACKEND.GPTQ_BITBLAS, BACKEND.AWQ_BITBLAS]:
            from ..utils.bitblas import prepare_model_for_bitblas_load

            # Prepare model for bitblas load.
            # If is bitblas serialized load then load directly. Otherwise, convert to bitblas.
            model = prepare_model_for_bitblas_load(
                model=model,
                qcfg=qcfg,
                quant_linear_class=preload_qlinear_kernel,
                dtype=dtype,
                model_save_name=model_save_name,
                device_map=device_map,
                desc_act=qcfg.desc_act,
                sym=qcfg.sym,
                load_checkpoint_in_model=load_checkpoint_in_model,
            )

        # If we use marlin or bitblas to load the quantized model, the model is already a converted model,
        # and we no longer need to call load_checkpoint_in_model()
        if load_checkpoint_in_model and backend not in [
            BACKEND.GPTQ_MACHETE,
            BACKEND.AWQ_MACHETE,
            BACKEND.GPTQ_MARLIN,
            BACKEND.AWQ_MARLIN,
            BACKEND.GPTQ_BITBLAS,
            BACKEND.AWQ_BITBLAS,
        ]:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=dtype,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                # offload_state_dict=True,
                # offload_buffers=True,
            )

        if native_gguf_qspec is not None:
            model = simple_dispatch_model(model, device_map)
            _load_quantized_gguf_checkpoint_into_model(
                model=model,
                gguf_checkpoint_path=gguf_checkpoint_path,
                tensor_key_mapping=gguf_tensor_key_mapping,
            )
        else:
            # TODO: Why are we using this custom function and not dispatch_model?
            model = simple_dispatch_model(model, device_map)

        if format_code == FORMAT.EXL3:
            qlinear_kernel = ExllamaV3TorchLinear if backend == BACKEND.EXL3_TORCH else ExllamaV3Linear
        else:
            qlinear_kernel = select_quant_linear(
                bits=qcfg.runtime_bits,
                dynamic=qcfg.dynamic,
                group_size=qcfg.group_size,
                desc_act=qcfg.desc_act,
                sym=qcfg.sym,
                backend=backend,
                format=format_code,
                quant_method=export_quant_method,
                device=device,
                pack_dtype=qcfg.pack_dtype,
            )

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions", "multimodal_max_length"]
        config_seq_len = find_config_seq_len(model_config, seq_len_keys)
        if config_seq_len is not None:
            model.seqlen = config_seq_len
        else:
            log.warn("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        if format_code != FORMAT.EXL3:
            # Any post-initialization that require device information, for example buffers initialization on device.
            model = gptqmodel_post_init(model, use_act_order=qcfg.desc_act, quantize_config=qcfg)

        model.eval()

        if backend == BACKEND.MLX:
            import tempfile
            try:
                from mlx_lm import load
                from mlx_lm.utils import save_config, save_model

                from ..utils.mlx import convert_gptq_to_mlx_weights, mlx_generate
            except ModuleNotFoundError as exception:
                raise type(exception)(
                    "GPT-QModel load mlx model required dependencies are not installed.",
                    "Please install via `pip install gptqmodel[mlx] --no-build-isolation`.",
                )

            with tempfile.TemporaryDirectory() as temp_dir:
                mlx_weights, mlx_config = convert_gptq_to_mlx_weights(model_id_or_path, model, qcfg.to_dict(), cls.lm_head)

                save_model(temp_dir, mlx_weights, donate_model=True)
                save_config(mlx_config, config_path=temp_dir + "/config.json")
                tokenizer.save_pretrained(temp_dir)

                model, _ = load(temp_dir)

                cls.generate = lambda _, **kwargs: mlx_generate(model=model, tokenizer=tokenizer, **kwargs)


        return cls(
            model,
            quantized=True,
            quantize_config=qcfg,
            tokenizer=tokenizer,
            qlinear_kernel=qlinear_kernel,
            load_quantized_model=True,
            trust_remote_code=trust_remote_code,
            model_local_path=model_local_path,
        )

    cls.from_quantized = from_quantized

    return cls
