# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import os
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List, Optional, Union

import torch
import transformers

from ..utils.structure import print_module_tree


if os.getenv('GPTQMODEL_USE_MODELSCOPE', 'False').lower() in ['true', '1']:
    try:
        from modelscope import snapshot_download
    except Exception:
        raise ModuleNotFoundError("env `GPTQMODEL_USE_MODELSCOPE` used but modelscope pkg is not found: please install with `pip install modelscope`.")
else:
    from huggingface_hub import snapshot_download

from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.modeling_utils import no_init_weights
from transformers.utils import is_flash_attn_2_available
from transformers.utils.generic import ContextManagers

from ..adapter.adapter import Adapter
from ..nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, METHOD, MIN_VERSION_WITH_V2
from ..utils.backend import BACKEND
from ..utils.importer import auto_select_device, normalize_device_device_map, select_quant_linear
from ..utils.logger import setup_logger
from ..utils.machete import _validate_machete_device_support
from ..utils.marlin import _validate_marlin_device_support
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


def get_model_local_path(pretrained_model_id_or_path, **kwargs):
    is_local = os.path.isdir(pretrained_model_id_or_path)
    if is_local:
        return os.path.normpath(pretrained_model_id_or_path)
    else:
        # Clone kwargs before modifying
        download_kwargs = kwargs.copy()
        download_kwargs.pop("attn_implementation", None)
        download_kwargs.pop("use_flash_attention_2", None)
        return snapshot_download(pretrained_model_id_or_path, **download_kwargs)


def ModelLoader(cls):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_id_or_path: str,
            quantize_config: QuantizeConfig,
            trust_remote_code: bool = False,
            dtype: [str | torch.dtype] = "auto",
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            device: Optional[Union[str, int]] = None,
            **model_init_kwargs,
    ):
        # quantization is unsafe with GIL=0 and torch.compile/graphs
        import torch._dynamo
        torch._dynamo.disable()

        load_start = time.perf_counter()

        # non-quantized models are always loaded into cpu
        cpu_device_map = {"": "cpu"}

        if quantize_config is None or not isinstance(quantize_config, QuantizeConfig):
            raise AttributeError("`quantize_config` must be passed and be an instance of QuantizeConfig.")

        quantize_config.calculate_bits_per_weight()

        if quantize_config.device is not None:
            if device is not None or device_map is not None:
                raise AttributeError("Passing device and device_map is not allowed when QuantizeConfig.device is set. Non-quantized model is always loaded as cpu. Please set QuantizeConfig.device for accelerator used in quantization or do not set for auto-selection.")

        if quantize_config.desc_act not in cls.supports_desc_act:
            raise ValueError(f"{cls} only supports desc_act={cls.supports_desc_act}, "
                             f"but quantize_config.desc_act is {quantize_config.desc_act}.")

        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{pretrained_model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs_version)

        model_local_path = get_model_local_path(pretrained_model_id_or_path, **model_init_kwargs)

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        model_init_kwargs["trust_remote_code"] = trust_remote_code

        config = AutoConfig.from_pretrained(model_local_path, **model_init_kwargs)

        atten_impl = model_init_kwargs.get("attn_implementation", None)

        if atten_impl is not None and atten_impl != "auto":
            log.info(f"Loader: overriding attn_implementation in config to `{atten_impl}`")
            config._attn_implementation = atten_impl

        # normalize and auto select quantization device is not passed
        if quantize_config.device is None:
            quantize_config.device = auto_select_device(None, None)
        else:
            quantize_config.device = normalize_device(quantize_config.device)

        if cls.require_dtype:
            dtype = cls.require_dtype

        if dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype):
            # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
            dtype = auto_dtype(config=config, device=quantize_config.device, quant_inference=False)

        if isinstance(dtype, torch.dtype) and getattr(config, "torch_dtype", None) != dtype:
            # Align config metadata with the dtype we will materialize weights in.
            config.torch_dtype = dtype

        # enforce some values despite user specified
        # non-quantized models are always loaded into cpu
        model_init_kwargs["device_map"] = cpu_device_map
        model_init_kwargs["dtype"] = dtype
        model_init_kwargs["_fast_init"] = cls.require_fast_init
        #model_init_kwargs["low_cpu_mem_usage"] = True

        cls.before_model_load(cls, load_quantized_model=False)
        from ..utils.hf import build_shell_model

        # XIELUActivation will use some weights when activation init, so can't use init_empty_weights
        if hasattr(config, "hidden_act") and config.hidden_act == "xielu":
            quantize_config.offload_to_disk = False

        if quantize_config.offload_to_disk:
            model = build_shell_model(cls.loader, config=config, **model_init_kwargs)
            model._model_init_kwargs = model_init_kwargs
            print_module_tree(model=model)

            # enable mmap with low_cpu_mem_usage
            turtle_spinner = log.spinner(title="Turtle model loading...", interval=0.1)
            try:
                turtle_model = cls.loader.from_pretrained(
                    model_local_path,
                    config=config,
                    low_cpu_mem_usage=True,
                    **model_init_kwargs,
                )
            finally:
                turtle_spinner.close()

            # TODO FIX ME...temp store model_init args
            turtle_model._model_init_kwargs = model_init_kwargs
            # print("actual turtle model-----------")
            # print_module_tree(model=turtle_model)
        else:
            print("loading model directly to CPU (not using meta device or turtle_model)-----------")
            model = cls.loader.from_pretrained(model_local_path, config=config, **model_init_kwargs)
            model._model_init_kwargs = model_init_kwargs
            print_module_tree(model=model)

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

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id_or_path, trust_remote_code=trust_remote_code)

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

        # normalized device + device_map into single device
        normalized_device = device if device_map is None else None  # let device_map dictate placement when present
        device = normalize_device_device_map(normalized_device, device_map)

        # TODO need to normalize backend and others in a unified api
        if isinstance(backend, str):
            backend =  (backend)
        device = auto_select_device(device, backend)

        if backend == BACKEND.VLLM:
            import os

            # to optimize vllm inference, set an environment variable 'VLLM_ATTENTION_BACKEND' to 'FLASHINFER'.
            os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'

        if backend == BACKEND.TRITON:
            from ..nn_modules.qlinear.tritonv2 import TRITON_AVAILABLE, TRITON_INSTALL_HINT
            if not TRITON_AVAILABLE:
                raise ValueError(TRITON_INSTALL_HINT)

        """load quantized model from local disk"""
        if cls.require_trust_remote_code and not trust_remote_code:
            raise ValueError(
                f"{model_id_or_path} requires trust_remote_code=True. Please set trust_remote_code=True to load this model."
            )

        check_versions(cls, cls.require_pkgs_version)

        model_local_path = get_model_local_path(model_id_or_path, **kwargs)

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        attn_implementation = kwargs.pop("attn_implementation", None)

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
        )

        if cls.require_dtype:
            dtype = cls.require_dtype

        if dtype is None or dtype == "auto" or not isinstance(dtype, torch.dtype) :
            # TODO FIX ME for `dynamic`, non-quantized modules should be in native type
            dtype = auto_dtype(config=config, device=device, quant_inference=True)

        if isinstance(dtype, torch.dtype) and getattr(config, "torch_dtype", None) != dtype:
            # Ensure flash attention kernels see an explicit dtype instead of relying on defaults.
            config.torch_dtype = dtype

        qcfg = QuantizeConfig.from_pretrained(model_local_path, **cached_file_kwargs, **kwargs)

        if qcfg.quant_method == METHOD.AWQ and qcfg.format in [FORMAT.GEMV_FAST]:
            # GEMV_FAST only supports torch.float16
            log.info("Loading Quantized Model: Auto fix `dtype` to `torch.float16`")
            dtype = torch.float16

        if backend == BACKEND.EXLLAMA_EORA:
            # EXLLAMA_EORA only supports torch.float16
            log.info("Loading Quantized Model: Auto fix `dtype` to `torch.float16`")
            dtype = torch.float16

        # inject adapter into qcfg
        if adapter is not None:
            qcfg.adapter = adapter

        qcfg.calculate_bits_per_weight()

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)

        if backend == BACKEND.VLLM or backend == BACKEND.SGLANG:
            if backend == BACKEND.VLLM:
                if qcfg.format != FORMAT.GPTQ and qcfg.format != FORMAT.GEMM:
                    raise ValueError(f"{backend} backend only supports FORMAT.GPTQ or FORMAT.GEMM: actual = {qcfg.format}")
            elif backend == BACKEND.SGLANG:
                if qcfg.format != FORMAT.GPTQ:
                    raise ValueError(f"{backend} backend only supports FORMAT.GPTQ: actual = {qcfg.format}")

            if backend == BACKEND.VLLM:
                from ..utils.vllm import load_model_by_vllm, vllm_generate

                model = load_model_by_vllm(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )

                model.config = model.llm_engine.model_config
                model.device = model.llm_engine.vllm_config.device_config.device

                cls.generate = lambda self, **kwargs: vllm_generate(self.model, **kwargs)

            elif backend == BACKEND.SGLANG:
                from ..utils.sglang import load_model_by_sglang, sglang_generate

                model, hf_config = load_model_by_sglang(
                    model=model_local_path,
                    trust_remote_code=trust_remote_code,
                    dtype=torch.float16,
                    **kwargs,
                )
                model.config = hf_config
                cls.generate = lambda self, **kwargs: sglang_generate(self.model, **kwargs)
            return cls(
                model,
                quantized=True,
                quantize_config=qcfg,
                tokenizer=tokenizer,
                qlinear_kernel=None,
                load_quantized_model=True,
                trust_remote_code=trust_remote_code,
                model_local_path=model_local_path,
            )

        if qcfg.format == FORMAT.MARLIN:
            # format marlin requires marlin kernel
            if backend not in [BACKEND.MARLIN, BACKEND.MARLIN_FP16] and backend != BACKEND.AUTO:
                raise TypeError(f"FORMAT.MARLIN requires BACKEND.AUTO or BACKEND.MARLIN: actual = `{backend}`.")
            backend = BACKEND.MARLIN

        # marlin_compatible = False if backend == BACKEND.IPEX else _validate_marlin_device_support()
        # check for marlin compat for cuda device only
        # if backend not in [BACKEND.MARLIN, BACKEND.MARLIN_FP16] and device == DEVICE.CUDA:
        #     unsupported = _validate_marlin_compatibility(qcfg)
        #     if unsupported is None and marlin_compatible:
        #         logger.info(
        #             "Hint: Model is compatible with the Marlin kernel. Marlin is optimized for batched inference on Nvidia GPU: `model = GPTQModel.load(..., backend=BACKEND.MARLIN)`."
        #         )

        if qcfg.format == FORMAT.BITBLAS:
            # format bitblas requires bitblas kernel
            if backend != BACKEND.BITBLAS and backend != BACKEND.AUTO:
                raise TypeError(f"FORMAT.BITBLAS requires BACKEND.AUTO or BACKEND.BITBLAS: actual = `{backend}`.")
            backend = BACKEND.BITBLAS

        if backend == BACKEND.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        possible_model_basenames = [
            f"gptq_model-{qcfg.bits}bit-{qcfg.group_size}g",
            "model",
        ]

        extensions = [".safetensors"]

        model_local_path = str(model_local_path)

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

        qcfg.runtime_format = qcfg.format

        model_save_name = resolved_archive_file  # In case a model is sharded, this would be `model.safetensors.index.json` which may later break.

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False

        init_contexts = [no_init_weights()]

        with (ContextManagers(init_contexts)):
            cls.before_model_load(cls, load_quantized_model=True)

            if config.architectures:
                model_class = getattr(transformers, config.architectures[0], None)
                if model_class is not None and hasattr(model_class, "_supports_flash_attn_2"):
                    supports_flash_attn = model_class._supports_flash_attn_2
                else:
                    supports_flash_attn = None
            else:
                supports_flash_attn = None

            args = {}
            if supports_flash_attn and device in [DEVICE.CUDA, DEVICE.ROCM]:
                if ATTN_IMPLEMENTATION in kwargs:
                    args[ATTN_IMPLEMENTATION] = kwargs.pop(ATTN_IMPLEMENTATION, None)
                elif is_flash_attn_2_available():
                    args = {ATTN_IMPLEMENTATION: "flash_attention_2"}
                    log.info("Loader: Auto enabling flash attention2")

            model = cls.loader.from_config(
                config, trust_remote_code=trust_remote_code, dtype=dtype, **args
            )
            model.checkpoint_file_name = model_save_name

            # Get the first layer to determine layer type
            layers, _ = get_module_by_name_prefix(model, cls.extract_layers_node())

            layers[0]

            modules = find_modules(model)
            ignore_modules = [cls.lm_head] + cls.get_base_modules(model)

            for name in list(modules.keys()):
                # allow loading of quantized lm_head
                if qcfg.lm_head and name == cls.lm_head:
                    continue

                if not any(name.startswith(prefix) for prefix in cls.extract_layers_node()) or any(name.startswith(ignore_module) for ignore_module in ignore_modules) or all(
                        not name.endswith(ignore_module) for sublist in cls.simple_layer_modules(config, qcfg) for ignore_module in sublist
                ):
                    # log non-lm-head quantized modules only
                    if name is not cls.lm_head:
                        log.info(f"The layer {name} is not quantized.")
                    del modules[name]

            preload_qlinear_kernel = make_quant(
                model,
                qcfg=qcfg,
                quant_result=modules,
                backend=backend,
                lm_head_name=cls.lm_head,
                device=device,
            )

        if isinstance(device_map, str) and device_map not in [
                "auto",
                "balanced",
                "balanced_low_0",
                "sequential",
            ]:
                raise ValueError(
                    "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                    "'sequential'."
                )



        import torch

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
            in_emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
            assign(in_emb, device_ids[0])

            # Alternating layers
            for i, layer in enumerate(layers):
                gpu = device_ids[i % num_gpus]
                assign(layer, gpu)

            # Ignored modules - skip input embeddings to avoid overriding GPU 0 assignment
            for mod in ignore_modules:
                if in_emb is not None and mod is in_emb:
                    continue  # Skip input embedding to preserve GPU 0 assignment
                assign(mod, device_ids[-1])

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
                n for n, _ in model.named_parameters()
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

            # optional logging for debug
            log.info(f"Loader: Built map across {num_gpus} GPU(s), "
                  f"{len(device_map)} entries. First 8: {list(device_map.items())[:8]}")

            return device_map

        log.info(f"Loader: device = {device}")
        layers, _ = get_module_by_name_prefix(model, cls.extract_layers_node())
        num_gpus = 1
        if device is DEVICE.CUDA:
            num_gpus = torch.cuda.device_count()
        elif device is DEVICE.XPU:
            num_gpus = torch.xpu.device_count()
        device_map = build_layerwise_device_map(model, device, layers, ignore_modules, num_gpus)
        log.info(f"Loader: device_map = {device_map}")

        load_checkpoint_in_model = True
        # compat: runtime convert checkpoint gptq(v1) to gptq_v2 format
        if qcfg.format in [FORMAT.GPTQ, FORMAT.GEMM]:
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

            if qcfg.format == FORMAT.GPTQ:
                # validate sym=False v1 loading needs to be protected for models produced with new v2 format codebase
                if not qcfg.sym and not qcfg.is_quantized_by_v2():
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

        if backend == BACKEND.MACHETE:
            if is_sharded:
                raise ValueError(
                    "Format: The loading of sharded checkpoints with Machete is currently not supported."
                )
            if not _validate_machete_device_support():
                raise ValueError(
                    f"Kernel: Machete kernel requires compute capability >= 9.0. Detected capability: {torch.cuda.get_device_capability()}"
                )

        if backend in [BACKEND.MARLIN, BACKEND.MARLIN_FP16] and (
                preload_qlinear_kernel == ExllamaV2QuantLinear or qcfg.format == FORMAT.MARLIN):
            if is_sharded:
                raise ValueError(
                    "Format: The loading of sharded checkpoints with Marlin is currently not supported."
                )
            if not _validate_marlin_device_support():
                raise ValueError(
                    f'Kernel: Marlin kernel does not support this gpu with compute capability of `{torch.cuda.get_device_capability()}`. Please do not use `back=BACKEND.MARLIN`.'
                )

            # Validate the model can run in Marlin.
            if dtype != torch.float16:
                raise ValueError("Marlin kernel requires dtype=torch.float16.")


        if backend == BACKEND.BITBLAS:
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
        if load_checkpoint_in_model and backend not in [BACKEND.MACHETE, BACKEND.MARLIN, BACKEND.MARLIN_FP16, BACKEND.BITBLAS]:
            load_checkpoint_in_model_then_tie_weights(
                model,
                dtype=dtype,
                # This is very hacky but works due to https://github.com/huggingface/accelerate/blob/bd72a5f1a80d5146554458823f8aeda0a9db5297/src/accelerate/utils/modeling.py#L292
                checkpoint=model_save_name,
                device_map=device_map,
                # offload_state_dict=True,
                # offload_buffers=True,
            )

        # TODO: Why are we using this custom function and not dispatch_model?
        model = simple_dispatch_model(model, device_map)

        qlinear_kernel = select_quant_linear(
            bits=qcfg.bits,
            dynamic=qcfg.dynamic,
            group_size=qcfg.group_size,
            desc_act=qcfg.desc_act,
            sym=qcfg.sym,
            backend=backend,
            format=qcfg.format,
            quant_method=qcfg.quant_method,
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
                    "GPTQModel load mlx model required dependencies are not installed.",
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
