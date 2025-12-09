# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from itertools import count
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type, Union

import torch
import torch._dynamo
import torch.nn as nn
from tokenicer import Tokenicer
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    modeling_utils,
)


try:  # Optional dependency for huggingface datasets support
    from datasets import Dataset as HFDataset
    from datasets import IterableDataset as HFIterableDataset
except Exception:  # pragma: no cover - datasets may not be installed
    HFDataset = None
    HFIterableDataset = None

from .. import DEVICE_THREAD_POOL
from ..adapter.adapter import Adapter
from ..nn_modules.qlinear import BaseQuantLinear
from ..nn_modules.qlinear.lookahead import configure_default_lookahead
from ..nn_modules.qlinear.torch import TorchQuantLinear
from ..quantization import QuantizeConfig
from ..quantization.config import FORMAT, METHOD, QUANTIZE_BLACK_LIST, VRAMStrategy, dynamic_get
from ..quantization.rotation.rotation import fuse_layer_norms, rotate_model
from ..utils.backend import BACKEND
from ..utils.calibration import prepare_calibration_dataset
from ..utils.device import get_device
from ..utils.hf import autofix_hf_model_config
from ..utils.importer import select_quant_linear
from ..utils.logger import QuantizationRegionTimer, setup_logger
from ..utils.model import MODALITY, find_modules, get_module_by_name_prefix, move_to
from ..utils.structure import alias_from_turtle_for_submodule
from ..utils.torch import TORCH_HAS_COMPILE, torch_compile
from ._const import (
    CPU,
    DEFAULT_MAX_SHARD_SIZE,
    DEVICE,
    EXPERT_INDEX_PLACEHOLDER,
    META,
)
from .loader import ModelLoader
from .writer import ModelWriter


if TYPE_CHECKING:
    try:
        from datasets import Dataset as HFDatasetType
        from datasets import IterableDataset as HFIterableDatasetType
    except Exception:  # pragma: no cover - optional dependency
        HFDatasetType = HFIterableDatasetType = object


class _ClassPropertyDescriptor:
    def __init__(self, fget, fset=None):
        self.fget = fget

    def __get__(self, instance, owner=None):
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


def generate_node_for_awq_scaling(inp, prev_op, module_kwargs, nodes_size, subset, module2inspect):
    n = {
        "prev_op": prev_op,
        "layers": subset,
        "inp": inp,
    }
    if nodes_size == 0:
        # Only the first node needs kwargs
        n["kwargs"] = module_kwargs

    if module2inspect is not None:
        n["module2inspect"] = module2inspect

    return n, None

def check_support_param_buffer_assignment(*args, **kwargs):
    return False


def apply_module_tree_override(module_tree, override):
    """
    Recursively find the corresponding key of override in module_tree and override it.
    """
    if isinstance(module_tree, dict) and isinstance(override, dict):
        for k, v in override.items():
            if k in module_tree and isinstance(module_tree[k], (dict, list)) and isinstance(v, (dict, list)):
                module_tree[k] = apply_module_tree_override(module_tree[k], v)
            else:
                module_tree[k] = v
    elif isinstance(module_tree, list) and isinstance(override, list):
        for o in override:
            if isinstance(o, dict):
                for b in module_tree:
                    if isinstance(b, dict):
                        apply_module_tree_override(b, o)
    return module_tree


NOT_QUANTIZE_FLAG = ":!"
CAPTURE_ONLY_FLAG = ":?"
NON_QUANTIZE_FLAGS = (NOT_QUANTIZE_FLAG, CAPTURE_ONLY_FLAG)


# Fix cpu memory leak.
# See https://github.com/huggingface/transformers/issues/34366
modeling_utils.check_support_param_buffer_assignment = check_support_param_buffer_assignment

log = setup_logger()

class BaseQModel(nn.Module):
    # name of lm_head
    lm_head: str = "lm_head"

    # a tree node of all the roots that contain quantizable modules
    module_tree: List[str] = None
    # Override module_tree according to different QUANT_METHOD
    module_tree_overrides: dict[METHOD, List[str]] = None

    # Strict=True -> all layer_modules must exists in model
    # Some models (deepseek2-lite) dynamically create lora modules based on config.rank
    layer_modules_strict = True

    pre_lm_head_norm_module: str = None

    # awq scaling optimizations requires some modules within same subset to strictly match the shape of previous module
    # list modules where they must match the shape of previous module in execution to consider for scaling optimization
    awq_scale_optimize_shape_dependent_modules: List[str] = None

    # some models require trust_remove_code = True (dbrx_converted)
    require_trust_remote_code = None
    # some models require transformer version(internalm require '<=4.42.2')
    require_pkgs_version: Optional[List[str]] = None
    # some models require a specific dtype, such as float16
    require_dtype: Optional[str|torch.dtype] = None
    require_fast_init: bool = True

    # some models require Processor? For example, Qwen2VLImageProcessor.
    require_load_processor = False

    # TODO: use a better name and what if the value is not at the config root?
    # allow dynamic expert n-count layer extraction
    # so moe model defs do not need to write out 64 layers if expert size is 64 (Qwen2Moe)
    # usage: set to property in model.config that holds this int value: total number of experts
    dynamic_expert_index: Optional[str] = None

    # some models require a different model loader, such as mllama which uses AutoModelForPreTraining
    loader = AutoModelForCausalLM

    # monkey patch api for trust_remote_code=True models that have broken transformer compat
    require_monkeypatch = False

    # VRAM strategy support list
    supported_vram_strategies: List[VRAMStrategy] = [VRAMStrategy.EXCLUSIVE, VRAMStrategy.BALANCED]

    # some models have broken attention mask codes so we need to only use batch 1 with no masks
    support_batch_quantize = True

    # allow models to define optional notes that output messages to users that want to use this model
    # list of supported keys: [ "notes" = print the notes value on model load ]
    info: Dict[str, str] = {}

    # Some models have optional layers that are not loaded or supported by HF so even when they exist in the original
    # model, they are not properly saved on save(). GLM 4.5/4.6 (air) with MTP layers is such example.
    # List the `dangling` optional tensor files here, and we will merge them in on model.save()
    out_of_model_tensor_files: Optional[List[str]] = None

    supports_desc_act = [True, False]

    modality: List[MODALITY] = [MODALITY.TEXT]

    quant_override_files: Dict[str, Union[str | Dict[str, Any]]] = {}

    server = None

    support_offload_to_disk = True

    moe_expert_module_name_prefixes = [".expert"]

    ATTENTION_MASKS_DTYPE = torch.bool # default to bool

    ATTENTION_MASKS_REQUIRED_FOR_INPUT: bool = False

    INPUT_EMBEDDING_EXTRA_ARGS = None

    def __init__(
        self,
        model: PreTrainedModel,
        quantized: bool,
        quantize_config: QuantizeConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        qlinear_kernel: nn.Module = None,
        load_quantized_model: bool = False,
        trust_remote_code: bool = False,
        model_local_path: str = None,
        # turtle model is a sympathetic model used to reduce cpu ram usage
        # during quantization stage.
        turtle_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__()

        quant_method = quantize_config.quant_method
        # override module_tree if need
        if self.module_tree_overrides is not None and self.module_tree_overrides.get(quant_method) is not None:
            log.info(f'Module Tree: overridden by METHOD.{quant_method.upper()}')
            # setting cls.module_tree
            type(self).module_tree = apply_module_tree_override(self.module_tree, self.module_tree_overrides[quant_method])

        if type(self).module_tree is None:
            type(self).module_tree = self._auto_detect_module_tree(model, quant_method)

        # If module_tree is still None after auto-detection, raise an error indicating unsupported model type
        if type(self).module_tree is None:
            raise ValueError(f"Unsupport model_type {model.config.model_type}, and failed to auto-detect module tree for model {model}")


        # record configuration early so model lifecycle hooks can rely on them
        self.compiled = False  # set to True while compile() is triggered successfully
        self.quantized = quantized
        self.load_quantized_model = load_quantized_model
        self.qlinear_kernel = qlinear_kernel
        self.trust_remote_code = trust_remote_code
        self.model_local_path = model_local_path
        self.quantize_config = quantize_config
        self.quant_region_timer = QuantizationRegionTimer(logger=log)
        self._turtle_reload_threshold_bytes = self._resolve_turtle_reload_threshold()
        self._turtle_reload_accum_bytes = 0
        self._turtle_materialized_ids: Set[int] = set()

        self.processor: ProcessorMixin = None

        self.model = self.after_model_load(model, load_quantized_model=load_quantized_model)
        self.turtle_model = turtle_model

        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")
            self.model.tokenizer = self.tokenizer.tokenizer # helpful for CI tests
        else:
            self.tokenizer = tokenizer # TODO none?
            self.model.tokenizer = tokenizer # helpful for CI tests # TODO none?

        # auto-fix model config erors
        if isinstance(self.model, PreTrainedModel):
            autofix_hf_model_config(self.model, path=model_local_path)

        self._turtle_lock = threading.RLock()

        # compat: state to assist in checkpoint_format gptq(v1) to gptq_v2 conversion
        # stores all per-layer quant stats such as avg loss and processing time
        self.quant_log = []

        if self.require_load_processor:
            self.processor = AutoProcessor.from_pretrained(model_local_path)

        # apply patching of broken trust_remote_code models here
        if self.require_monkeypatch:
            self.monkey_patch()

        # hack: circular import
        from ..adapter.adapter import Lora

        # check adapter load and print info so users knows lora(s) are applied
        if isinstance(self.quantize_config.adapter, Lora):
            loaded_loras = 0
            qmodules = find_modules(self.model, layers=[BaseQuantLinear])
            for name, m in qmodules.items():
                if all(hasattr(m.adapter, name) for name in Lora.parameter_keys()):
                    loaded_loras += 1

            log.info(f"Adapter: `{loaded_loras}` EoRA/Lora adapters loaded for `{len(qmodules)}` modules.")

        # print kernel info:
        log.info(f"Kernel: loaded -> `[{', '.join(cls.__name__ for cls in self.kernels())}]`")

        self._auto_configure_lookahead()

    @classmethod
    def extract_layers_node(cls):
        """
        Given a module_tree structure, return the layers_node string.
        It concatenates everything up to (but not including) the first "#" with '.'.
        Example:
            ["model", "layers", "#", {...}] -> ["model.layers"]
        """

        prefix_parts = []
        for node in cls.module_tree:
            if node == "#":
                break
            if isinstance(node, str):
                prefix_parts.append(node)
            else:
                break  # stop if unexpected nested structure

        return [".".join(prefix_parts)] if prefix_parts else []

    @classmethod
    def build_moe_modules_if_need(cls, model_config, layer_modules, is_awq_quantize: bool = False):
        # MoE models
        if model_config is not None and cls.dynamic_expert_index is not None:
            num_experts = cls.get_num_experts(model_config)

            moe_simple = []
            capture_only_modules = None
            for names in layer_modules:
                moe_simple.append([])

                has_expert = any(EXPERT_INDEX_PLACEHOLDER in n for n in names)
                has_capture_only = all(CAPTURE_ONLY_FLAG in n for n in names)
                if has_capture_only:
                    capture_only_modules = list(names)
                    continue

                if not has_expert:
                    moe_simple[-1].extend(names)
                    continue

                if is_awq_quantize:
                    # AWQ Required
                    # result like: ['mlp.experts.0.gate_proj', 'mlp.experts.0.up_proj', 'mlp.experts.1.gate_proj', 'mlp.experts.1.up_proj', ...]
                    for index in range(num_experts):
                        for n in names:
                            if EXPERT_INDEX_PLACEHOLDER in n:
                                moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))
                    # added 'mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj'
                    for n in names:
                        if EXPERT_INDEX_PLACEHOLDER not in n:
                            moe_simple[-1].append(n)
                    # Currently, only need to add `capture_only_modules` to `['mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']`
                    # or ['mlp.shared_expert.gate_proj', 'mlp.shared_expert.up_proj', 'mlp.experts.#.gate_proj', 'mlp.experts.#.up_proj']
                    add_capture_only_module = len(names) == (4 if any("shared_expert" in n for n in names) else 2)
                    if add_capture_only_module and capture_only_modules:
                        # Extend all elements in capture_only_modules
                        moe_simple[-1].extend(capture_only_modules)
                else:
                    # result like: ['mlp.experts.0.gate_proj', 'mlp.experts.1.gate_proj', 'mlp.experts.0.up_proj', 'mlp.experts.1.up_proj', ...]
                    for n in names:
                        for index in range(num_experts):
                            moe_simple[-1].append(n.replace(EXPERT_INDEX_PLACEHOLDER, str(index)))

            return moe_simple

        return layer_modules

    @classmethod
    def get_num_experts(cls, model_config):
        if hasattr(model_config, "text_config"):
            num_experts = getattr(model_config.text_config, cls.dynamic_expert_index)
        elif hasattr(model_config, "thinker_config"):
            num_experts = getattr(model_config.thinker_config.text_config, cls.dynamic_expert_index)
        else:
            num_experts = getattr(model_config, cls.dynamic_expert_index)
        return num_experts

    @classmethod
    def filter_not_quantize_module(cls, layer_modules, quantize_config):
        layer_modules = [
            [name for name in block if NOT_QUANTIZE_FLAG not in name]
            for block in layer_modules
        ]
        layer_modules = [block for block in layer_modules if block]  # 去掉空 block

        if getattr(quantize_config, "dynamic", None):
            new_layer_modules = []
            for modules in layer_modules:
                filtered = [
                    m for m in modules
                    if dynamic_get(quantize_config.dynamic, module_name=m) is not False
                ]
                if filtered:
                    new_layer_modules.append(filtered)
            layer_modules = new_layer_modules

        return layer_modules

    # Inside each `LlamaDecoderLayer` layer are many internal modules
    # List them in the order executed in model forward() code
    # Many models have same execution order of: attention (q_k_v) projection, attention (output) projection, mlp (n) projections
    @classmethod
    def simple_layer_modules(cls, model_config, quantize_config, is_awq_quantize: bool = False, include_capture_only: bool = False):
        layer_modules = cls.build_layer_modules(cls.module_tree, include_capture_only=include_capture_only)

        layer_modules = cls.build_moe_modules_if_need(model_config, layer_modules, is_awq_quantize)

        layer_modules = cls.filter_not_quantize_module(layer_modules, quantize_config)

        # print(f"simple_layer_modules layer_modules: {layer_modules}")
        return layer_modules

    @classmethod
    def full_layer_modules(cls, model_config=None, is_awq_quantize: bool = False, include_capture_only: bool = False):
        full = cls.build_layer_modules(cls.module_tree, include_capture_only=include_capture_only)
        full = cls.build_moe_modules_if_need(model_config, full, is_awq_quantize)
        # print(f"full layer_modules: {full}")
        return full

    def prepare_dataset(
        self,
        calibration_dataset: Union[
            List[Dict[str, Union[List[int], torch.LongTensor]]],
            List[str],
            List[List[int]],
            "HFDatasetType",
            "HFIterableDatasetType",
        ],
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        calibration_data_min_length: int = 10,
        calibration_concat_separator: Optional[str] = None,
    ):
        return prepare_calibration_dataset(
            self,
            calibration_dataset=calibration_dataset,
            calibration_dataset_concat_size=calibration_dataset_concat_size,
            calibration_dataset_sort=calibration_dataset_sort,
            batch_size=batch_size,
            calibration_data_min_length=calibration_data_min_length,
            calibration_concat_separator=calibration_concat_separator,
            logger=log,
        )

    def quantize(
        self,
        calibration: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
        calibration_concat_size: Optional[int] = None,
        calibration_sort: Optional[str] = "desc",  # valid values are asc, desc, shuffle
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        backend: Optional[BACKEND] = BACKEND.AUTO,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter = None,
        adapter_calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]] = None,
        # minimum length of calibration data, default is 10
        calibration_data_min_length: int = 10,
        calibration_concat_separator: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        if self.quantized:
            raise EnvironmentError("quantize() is called a model that is already quantized")

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.reset()

        self._turtle_reload_accum_bytes = 0
        self._turtle_materialized_ids = set()

        if self.quantize_config.quant_method in QUANTIZE_BLACK_LIST:
            raise ValueError(
                f"Unsupported quantization operation for quant method: {self.quantize_config.quant_method}"
            )

        if not self.support_batch_quantize:
            log.warn("Quantize: batch_size overridden by model class definition to `disabled`")
            batch_size = 1 # but actually disabled

        if self.quantize_config.format == FORMAT.MARLIN:
            raise ValueError(
                "FORMAT.MARLIN is deprecated for quantization. Please switch to FORMAT.GPTQ. GPTQMOdel will auto-use Marlin kernel for accelerated inference for FORMAT.GPTQ."
            )

        if self.quantize_config.quant_method == METHOD.AWQ:
            if self.quantize_config.format == FORMAT.GEMV_FAST:
                # AWQ GEMV_FAST only supports pack_dtype is torch.int16
                log.info("Quantize Model: Auto fix `pack_dtype` to `torch.int16`")
                self.quantize_config.pack_dtype = torch.int16
            elif self.quantize_config.format == FORMAT.MARLIN:
                # AWQ MARLIN only supports zero_point is false
                log.info("Quantize Model: Auto fix `zero_point` to `False`")
                self.quantize_config.zero_point = False

        if self.support_batch_quantize is False:
            batch_size = 1
            log.warn("Batch quantization is not supported for this model. Setting batch_size to 1.")

        requested_backend = backend
        if isinstance(requested_backend, str):
            requested_backend = BACKEND(requested_backend.lower())

        preferred_backend = requested_backend
        if preferred_backend in (None, BACKEND.AUTO):
            preferred_backend = BACKEND.TORCH

        # Validate quant linear before quantization starts
        _ = select_quant_linear(
            bits=self.quantize_config.bits,
            dynamic=self.quantize_config.dynamic,
            group_size=self.quantize_config.group_size,
            desc_act=self.quantize_config.desc_act,
            sym=self.quantize_config.sym,
            backend=preferred_backend,
            format=self.quantize_config.format,
            quant_method=self.quantize_config.quant_method,
            device=DEVICE(self.quantize_config.device),
            pack=True,
            pack_dtype=self.quantize_config.pack_dtype,
        )

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        if self.quantize_config.format == FORMAT.BITBLAS:
            from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
            if BITBLAS_AVAILABLE is False:
                raise ValueError(BITBLAS_INSTALL_HINT)

        # overwrite quantize_config.adapter
        if adapter is not None:
            self.quantize_config.adapter = adapter

        from ..adapter.adapter import Lora
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper

        # has lora process
        needs_lora = isinstance(self.quantize_config.adapter, Lora)

        args = {
            "tokenizer": self.tokenizer,
            "qcfg": self.quantize_config,
            "calibration": calibration,
            "prepare_dataset_func": self.prepare_dataset,
            "calibration_concat_size": calibration_concat_size,
            "calibration_sort": calibration_sort,
            "calibration_concat_separator": calibration_concat_separator,
            "batch_size": batch_size,
            "calculate_w_wq_diff": needs_lora,  # lora needs original w - wq delta
        }

        self.qlinear_kernel = select_quant_linear(
                bits=self.quantize_config.bits,
                group_size=self.quantize_config.group_size,
                desc_act=self.quantize_config.desc_act,
                sym=self.quantize_config.sym,
                pack=True,
                dynamic=self.quantize_config.dynamic,
                device=self.quantize_config.device,
                pack_dtype=self.quantize_config.pack_dtype,
                multi_select=False,
                backend=preferred_backend,
                format=self.quantize_config.format,
                quant_method=self.quantize_config.quant_method,
            )

        # rotate model
        if self.quantize_config.rotation:
            from gptqmodel.models.definitions.llama import LlamaQModel
            from gptqmodel.models.definitions.qwen2 import Qwen2QModel
            if not isinstance(self, (LlamaQModel, Qwen2QModel)):
                raise ValueError(f"rotation only supports: llama/qwen2 model, "
                                    f"current model is {self.__class__.__name__}")

            if self.model.config.tie_word_embeddings:
                log.info("Rotation requires word embeddings to be untied. Untying.")
                self.model.config.tie_word_embeddings = False
                lm_head, _ = get_module_by_name_prefix(self.model, self.lm_head)
                lm_head.weight = nn.Parameter(lm_head.weight.data.clone())

            module_name_args = {
                "layers_node": self.extract_layers_node(),
                "lm_head_name": self.lm_head
            }
            self.model = fuse_layer_norms(model=self.model,
                                            pre_lm_head_norm_module_name=self.pre_lm_head_norm_module,
                                            **module_name_args)

            # MPS does not support float64.
            rotation_device = self.quantize_config.device if self.quantize_config.device != DEVICE.MPS else DEVICE.CPU
            self.model, _ = rotate_model(model=self.model, rotate_mode=self.quantize_config.rotation,
                                            device=rotation_device, **module_name_args)

        # init processor with default GPTQ processor
        from ..looper.tensorparallel_weight_processor import TensorParallelWeightProcessor

        if self.quantize_config.quant_method == METHOD.QQQ:
            from ..looper.qqq_processor import QQQProcessor

            quantize_processor = [
                TensorParallelWeightProcessor(**args),
                QQQProcessor(**args),
            ]
        elif self.quantize_config.quant_method == METHOD.AWQ:
            from ..looper.awq_processor import AWQProcessor

            os.environ["AWQ_BATCH_SIZE"] = str(batch_size)

            # if self.model.config.model_type not in AWQ_CAUSAL_LM_MODEL_MAP.keys():
            #     raise TypeError(f"{self.model.config.model_type} isn't supported yet.")

            awq_args = dict(args)
            awq_args["gptq_model"] = self
            awq_args["model"] = self.model
            awq_args["batch_size"] = batch_size

            quantize_processor = [
                TensorParallelWeightProcessor(**args),
                AWQProcessor(**awq_args),
            ]
        else:
            from ..looper.gptq_processor import GPTQProcessor

            quantize_processor = [
                TensorParallelWeightProcessor(**args),
                GPTQProcessor(**args),
            ]

        if self.quantize_config.gptaq is True:
            from ..looper.native_processor import NativeProcessor

            # During the deepcopy process, self.prepare_dataset will be deeply copied along with self. However,
            # self has a threading.RLock() , which is not serializable.
            args_to_copy = {k: v for k, v in args.items() if k != "prepare_dataset_func"}
            args_clone = copy.deepcopy(args_to_copy)
            args_clone["prepare_dataset_func"] = args["prepare_dataset_func"]

            args_clone.pop("calculate_w_wq_diff", None)
            quantize_processor.insert(0, NativeProcessor(**args_clone))

        processors = quantize_processor
        # Append EoRA processor for lora adapter
        if needs_lora:
            processors.append(
                EoraProcessor(
                    tokenizer=self.tokenizer,
                    qcfg=self.quantize_config,
                    calibration=adapter_calibration_dataset if adapter_calibration_dataset is not None else calibration,
                    prepare_dataset_func=self.prepare_dataset,
                    calibration_concat_size=calibration_concat_size,
                    calibration_sort=calibration_sort,
                    calibration_concat_separator=calibration_concat_separator,
                    batch_size=batch_size,
                )
            )

        # prepare processor worker (looper)
        module_looper = ModuleLooper(self, processors=processors)

        result = module_looper.loop(
            backend=backend,
            fail_safe=self.quantize_config.fail_safe,
        )

        timer = getattr(self, "quant_region_timer", None)
        if timer is not None:
            timer.flush()

        return result

    def _eora_generate(
        self,
        # eora adapter generation needs config Lora(rank=1, path='lora.safetensors')
        adapter: Adapter,
        quantized_modules: Dict[str, TorchQuantLinear],
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        calibration_dataset_concat_size: Optional[int] = None,
        calibration_dataset_sort: Optional[str] = None,
        batch_size: int = 1,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        calibration_concat_separator: Optional[str] = None,
    ):
        if self.quantized:
            raise EnvironmentError("eora_generate() is called a model that is already quantized")

        # Use the provided tokenizer if one is passed to quantize()
        if tokenizer is not None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                # TODO FIX ME...this is a bug
                self.tokenizer = Tokenicer.load(tokenizer, trust_remote_code=self.trust_remote_code)
            else:
                raise ValueError(
                    f"Unsupported `tokenizer` type: Expected `PreTrainedTokenizerBase`, actual = `{type(tokenizer)}`.")

        from ..adapter.adapter import Lora
        from ..looper.dequantize_processor import DequantizeProcessor
        from ..looper.eora_processor import EoraProcessor
        from ..looper.module_looper import ModuleLooper
        from ..looper.tensorparallel_weight_processor import TensorParallelWeightProcessor

        self.quantize_config.adapter = adapter

        assert isinstance(self.quantize_config.adapter, Lora)

        # init processor with EoRA processor
        processors = [
            TensorParallelWeightProcessor(
                tokenizer=self.tokenizer,
                qcfg=self.quantize_config,
                calibration=calibration_dataset,
                prepare_dataset_func=self.prepare_dataset,
                calibration_concat_size=calibration_dataset_concat_size,
                calibration_sort=calibration_dataset_sort,
                calibration_concat_separator=calibration_concat_separator,
                batch_size=batch_size,
            ),
            DequantizeProcessor(
                quantized_modules=quantized_modules,
            ),
            EoraProcessor(
                tokenizer=self.tokenizer,
                qcfg=self.quantize_config,
                calibration=calibration_dataset,
                prepare_dataset_func=self.prepare_dataset,
                calibration_concat_size=calibration_dataset_concat_size,
                calibration_sort=calibration_dataset_sort,
                calibration_concat_separator=calibration_concat_separator,
                batch_size=batch_size,
            ),
        ]

        # prepare processor worker (looper)
        module_looper = ModuleLooper(model=self, processors=processors)

        module_looper.loop()

        self.eora_save(save_dir=adapter.path, model_save_dir=self.model_local_path)
        return

    def to(self, device: Union[str, torch.device]):
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
            return self
        else:
            raise f"{self.model.__class__.__name__} does not support the to() method"

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, inputs=None, **kwargs):
        with torch.inference_mode():
            # fix hf generate not applying correct pad token
            pad_token_id = kwargs.get("pad_token_id", None)
            if pad_token_id is None and self.tokenizer:
                kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)):
                if self.tokenizer is None:
                    raise ValueError("You passed in an `input` to `generate()` of type `str` but model is missing `model.tokenizer`. Please set `model.tokenizer = my_tokenizer`.")
                inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
                return self.model.generate(**inputs, **kwargs)

            return self.model.generate(inputs=inputs, **kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """shortcut for model.prepare_inputs_for_generation"""
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    # placeholder, noop, and alert users to correct static api
    def push_to_hub(self,
                    repo_id: str,
                    quantized_path: str,  # saved local directory path
                    private: bool = False,
                    exists_ok: bool = False,  # set to true if repo already exists
                    token: Optional[str] = None):

        log.error("`push_to_hub()` api cannot be used on the model instance. Please use `GPTQModel.push_to_hub()` static api instead.")

    def save(
            self,
            save_dir: str,
            safetensors_metadata: Optional[Dict[str, str]] = None,
            max_shard_size: Optional[Union[int, str]] = DEFAULT_MAX_SHARD_SIZE,
            meta_quantizer: Optional[str] = None,
            eora_path: Optional[str] = None,
            **kwargs,
    ):
        timer = getattr(self, "quant_region_timer", None)
        start_time = time.perf_counter() if timer else None

        try:
            if self.quantized:
                # Safetensors is unable to save tied weights, so we untie them here. Reference: https://github.com/huggingface/safetensors/issues/202
                #untie_weights(self.model)

                self.save_quantized(
                    save_dir=save_dir,
                    safetensors_metadata=safetensors_metadata,
                    max_shard_size=max_shard_size,
                    meta_quantizer=meta_quantizer,
                    eora_path=eora_path)

                # overwrite quant_override_files
                for name, value in self.quant_override_files.items():
                    json_path = os.path.join(save_dir, name)
                    with open(json_path, "w", encoding="utf-8") as f:
                        if isinstance(value, str):
                            f.write(value)
                        else:
                            f.write(json.dumps(value))
            else:
                self.save_pretrained(save_dir=save_dir, **kwargs)
        finally:
            if timer is not None and start_time is not None:
                try:
                    target = os.path.abspath(save_dir)
                except (TypeError, ValueError, OSError):
                    target = str(save_dir)
                timer.record(
                    "model_save",
                    time.perf_counter() - start_time,
                    source=target,
                )
                timer.flush()


    # returns all the loaded qlinear types, returns empty [] if non-found
    def kernels(self) -> List[Type[BaseQuantLinear]]:
        if not isinstance(self.model, nn.Module):
            return []
        loaded_kernels = set()
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for k, v in modules.items():
            loaded_kernels.add(v.__class__)

        return list(loaded_kernels)

    def _auto_configure_lookahead(self) -> None:
        if not isinstance(self.model, nn.Module):
            return

        quant_modules = [module for module in self.model.modules() if isinstance(module, TorchQuantLinear)]
        if not quant_modules:
            return

        if not any(getattr(module, "_lookahead_enabled", False) for module in quant_modules):
            return

        configure_default_lookahead(self.model)

    def compile(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        log.warn("Deprecation: `model.compile()` is deprecated. Please use `model.optimize()` instead.")
        return self.optimize(backend=backend, mode=mode, fullgraph=fullgraph)

    def optimize(self, backend: str = "inductor", mode: str = None, fullgraph: bool = False):
        if not self.quantized:
            log.warn("model is not quantized, skip compiling...")
            return self

        if TORCH_HAS_COMPILE:
            self.compiled = False
            log.warn("To use compile(), you need to have torch version >= 2.6.0, please "
                           "upgrade it by `pip install -U torch torchaudio torchvision`")
            return self

        # needed by eora
        # torch._dynamo.config.capture_scalar_outputs = True

        log.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        modules = find_modules(self.model, layers=[BaseQuantLinear])
        for name in modules.keys():
            modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # supress errors until PyTorch fixed: https://github.com/pytorch/pytorch/issues/132635
        # torch._dynamo.config.suppress_errors = True
        log.info(f"Compiling model with backend: `{backend}`, mode: `{mode}`")

        self.model = torch_compile(self.model, fullgraph=fullgraph, backend=backend, mode=mode)

        #trigger kernel compilation hooks
        # if self.compiled:
        #     modules = find_modules(self.model, layers=[BaseQuantLinear])
        #     for name in modules.keys():
        #         modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        # logger.info(f"Compiling qlinear modules with backend: `{backend}`, mode: `{mode}`")
        # modules = find_modules(self.model, layers=[BaseQuantLinear])
        # for name in modules.keys():
        #     modules[name].optimize(fullgraph=False, backend=backend, mode=mode)

        return self

    def serve(self,
               host: str = "0.0.0.0",
               port: int = 80,
               async_mode: bool = False):
        from ..utils.openai_server import OpenAiServer
        self.server = OpenAiServer(model=self)
        self.server.start(host=host, port=port, async_mode=async_mode)

    def serve_shutdown(self):
        if self.server is not None:
            self.server.shutdown()

    def serve_wait_until_ready(self, timeout: int = 30, check_interval: float = 0.1):
        if self.server is not None:
            self.server.wait_until_ready(timeout=timeout, check_interval=check_interval)

    def before_model_load(self, load_quantized_model):
        pass

    def after_model_load(self, model, load_quantized_model):
        return model

    def pre_quantize_generate_hook_start(self):
        pass

    def pre_quantize_generate_hook_end(self):
        if self.quantize_config.offload_to_disk:
            # This hook is now disabled as it's handled by the ModuleLooper after input capture.
            # offload_to_disk(model=self.model, module=self.get_base_modules(model=self.model), disk_path=self.quantize_config.offload_to_disk_path)
            pass

    def lm_head_pre_quantize_generate_hook(self, inputs: List[List[torch.tensor]]) -> List[List[torch.tensor]]:
        if self.pre_lm_head_norm_module:
            norm, _ = get_module_by_name_prefix(self.model, [self.pre_lm_head_norm_module])
            norm = self.pre_quantize(norm)

            for element in inputs:
                for i in range(len(element)):
                    element[i] = norm(element[i])

            self.post_quantize(norm)
        return inputs

    def pre_quantize(self, module: nn.Module) -> nn.Module:
        if get_device(module) == META:
            return self.shell_module_materialize(
                target_submodule=module,
                device=self.quantize_config.device,
            )
        elif get_device(module) == CPU and self.quantize_config.device != CPU:
            return move_to(module, device=self.quantize_config.device)
        else:
            return module

    def post_quantize(self, module: nn.Module) -> nn.Module:
        #return self.offload_to_disk(module=module)
        return move_to(module, device=CPU)

    def move_embed(self, device: str):
        for embed_module_name in self.get_base_modules(self.model):
            embed_module, _ = get_module_by_name_prefix(self.model, embed_module_name)
            if embed_module is not None:
                self.shell_module_materialize(
                    target_submodule=embed_module,
                    device=device,
                )

    def awq_skip_modules_for_scaling(self) -> bool:
        pass

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        nodes = []
        last_module = None  # most recent norm obj (from a '!...' block)
        last_module_name = None
        last_module_root = None  # self_attn.* has root == self_attn, mlp.* has root == mlp

        if self.model.config is not None and self.dynamic_expert_index is not None:
            self.get_num_experts(self.model.config)

        def strip_non_quantize_flags(module_name):
            for flag in NON_QUANTIZE_FLAGS:
                if flag in module_name:
                    module_name = module_name.replace(flag, "")
            return module_name

        def _select_feature_name(names):
            """Return the first quantized child that has captured activations."""
            for raw in names:
                stripped = strip_non_quantize_flags(raw)
                if stripped in input_feat:
                    return stripped
            return strip_non_quantize_flags(names[0]) if names else None

        def _try_update_last_module(candidate_name: str) -> bool:
            nonlocal last_module, last_module_name, last_module_root

            resolved_module, _ = get_module_by_name_prefix(module, candidate_name)
            if resolved_module is None:
                log.debug(
                    "awq_get_modules_for_scaling: last-module candidate `%s` missing; retaining previous `%s`",
                    candidate_name,
                    last_module_name,
                )
                return False

            last_module = resolved_module
            last_module_name = candidate_name
            if "." in candidate_name:
                last_module_root = candidate_name.split(".", 1)[0]
            return True

        full_layer_modules = self.full_layer_modules(
            self.model.config,
            is_awq_quantize=True,
            include_capture_only=True,
        )
        for i, block in enumerate(full_layer_modules):
            not_quantized = all(any(flag in name for flag in NON_QUANTIZE_FLAGS) for name in block)
            if not_quantized:
                # If both the current block and the previous one are marked as not quantized,
                # skip remembering the current block. This ensures that when two consecutive
                # blocks are not quantized, only the first one is remembered as last_module.
                if i > 0 and all(any(flag in name for flag in NON_QUANTIZE_FLAGS) for name in full_layer_modules[i - 1]):
                    continue

                # Remember the latest norm (use the last entry if multiple are present)
                candidate_name = strip_non_quantize_flags(block[-1])
                _try_update_last_module(candidate_name)
                continue

            is_moe_block = any(any(k in name for k in self.moe_expert_module_name_prefixes) for name in block)
            is_moe_down_block = is_moe_block and any("down" in name for name in block)
            is_moe_gate_up_block = is_moe_block and any("gate" in name for name in block) and any("up" in name for name in block)
            if is_moe_down_block and last_module is not None and last_module_name is not None:
                # mlp.experts.0.down_proj
                target_suffix = last_module_name.split(".")[-1]
                for name in block:
                    prev_op_name = ".".join(name.split(".")[:-1] + [target_suffix])
                    prev_op, _ = get_module_by_name_prefix(module, prev_op_name)
                    if prev_op is None or name not in input_feat:
                        log.debug("awq_get_modules_for_scaling: skipping expert `%s` due to missing prev_op or features", name)
                        continue

                    m, _ = get_module_by_name_prefix(module, name)
                    if m is None:
                        log.debug("awq_get_modules_for_scaling: skipping missing expert module `%s`", name)
                        continue
                    subset = [m]
                    n, root = generate_node_for_awq_scaling(inp=input_feat[name], prev_op=prev_op,
                                                            module_kwargs=module_kwargs, nodes_size=len(nodes),
                                                            subset=subset, module2inspect=None)
                    if root is not None and last_module_root != root:
                        last_module_root = root

                    nodes.append(n)
            else:
                # Normal execution subset
                subset = []  # preserve execution order while collecting quantizable modules
                skip = False
                for name in block:
                    if all(flag not in name for flag in NON_QUANTIZE_FLAGS):
                        m, _ = get_module_by_name_prefix(module, name)
                        # If the Model uses GQA (Grouped Query Attention), attention out will be skipped.
                        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
                        if (self.awq_scale_optimize_shape_dependent_modules is not None
                                and name in self.awq_scale_optimize_shape_dependent_modules
                                and isinstance(last_module, nn.Linear)
                                and last_module.weight.shape != m.weight.shape):
                            # log.debug(f'"{name}" attention out skipped.')
                            skip = True

                        if m is None:
                            log.debug("awq_get_modules_for_scaling: skipping missing module `%s`", name)
                            skip = True
                            break
                        subset.append(m)

                if skip or not subset:
                    continue

                prev_op = last_module
                if prev_op is None:
                    log.debug("awq_get_modules_for_scaling: skipping block %s due to missing previous module", block)
                    continue

                # Match the activation bucket to the first quantized child in this block
                feature_name = _select_feature_name(block) or strip_non_quantize_flags(block[0])
                root_split = feature_name.split(".")
                module2inspect = None
                if len(root_split) >= 2:
                    root = root_split[0]
                    if root != last_module_root:
                        last_module_root = root
                        module2inspect, _ = get_module_by_name_prefix(module, root)

                # process ['mlp.experts.#.gate_proj', 'mlp.experts.#.gup_proj']
                if is_moe_gate_up_block and module2inspect is not None:
                    if last_module_root not in input_feat:
                        log.debug(
                            "awq_get_modules_for_scaling: missing input feature for `%s` while processing experts block (layer block size=%s)",
                            last_module_root,
                            len(block),
                        )
                    inp = input_feat.get(last_module_root, input_feat.get(_select_feature_name(block)))
                else:
                    inp = input_feat.get(_select_feature_name(block))

                if inp is None:
                    log.debug("awq_get_modules_for_scaling: skipping block %s due to missing input features", block)
                    continue

                n, root = generate_node_for_awq_scaling(inp=inp, prev_op=prev_op,
                                                        module_kwargs=module_kwargs, nodes_size=len(nodes),
                                                        subset=subset, module2inspect=module2inspect)

                nodes.append(n)

            # Update tracker to the LAST item of this block
            if is_moe_gate_up_block:
                # The block content is [...,  mlp.experts.{last_index}.up_proj, shared_expert.gate_proj, shared_expert.up_proj, mlp]
                # mlp.experts.{last_index}.up_proj should be selected as last_module
                # Find all indices that contain both ".experts" and "gate_proj"/"up_proj"
                gate_up_proj_indices = [
                    i for i, name in enumerate(block)
                    if any(k in name for k in self.moe_expert_module_name_prefixes) and ("gate" in name or "up" in name)
                ]

                # Use the last one if any exist
                assert len(gate_up_proj_indices) > 0, "No expert gate_proj/up_proj found in block."
                last_up_proj_index = gate_up_proj_indices[-1]

                candidate_name = strip_non_quantize_flags(block[last_up_proj_index])
                assert "gate" in candidate_name or "up" in candidate_name
            else:
                candidate_name = strip_non_quantize_flags(block[-1])
            _try_update_last_module(candidate_name)

        import torch
        def format_nodes(nodes):
            out = []
            for n in nodes:
                entry = {}
                for k, v in n.items():
                    if isinstance(v, torch.Tensor):
                        entry[k] = f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
                    elif isinstance(v, dict):
                        entry[k] = [
                            f"Key: {kk}, Value: Tensor(shape={tuple(x.shape)}, dtype={x.dtype}); " if isinstance(x,
                                                                                                                 torch.Tensor) else type(
                                x).__name__
                            for kk, x in v.items()
                        ]
                    else:
                        entry[k] = v
                out.append(entry)
            return out

        # print("DEBUG AWQ NODES:", format_nodes(nodes))
        return nodes

    def _clone_model_init_kwargs(self, source: PreTrainedModel) -> Dict[str, Any]:
        kwargs = getattr(source, "_model_init_kwargs", {}) or {}
        if isinstance(kwargs, dict):
            return dict(kwargs)
        return copy.deepcopy(kwargs)

    def _resolve_turtle_reload_threshold(self) -> int:
        if not getattr(self.quantize_config, "offload_to_disk", False):
            return 0

        default_bytes = 512 * 1024 ** 2 #512MB
        raw = os.getenv("GPTQMODEL_RELOAD_THRESHOLD")
        if raw is None or raw.strip() == "":
            return default_bytes

        value = raw.strip().lower()
        if value in {"0", "off", "disable", "disabled", "none"}:
            return 0

        units = {
            "b": 1,
            "kb": 1024,
            "mb": 1024 ** 2,
            "gb": 1024 ** 3,
            "tb": 1024 ** 4,
        }

        match = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-z]*)$", value)
        if match is None:
            log.warn(
                "GPTQMODEL_RELOAD_THRESHOLD value `%s` is invalid; defaulting to 512MB.",
                raw,
            )
            return default_bytes

        amount = float(match.group(1))
        unit = match.group(2) or "b"
        multiplier = units.get(unit, None)
        if multiplier is None:
            log.warn(
                "GPTQMODEL_RELOAD_THRESHOLD unit `%s` is unsupported; defaulting to bytes.",
                unit,
            )
            multiplier = 1

        threshold = int(amount * multiplier)
        if threshold < 0:
            threshold = 0
        return threshold

    def _estimate_module_bytes(self, module: nn.Module) -> int:
        if module is None:
            return 0

        total = 0
        seen: Set[int] = set()
        tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.device.type == "meta":
                continue
            try:
                ptr = tensor.data_ptr()
            except (RuntimeError, AssertionError):
                ptr = None
            if ptr is not None:
                if ptr in seen:
                    continue
                seen.add(ptr)
            total += tensor.numel() * tensor.element_size()
        return total

    def _maybe_auto_reload_after_alias(
        self,
        module: nn.Module,
        target_submodule: nn.Module,
    ) -> None:
        if self.turtle_model is None:
            return

        threshold = self._turtle_reload_threshold_bytes
        if threshold <= 0:
            return

        module_id = id(module)
        if module_id in self._turtle_materialized_ids:
            return

        bytes_added = self._estimate_module_bytes(module)
        self._turtle_materialized_ids.add(module_id)

        if bytes_added <= 0:
            return

        self._turtle_reload_accum_bytes += bytes_added

        if self._turtle_reload_accum_bytes >= threshold:
            label = (
                getattr(target_submodule, "full_name", None)
                or getattr(target_submodule, "name", None)
                or getattr(module, "full_name", None)
                or module.__class__.__name__
            )
            self.reload_turtle_model(source=f"auto:{label}")
            self._turtle_reload_accum_bytes = 0

    def reload_turtle_model(self, *, source: Optional[str] = None) -> None:
        if self.quantize_config.offload_to_disk is False:
            return

        timer = getattr(self, "quant_region_timer", None)
        timing_ctx = timer.measure("model_reload", source=source) if timer else nullcontext()

        with timing_ctx:
            def _do_reload():
                with self._turtle_lock:
                    turtle_model = self.turtle_model
                    model_local_path = self.model_local_path
                    loader = self.loader

                    assert turtle_model is not None and model_local_path is not None

                    reload_kwargs = self._clone_model_init_kwargs(turtle_model)
                    config = turtle_model.config
                    del turtle_model

                    new_model = loader.from_pretrained(
                        model_local_path,
                        config=config,
                        low_cpu_mem_usage=True,
                        **reload_kwargs,
                    )
                    new_model._model_init_kwargs = reload_kwargs
                    new_model.eval()
                    self.turtle_model = new_model
                    self._turtle_reload_accum_bytes = 0
            reload_spinner = log.spinner(title="Turtle model reloading...", interval=0.1)
            try:
                DEVICE_THREAD_POOL.submit("model_loader:cpu", _do_reload).result()
            finally:
                reload_spinner.close()

    # transfer actually materizlied module from turtle (real) to shell
    def shell_module_materialize(
            self,
            target_submodule: torch.nn.Module,
            device: torch.device,
            non_blocking: bool = False,
    ) -> torch.nn.Module:
        with self._turtle_lock:
            turtle_model = self.turtle_model

            if turtle_model is None:
                if get_device(target_submodule) != device:
                    target_submodule.to(device)

                return target_submodule

            module = alias_from_turtle_for_submodule(
                target_model=self.model,
                turtle_model=turtle_model,
                target_submodule=target_submodule,
                device=device,
            )
        self._maybe_auto_reload_after_alias(module, target_submodule)
        return module

    ## overrides nn.module.train()
    # def train(self, mode=True):
    #     old_mode = self.training
    #     # Call the parent class's train() method to set the training mode
    #     super().train(mode)
    #
    #     if old_mode == mode:
    #         return
    #
    #     # Custom behavior when switching to training mode
    #     if mode:
    #         if not self.SUPPORTS_TRAINING:
    #             err = f"{self.__class__.__name__}: MODEL switching to training mode."
    #             log.error(err)
    #             raise NotImplementedError(err)
    #         else:
    #             log.info(f"{self.__class__.__name__}: MODEL switching to training mode.")
    #     else:
    #         log.info(f"{self.__class__.__name__}: `MODEL switching to eval mode.")
    @classmethod
    def build_layer_modules(cls, tree, include_capture_only: bool = False):
        """
        tree format:
          [<model_name>, <submodule>, "#", { parent_module: ( "child[:!][:grp]", ... ), ... }]
        Rules:
          - ':!' means participates in inference but is NOT quantized; keep this marker in output.
          - ':?' marks capture-only nodes; activations are recorded but the module is not quantized.
          - ':<digit>' means grouping; children with the same group id are emitted in the same block.
          - Both can appear together, e.g. 'module_name:!:2'.
          - Supports nested dict structures for MoE models with experts.
          - Special key "#" in nested dicts means direct children under parent (no additional nesting).
          - EXPERT_INDEX_PLACEHOLDER in keys will be handled by simple_layer_modules for MoE expansion.
        Output:
          _layer_modules = [ [items...], [items...], ... ]
        """
        mapping = None
        for item in tree:
            if isinstance(item, dict):
                mapping = item
                break
        if mapping is None:
            raise ValueError("Mapping configuration not found in the tree.")

        out_blocks = []
        alias_groups: Dict[tuple[str | None, int], List[tuple[str, bool, bool]]] = {}
        alias_meta: Dict[tuple[str | None, int], Dict[str, int]] = {}
        alias_seq = count()
        group_seq = count()

        def _parse_token(token: str) -> tuple[str, List[str]]:
            parts = token.split(":")
            name = parts[0]
            flags = [p for p in parts[1:] if p]
            return name, flags

        def _group_from_flags(flags: List[str]) -> int:
            for flag in flags:
                if flag.isdigit():
                    return int(flag)
            return 0

        def _has_numeric_flag(flags: List[str]) -> bool:
            return any(flag.isdigit() for flag in flags)

        def _get_scope(parent_name: str) -> str | None:
            if not parent_name:
                return None
            return parent_name.split(".", 1)[0]

        def process_entries(parent_token: str, entries, parent_group_offset: int = 0, scope_key: str | None = None):
            """Process entries recursively to handle nested dict structures for MoE"""
            groups: defaultdict[int, List[tuple]] = defaultdict(list)

            parent_name, parent_flags = _parse_token(parent_token)
            parent_rel_group = _group_from_flags(parent_flags)
            parent_group = parent_group_offset + parent_rel_group
            parent_has_bang = "!" in parent_flags
            parent_capture_only = "?" in parent_flags
            parent_has_numeric = _has_numeric_flag(parent_flags)

            scope = scope_key if scope_key is not None else _get_scope(parent_name)
            parent_alias_scope = scope if parent_has_numeric else parent_name

            def _make_entry(full_path: str, has_bang: bool, capture_only: bool, *, alias_base: int, alias_rel: int, alias_scope: str | None) -> tuple:
                return (full_path, has_bang, capture_only, alias_scope, (alias_base, alias_rel))

            child_group_offset = parent_group_offset
            add_parent = parent_has_bang or (parent_capture_only and include_capture_only)
            if add_parent:
                alias_base = parent_rel_group if parent_has_numeric else parent_group
                parent_entry_scope = f"{parent_alias_scope}.__parent__" if parent_alias_scope is not None else None
                groups[parent_group].append(
                    _make_entry(
                        parent_name,
                        parent_has_bang,
                        parent_capture_only,
                        alias_base=alias_base,
                        alias_rel=0,
                        alias_scope=parent_entry_scope,
                    )
                )
                child_group_offset = max(child_group_offset, parent_group + 1)

            # Handle tuple/list of strings (traditional format)
            if isinstance(entries, (tuple, list)):
                for ent in entries:
                    child_name, child_flags = _parse_token(ent)

                    has_bang = "!" in child_flags
                    capture_only = "?" in child_flags
                    # first numeric tag is the group id; default 0
                    child_rel_group = _group_from_flags(child_flags)
                    grp = child_group_offset + child_rel_group
                    # Apply parent group offset to avoid conflicts between different nesting levels
                    # Store the full path including parent for later use
                    if parent_name.endswith(f".{child_name}") or parent_name == child_name:
                        full_path = parent_name
                    elif parent_name:
                        full_path = f"{parent_name}.{child_name}"
                    else:
                        full_path = child_name

                    if capture_only and not include_capture_only:
                        continue
                    alias_scope = scope if parent_has_numeric else parent_name
                    alias_base = parent_rel_group if parent_has_numeric else grp
                    alias_rel = child_rel_group if parent_has_numeric else 0
                    groups[grp].append(
                        _make_entry(
                            full_path,
                            has_bang,
                            capture_only,
                            alias_base=alias_base,
                            alias_rel=alias_rel,
                            alias_scope=alias_scope,
                        )
                    )

            elif isinstance(entries, dict):
                # Calculate max group number used at current level to avoid conflicts
                max_current_group = 0
                for sub_parent, sub_entries in entries.items():
                    if isinstance(sub_entries, (tuple, list)):
                        for ent in sub_entries:
                            _, ent_flags = _parse_token(ent)
                            max_current_group = max(max_current_group, _group_from_flags(ent_flags))

                # Process nested entries with appropriate group offset
                current_offset = child_group_offset
                for sub_parent, sub_entries in entries.items():
                    if sub_parent == "#":
                        # Special case: "#" means expert index placeholder
                        # Create a template path that will be expanded later by simple_layer_modules
                        template_parent = (
                            f"{parent_name}.{EXPERT_INDEX_PLACEHOLDER}"
                            if parent_name else EXPERT_INDEX_PLACEHOLDER
                        )
                        template_parent_token = (
                            f"{template_parent}:{parent_rel_group}"
                            if parent_has_numeric
                            else template_parent
                        )
                        # Use a higher offset for expert modules to avoid conflicts with parent level
                        expert_offset = current_offset + max_current_group + 100  # Large offset to avoid conflicts

                        # Handle special case where sub_entries is ("#",) or "#" - this means use the parent path directly
                        if (isinstance(sub_entries, (tuple, list)) and len(sub_entries) == 1 and sub_entries[0] == "#") or sub_entries == "#":
                            # For ("#",) or "#" format, use the template_parent directly with default group 0
                            alias_scope = scope if parent_has_numeric else template_parent
                            alias_base = parent_rel_group if parent_has_numeric else expert_offset
                            groups[expert_offset].append(
                                _make_entry(
                                    template_parent,
                                    False,
                                    False,
                                    alias_base=alias_base,
                                    alias_rel=0,
                                    alias_scope=alias_scope,
                                )
                            )
                        else:
                            sub_groups = process_entries(template_parent_token, sub_entries, expert_offset, scope)
                            for grp, items in sub_groups.items():
                                groups[grp].extend(items)
                    else:
                        # Nested structure: process recursively with full path
                        # Special case: empty string key means use parent path directly
                        if sub_parent == "":
                            full_sub_parent = parent_name
                        else:
                            full_sub_parent = (
                                f"{parent_name}.{sub_parent}"
                                if parent_name else sub_parent
                            )
                        sub_groups = process_entries(full_sub_parent, sub_entries, current_offset, scope)
                        for grp, items in sub_groups.items():
                            groups[grp].extend(items)
                        # Update offset for next sibling to avoid conflicts
                        if sub_groups:
                            current_offset = max(sub_groups.keys()) + 1

            return groups

        def _register_alias(order_idx: int, item: tuple[str, bool, bool, str | None, tuple[int, int]]):
            full_path, has_bang, capture_only, scope, alias_parts = item
            if capture_only and not include_capture_only:
                return
            alias_scope = scope
            alias_base, alias_rel = alias_parts
            alias_index = alias_base + alias_rel
            key = (alias_scope, alias_index)
            meta = alias_meta.get(key)
            if meta is None:
                alias_meta[key] = {"order": order_idx, "seq": next(alias_seq)}
                alias_groups[key] = [(full_path, has_bang, capture_only)]
            else:
                meta["order"] = min(meta["order"], order_idx)
                alias_groups[key].append((full_path, has_bang, capture_only))

        for parent, entries in mapping.items():
            groups = process_entries(parent, entries)

            for g in sorted(groups):
                order_idx = next(group_seq)
                items = groups[g]
                for item in items:
                    if len(item) == 3:
                        full_path, has_bang, capture_only = item
                        scope = full_path
                        alias_parts = (g, 0)
                        _register_alias(order_idx, (full_path, has_bang, capture_only, scope, alias_parts))
                    else:
                        _register_alias(order_idx, item)

        for key in sorted(alias_groups.keys(), key=lambda k: (alias_meta[k]["order"], alias_meta[k]["seq"])):
            block = []
            for full_path, has_bang, capture_only in alias_groups[key]:
                name = full_path
                if has_bang:
                    name += NOT_QUANTIZE_FLAG
                if capture_only and include_capture_only:
                    name += CAPTURE_ONLY_FLAG
                block.append(name)
            out_blocks.append(block)

        return out_blocks

    @classmethod
    def get_base_modules(cls, model):
        """
        Return list of base modules directly under 'model' but not 'model.layers'.
        """
        # Find the index of "#"
        tree = cls.module_tree
        try:
            sharp_idx = tree.index("#")
        except ValueError:
            raise ValueError("module_tree must contain '#' to separate hierarchy")

        assert sharp_idx > 0, "failed to get_base_modules"
        # root_path = ["model"] or ["model", "language_model"]
        root_path = tree[:sharp_idx-1]

        out = []
        # Traverse each layer in root_path
        for i in range(len(root_path)):
            path = root_path[:i + 1]
            base = model
            exclude = tree[len(path)]

            for node in path:
                base = getattr(base, node)

            for name, _ in base.named_children():
                if name != exclude:
                    out.append(".".join(path + [name]))

        # print(f"Base Modules: {out}")
        return out

    def generate_layers_modules_tree_simple(self, node):
        """
        Recursively walk a nested list/dict structure and:
          1. Drop dict entries where *all* values are ':!' or ':?' flagged.
          2. Remove ':!' / ':?' and ':<digit>' markers from strings.
        """

        # If it's a list, recurse into each element
        if isinstance(node, list):
            return [self.generate_layers_modules_tree_simple(x) for x in node]

        # If it's a dict, process each key -> value
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                # Expand tuple-of-strings blocks (special handling)
                if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
                    # Rule 1: check if ALL entries are :!
                    if all(any(p in {"!", "?"} for p in x.split(":")[1:]) for x in v):
                        continue  # skip this parent entirely

                    # Rule 2: strip :! and :digit markers
                    cleaned = tuple(x.split(":")[0] for x in v)
                    new_dict[k] = cleaned
                else:
                    # Recurse deeper
                    new_dict[k] = self.generate_layers_modules_tree_simple(v)
            return new_dict

        # If it's a plain string (unlikely here), strip markers
        if isinstance(node, str):
            return node.split(":")[0]

        # For other types, return as-is
        return node

    def tied_word_embedding(self) -> bool:
        return getattr(self.model.config, "tie_word_embeddings", False)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception as exc:  # torch Modules raise AttributeError here
            model = self.__dict__.get("model")
            if model is None:
                model = self._modules.get("model") if hasattr(self, "_modules") else None
            if model is not None and item != "model":
                return getattr(model, item)
            raise exc

    def _auto_detect_module_tree(self, model: PreTrainedModel, quant_method: METHOD):
        log.warn("Model not yet support, attempting Module Tree AutoCompat...")

        if quant_method != METHOD.GPTQ:
            log.warn(f"Module Tree AutoCompat: Failed, quant_method={quant_method}, only support GPTQ")
            return None

        def _get(path):
            base = model
            for p in path.split("."):
                base = getattr(base, p, None)
                if base is None:
                    return None
            return base

        candidates = [
            "model.layers",
            "language_model.layers",
            "model.decoder.layers",
            "transformer.h",
            "transformer.blocks",
            "layers",
            "blocks",
            "model.blocks",
        ]

        chosen = None
        for c in candidates:
            m = _get(c)
            if isinstance(m, (nn.ModuleList, list, tuple)) and len(m) > 0 and isinstance(m[0], nn.Module):
                chosen = c
                log.warn(f"Module Tree AutoCompat: Matched candidate path '{c}', type={type(m).__name__}")
                break

        if chosen is None:
            log.warn("Module Tree AutoCompat: All candidate paths invalid, return None")
            return None

        layer0 = _get(chosen)[0]
        log.warn(f"Module Tree AutoCompat: Using layer0: {type(layer0).__name__}")

        def _linear_names(module):
            mods = find_modules(module, layers=[nn.Linear, nn.Conv1d, nn.Conv2d])
            log.warn(f"Module Tree AutoCompat: _linear_names: found {len(mods)} Linear/Conv modules in {type(module).__name__}")
            return list(mods.keys())

        all_linear = _linear_names(layer0)
        if len(all_linear)>0:
            log.warn(f"Module Tree AutoCompat: found {len(all_linear)} Linear/Conv modules in {type(layer0).__name__}: {all_linear}")
        else:
            log.warn("Module Tree AutoCompat: No Linear/Conv names in layer0, return None")
            return None

        mapping = {}

        def _find_parents(module, possible_names):
            found = set()
            for n, _ in module.named_children():
                l = n.lower()
                if any(k in l for k in possible_names):
                    found.add(n)
            return found

        def _leaf_tokens(prefix):
            return tuple(x.split(".")[-1] for x in all_linear if x.startswith(f"{prefix}."))

        possible_parent = ["attn", "attention", "self_attn", "mlp", "ffn", "feed", "dense"]

        found_parents = _find_parents(layer0, possible_parent)

        for p in found_parents:
            t = _leaf_tokens(p)
            if t:
                mapping[p] = t

        if not mapping:
            blocks = tuple(n.split(".")[-1] for n in all_linear)
            mapping[""] = blocks
            log.warn(f"Module Tree AutoCompat: Mapping empty, using all Linear as fallback: {blocks}")

        parts = chosen.split(".")
        tree = parts + ["#", mapping]
        log.warn(f"Module Tree AutoCompat: Final module_tree: {tree}")
        return tree

__all__ = ["BaseQModel"]

BaseQModel = ModelLoader(ModelWriter(BaseQModel))
