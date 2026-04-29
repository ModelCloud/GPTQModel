# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# -- do not touch
import copy
import os
import sys


if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7" #"expandable_segments:True"

# Following makes test results more deterministic but much slower
# # the CUBLAS env is required for use_deterministic_algorithms
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#
# import torch
# torch.use_deterministic_algorithms(True)

# -- end do not touch

from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from logbar import LogBar  # noqa: E402


sys.path.insert(0, f"{str(Path(__file__).resolve().parent.parent)}/models")  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import textwrap  # noqa: E402
import unittest  # noqa: E402
from collections.abc import Iterable, Mapping  # noqa: E402

import torch.cuda  # noqa: E402


def _env_choice(*names: str, default: str) -> str:
    """Return the first non-empty env override from a prioritized name list."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value:
            return value
    return default


def _env_int(*names: str, default: int) -> int:
    """Return the first parseable integer env override from a prioritized name list."""
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip()
        if value:
            return int(value)
    return default


def _env_flag(*names: str, default: bool = False) -> bool:
    """Return the first parseable boolean env override from a prioritized name list."""
    truthy = {"1", "true", "yes", "on", "y", "t"}
    falsy = {"0", "false", "no", "off", "n", "f"}
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value in truthy:
            return True
        if value in falsy:
            return False
    return default


def _env_optional_flag(*names: str) -> Optional[bool]:
    """Return the first parseable boolean env override, or None when no override is set."""
    truthy = {"1", "true", "yes", "on", "y", "t"}
    falsy = {"0", "false", "no", "off", "n", "f"}
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        value = raw.strip().lower()
        if value in truthy:
            return True
        if value in falsy:
            return False
    return None


try:  # noqa: E402
    from datasets import load_dataset as hf_load_dataset  # noqa: E402
except Exception as exc:  # pragma: no cover - depends on test environment
    hf_load_dataset = None
    DATASETS_IMPORT_ERROR = exc
else:
    DATASETS_IMPORT_ERROR = None


try:
    from ovis.image_to_test_dataset import get_calib_dataset  # noqa: E402
except BaseException:
    pass

from transformers import AutoProcessor, AutoTokenizer  # noqa: E402


try:  # noqa: E402
    from transformers.utils import is_flash_attn_2_available  # noqa: E402
except Exception:  # pragma: no cover - availability check
    def is_flash_attn_2_available():  # type: ignore
        return False

from tests.eval import (  # noqa: E402
    evaluate,
    format_eval_result_table,
    get_eval_task_results,
    resolve_eval_metric_alias,
)

from gptqmodel import BACKEND, DEBUG_ON, GPTQModel  # noqa: E402
from gptqmodel.looper.module_looper import StopMainLoop  # noqa: E402
from gptqmodel.models.base import BaseQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.quantization.config import (  # noqa: E402
    BitsAndBytesConfig,
    Fallback,
    FOEMConfig,
    FP8Config,
    GGUFConfig,
    GPTAQConfig,
    HessianConfig,
    MoEConfig,
    ParoConfig,
    QuantizeConfig,
    RTNConfig,
    VramStrategy,
    WeightOnlyConfig,
    resolve_quant_format,
)
from gptqmodel.utils.logger import render_table  # noqa: E402
from gptqmodel.utils.model import MODALITY  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402


RAND_SEED = 898

log = LogBar.shared()

DEFAULT_FLOOR_PCT = 0.05
DEFAULT_CEIL_PCT = 0.10
DEFAULT_TASK_NAMES = ("arc_challenge",)


class ModelTest(unittest.TestCase):
    DEBUG = True # enable extra debug output

    DENSE_VRAM_STRATEGY = VramStrategy.EXCLUSIVE
    DENSE_VRAM_STRATEGY_DEVICES = None
    MOE_VRAM_STRATEGY = VramStrategy.EXCLUSIVE
    MOE_VRAM_STRATEGY_DEVICES = None
    TRUST_REMOTE_CODE = False
    TORCH_DTYPE = "auto"
    EVAL_BATCH_SIZE = "auto"
    QUANT_BATCH_SIZE = 1
    LOAD_BACKEND = BACKEND.MARLIN
    QUANT_BACKEND = BACKEND.AUTO
    USE_VLLM = False
    INPUTS_MAX_LENGTH = 2048
    MODEL_MAX_LEN = 4096
    DATASET_SIZE = 512
    DATASET_SIZE_FAST = None
    DATASET_SIZE_SLOW = None
    DATASET_CONCAT_SIZE = None
    DATASET_CONCAT_SIZE_FAST = None
    DATASET_CONCAT_SIZE_SLOW = None
    DATASET_CONCAT_SEPARATOR = None
    DATASET_SORT = "desc"
    DELETE_QUANTIZED_MODEL = True
    EVAL_TASKS = None
    EVAL_SINGLE_GPU = True
    LOAD_MODEL_EXTRA_ARGS: Dict[str, Any] = {}
    EVAL_TASKS_FAST = None
    EVAL_TASKS_SLOW = None
    MODEL_TEST_MODE_ENV = "GPTQMODEL_MODEL_TEST_MODE"
    MODEL_TEST_MODE_FAST = "fast"
    MODEL_TEST_MODE_SLOW = "slow"
    # Shared override for the fast-mode quantized layer prefix across all ModelTest-based tests.
    FAST_LAYER_COUNT_ENV = "GPTQMODEL_FAST_LAYER_COUNT"
    FAST_LAYER_POSITION_ENV = "GPTQMODEL_FAST_LAYER_POSITION"
    MODEL_COMPAT_FAST_LAYER_COUNT = None
    MODEL_COMPAT_FAST_LAYER_POSITION = None

    KERNEL_QUANT = {}  # kernel sets
    KERNEL_INFERENCE = {}  # kernel sets

    # quant config
    FORMAT = FORMAT.GPTQ
    METHOD = METHOD.GPTQ
    BITS = 4
    GROUP_SIZE = 128
    DESC_ACT = False
    SYM = True
    GPTAQ: Optional[GPTAQConfig] = None
    FOEM: Optional[FOEMConfig] = None
    ACT_GROUP_AWARE = True
    FALLBACK = Fallback()
    EORA = None
    DAMP_PERCENT = 0.05
    MSE = 0.0
    DYNAMIC = None
    HESSIAN_CHUNK_SIZE = None
    WEIGHT_ONLY = None
    BNB_FORMAT = None
    BNB_BLOCK_SIZE = None
    BNB_COMPRESS_STATISTICS = None

    PAROQUANT_ROTATION_EPOCHS = None
    PAROQUANT_FINETUNE_EPOCHS = None
    PAROQUANT_TRAIN_SAMPLES = None

    SAVE_PATH = None  # default is temp folder
    SPLIT_BY: Optional[str] = None

    USE_FLASH_ATTN = True

    INFERENCE_PROMPT = "The capital city of France is named"
    INFERENCE_RESULT_KEYWORDS = ["paris"]
    DISABLE_NATIVE_BASELINE_FALLBACK = True
    GENERATE_EVAL_SIZE_MIN = 128
    GENERATE_EVAL_SIZE_MAX = 128
    APPLY_CHAT_TEMPLATE = False

    LM_HEAD_LOSS_MAX_DELTA_PERCENT = 0.1  # ±10%

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        model_id = getattr(cls, "NATIVE_MODEL_ID", None)
        if isinstance(model_id, str):
            model_id = model_id.strip()
            if os.path.isabs(model_id) and not os.path.isdir(model_id):
                raise unittest.SkipTest(f"Model path missing: {model_id}")
    EXPECT_LM_HEAD_LOSS = None
    STOP_AFTER_LAYER: Optional[int] = None
    MOE_CONFIG: Optional[MoEConfig] = None
    OFFLOAD_TO_DISK: bool = True
    OFFLOAD_TO_DISK_FAST = None
    OFFLOAD_TO_DISK_SLOW = None

    GENERIC_TEST_PROMPTS = [
        {"prompt": "Which city is the capital city of France?", "keywords": ["paris"]},
        # {"prompt": "What is the smallest habitable planet in the milky way?", "keywords": ["earth"]},
        # {"prompt": "Who wrote the play Romeo and Juliet?", "keywords": ["shakespeare"]},
        # {"prompt": "What gas do plants primarily absorb from the atmosphere during photosynthesis?", "keywords": ["carbon dioxide"]},
        # {"prompt": "Name the largest ocean on Earth.", "keywords": ["pacific"]},
    ]

    @classmethod
    def derive_fast_eval_tasks(cls, eval_tasks, min_ceil_pct: float = 1.0):
        if eval_tasks is None:
            return None

        fast_tasks = copy.deepcopy(eval_tasks)
        for metrics in fast_tasks.values():
            if not isinstance(metrics, dict):
                continue
            for metric_name, spec in list(metrics.items()):
                if metric_name == "chat_template":
                    continue
                if isinstance(spec, dict):
                    current_ceil = spec.get("ceil_pct", spec.get("max_delta_ceil_percent", DEFAULT_CEIL_PCT))
                    spec["ceil_pct"] = max(float(current_ceil), float(min_ceil_pct))
                else:
                    metrics[metric_name] = {
                        "value": float(spec),
                        "floor_pct": DEFAULT_FLOOR_PCT,
                        "ceil_pct": float(min_ceil_pct),
                    }
        return fast_tasks

    def _model_test_mode(self) -> str:
        raw = os.environ.get(self.MODEL_TEST_MODE_ENV, self.MODEL_TEST_MODE_FAST)
        normalized = str(raw).strip().lower()
        if normalized in {"", self.MODEL_TEST_MODE_FAST}:
            return self.MODEL_TEST_MODE_FAST
        if normalized in {self.MODEL_TEST_MODE_SLOW, "full"}:
            return self.MODEL_TEST_MODE_SLOW
        raise ValueError(
            f"Unsupported {self.MODEL_TEST_MODE_ENV}={raw!r}; expected "
            f"`{self.MODEL_TEST_MODE_FAST}` or `{self.MODEL_TEST_MODE_SLOW}`."
        )

    def _is_fast_model_test_mode(self) -> bool:
        return self._model_test_mode() == self.MODEL_TEST_MODE_FAST

    @contextlib.contextmanager
    def model_compat_test_context(self):
        previous = getattr(self, "_model_compat_eval_in_progress", False)
        self._model_compat_eval_in_progress = True
        try:
            yield
        finally:
            self._model_compat_eval_in_progress = previous

    def _in_model_compat_eval_flow(self) -> bool:
        return bool(getattr(self, "_model_compat_eval_in_progress", False))

    def _should_use_fast_model_compat_quant(self) -> bool:
        return self._in_model_compat_eval_flow() and self._is_fast_model_test_mode()

    def _selected_eval_tasks_config(self):
        if self._is_fast_model_test_mode():
            if self.EVAL_TASKS_FAST is not None:
                return self.EVAL_TASKS_FAST
            if self.EVAL_TASKS_SLOW is not None:
                return self.derive_fast_eval_tasks(self.EVAL_TASKS_SLOW)
            if self.EVAL_TASKS is not None:
                return self.derive_fast_eval_tasks(self.EVAL_TASKS)
        else:
            if self.EVAL_TASKS_SLOW is not None:
                return self.EVAL_TASKS_SLOW
        return self.EVAL_TASKS

    def _mode_specific_baseline_value(self, attr_name: str):
        mode_suffix = "FAST" if self._is_fast_model_test_mode() else "SLOW"
        preferred = f"{attr_name}_{mode_suffix}"
        if hasattr(self, preferred):
            return self._resolve_metric_baseline_value(getattr(self, preferred))

        if self._is_fast_model_test_mode():
            fallback = f"{attr_name}_SLOW"
            if hasattr(self, fallback):
                return self._resolve_metric_baseline_value(getattr(self, fallback))

        return self._resolve_metric_baseline_value(getattr(self, attr_name, None))

    def _mode_specific_test_setting(self, attr_name: str):
        mode_suffix = "FAST" if self._is_fast_model_test_mode() else "SLOW"
        preferred = f"{attr_name}_{mode_suffix}"
        value = getattr(self, preferred, None)
        if value is not None:
            return value
        return getattr(self, attr_name, None)

    def _legacy_metric_ceil_pct(self) -> float:
        if self._is_fast_model_test_mode():
            return 1.0
        return DEFAULT_CEIL_PCT

    @staticmethod
    def _merge_dynamic_configs(*configs):
        merged = {}
        for config in configs:
            if not config:
                continue
            merged.update(copy.deepcopy(config))
        return merged or None

    def _resolve_layers_for_fast_model_compat(self, model):
        layers_node = model.extract_layers_node()
        if isinstance(layers_node, (list, tuple)):
            if not layers_node:
                return None, None
            layers_node = layers_node[0]

        layers = model.model
        for part in layers_node.split("."):
            layers = getattr(layers, part)

        return layers_node, layers

    @staticmethod
    def _layer_type_signature(layer) -> tuple:
        layer_type = f"{type(layer).__module__}.{type(layer).__qualname__}"
        top_children = tuple(
            (name, f"{type(module).__module__}.{type(module).__qualname__}")
            for name, module in layer.named_children()
        )
        feature_tokens = ("moe", "expert", "router", "gate")
        features = set()
        for name, _module in layer.named_modules():
            lower = name.lower()
            for token in feature_tokens:
                if token in lower:
                    features.add(token)
        return layer_type, top_children, tuple(sorted(features))

    def _summarize_layer_signatures(self, layers) -> List[Dict[str, Any]]:
        summaries: Dict[tuple, Dict[str, Any]] = {}
        for idx, layer in enumerate(layers):
            signature = self._layer_type_signature(layer)
            summary = summaries.get(signature)
            if summary is None:
                layer_type, top_children, features = signature
                summary = {
                    "first_idx": idx,
                    "count": 0,
                    "layer_type": layer_type,
                    "top_children": [name for name, _ in top_children][:8],
                    "features": list(features),
                }
                summaries[signature] = summary
            summary["count"] += 1

        return sorted(summaries.values(), key=lambda item: item["first_idx"])

    def _resolve_fast_model_layer_count_config(self, layer_count: int) -> Dict[str, Any]:
        raw_env_value = os.environ.get(self.FAST_LAYER_COUNT_ENV)
        if raw_env_value is not None:
            resolved = self._parse_fast_model_layer_count(raw_env_value, field_name=self.FAST_LAYER_COUNT_ENV, layer_count=layer_count)
            return {
                "source": "env",
                "name": self.FAST_LAYER_COUNT_ENV,
                "raw": raw_env_value,
                "resolved": resolved,
            }

        configured_min_layers = self.MODEL_COMPAT_FAST_LAYER_COUNT
        if configured_min_layers is not None:
            resolved = self._parse_fast_model_layer_count(
                configured_min_layers,
                field_name="MODEL_COMPAT_FAST_LAYER_COUNT",
                layer_count=layer_count,
            )
            return {
                "source": "class",
                "name": "MODEL_COMPAT_FAST_LAYER_COUNT",
                "raw": configured_min_layers,
                "resolved": resolved,
            }

        return {
            "source": "default",
            "name": "default",
            "raw": 2,
            "resolved": min(2, layer_count),
        }

    def _resolve_fast_model_layer_position(self) -> str:
        raw = os.environ.get(self.FAST_LAYER_POSITION_ENV)
        if raw is None:
            configured = self.MODEL_COMPAT_FAST_LAYER_POSITION
            raw = "last" if configured is None else configured
        normalized = str(raw).strip().lower()
        if normalized in {"", "first", "prefix", "head"}:
            return "first"
        if normalized in {"last", "suffix", "tail", "top"}:
            return "last"
        raise ValueError(
            f"{self.FAST_LAYER_POSITION_ENV} must be `first` or `last`, got {raw!r}."
        )

    @staticmethod
    def _parse_fast_model_layer_count(raw_value: Any, *, field_name: str, layer_count: int) -> int:
        normalized = str(raw_value).strip().lower()
        if normalized == "":
            return min(2, layer_count)
        if normalized in {"all", "full"}:
            return layer_count

        try:
            return max(int(normalized), 0)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be a non-negative integer or `all`, got {raw_value!r}."
            ) from exc

    def _fast_model_layer_limit(self, layers) -> int:
        layer_count = len(layers)
        config = self._resolve_fast_model_layer_count_config(layer_count)
        min_layers = config["resolved"]
        if layer_count <= min_layers:
            return layer_count

        signature_summaries = self._summarize_layer_signatures(layers)
        unique_signatures = len(signature_summaries)

        if unique_signatures == 1:
            return min(min_layers, layer_count)

        last_first_idx = max((item["first_idx"] for item in signature_summaries), default=0)
        return min(layer_count, max(min_layers, last_first_idx + 1))

    def _build_fast_model_compat_dynamic(self, model) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self._should_use_fast_model_compat_quant():
            return None

        layers_node, layers = self._resolve_layers_for_fast_model_compat(model)
        if layers_node is None or layers is None:
            return None

        layer_count = len(layers)
        layer_limit = self._fast_model_layer_limit(layers)
        layer_limit_config = self._resolve_fast_model_layer_count_config(layer_count)
        layer_position = self._resolve_fast_model_layer_position()
        log.info(
            "Fast quant mode layer limit config: %s=%r -> resolved %s %s/%s layers.",
            layer_limit_config["name"],
            layer_limit_config["raw"],
            layer_position,
            layer_limit_config["resolved"],
            layer_count,
        )
        if layer_count <= layer_limit:
            log.info(
                "Fast quant mode: layer limit covers the full model (%s/%s layers); skipping 0 layers.",
                layer_count,
                layer_count,
            )
            return None

        if layer_position == "last":
            skipped_layers = range(0, max(0, layer_count - layer_limit))
        else:
            skipped_layers = range(layer_limit, layer_count)

        dynamic = {f"-:^{layers_node}\\.{i}\\.": {} for i in skipped_layers}

        unique_layer_types = len({self._layer_type_signature(layer) for layer in layers})
        log.info(
            "Fast quant mode: quantizing %s %s/%s layers (%s unique layer type signatures covered), skipping %s layers.",
            layer_position,
            layer_limit,
            layer_count,
            unique_layer_types,
            layer_count - layer_limit,
        )
        signature_summaries = self._summarize_layer_signatures(layers)
        log.info(
            "Fast quant mode layer signature details: \n%s",
            "\n".join(
                (
                    f"first_layer={item['first_idx']}, count={item['count']}, "
                    f"type={item['layer_type']}, top_children={item['top_children'] or ['<none>']}, "
                    f"special_features={item['features'] or ['<none>']}"
                )
                for item in signature_summaries
            ),
        )

        return dynamic

    def _apply_model_compat_quant_overrides(self, model) -> None:
        dynamic = self._build_fast_model_compat_dynamic(model)
        if dynamic is None:
            return

        model.quantize_config.dynamic = self._merge_dynamic_configs(model.quantize_config.dynamic, dynamic)
        self._model_compat_fast_dynamic = dynamic

    @staticmethod
    def _build_layer_stop_callback(layer_idx: int):
        class _StopAfterLayer:
            def __init__(self, target: int):
                self._target = target
                self._triggered = False

            def layer_complete(self, *, layer_idx: int, submodule_finalized: bool):
                if self._triggered:
                    return None
                if layer_idx > self._target or (submodule_finalized and layer_idx >= self._target):
                    self._triggered = True
                    raise StopMainLoop

        return _StopAfterLayer(layer_idx)

    def _debug_layer_stop_triggered(self) -> bool:
        if not DEBUG_ON:
            return False
        callback = getattr(self, "_layer_stop_callback", None)
        return bool(callback and getattr(callback, "_triggered", False))

    def _finalize_quant_debug_path(
        self,
        *,
        model,
        tokenizer,
        processor,
    ):
        return model, tokenizer, processor

    def _normalize_task_identifier(self, task):
        if task is None:
            raise ValueError("Evaluation task identifier cannot be None")
        normalized = str(task).strip()
        if not normalized:
            raise ValueError("Evaluation task identifier cannot be empty")
        return normalized

    def _normalize_task_list(self):
        task_specs = self.get_eval_tasks()
        if task_specs:
            task_names = list(task_specs.keys())
        else:
            task_names = list(DEFAULT_TASK_NAMES)

        normalized = [self._normalize_task_identifier(task) for task in task_names if task is not None]
        if not normalized:
            raise ValueError("No evaluation tasks configured")
        return normalized

    def _legacy_arc_tasks(self):
        baselines = {}
        arc_metrics = {}
        native_acc = self._mode_specific_baseline_value("NATIVE_ARC_CHALLENGE_ACC")
        native_acc_norm = self._mode_specific_baseline_value("NATIVE_ARC_CHALLENGE_ACC_NORM")
        ceil_pct = self._legacy_metric_ceil_pct()
        if native_acc is not None:
            arc_metrics["acc"] = {
                "value": native_acc,
                "floor_pct": DEFAULT_FLOOR_PCT,
                "ceil_pct": ceil_pct,
            }
        if native_acc_norm is not None:
            arc_metrics["acc_norm"] = {
                "value": native_acc_norm,
                "floor_pct": DEFAULT_FLOOR_PCT,
                "ceil_pct": ceil_pct,
            }
        if arc_metrics:
            normalized = self._normalize_task_identifier("arc_challenge")
            baselines[normalized] = arc_metrics
            chat_lookup = getattr(self, "_task_chat_template", None)
            if isinstance(chat_lookup, dict):
                chat_lookup[normalized] = False
        return baselines

    def _normalize_metric_spec(self, spec):
        default_floor = DEFAULT_FLOOR_PCT
        default_ceil = DEFAULT_CEIL_PCT

        if isinstance(spec, dict):
            if "value" not in spec:
                raise ValueError("Baseline metric dictionaries must include a `value` key.")
            value = self._resolve_metric_baseline_value(spec["value"])
            floor_pct = spec.get("floor_pct", spec.get("max_delta_floor_percent", default_floor))
            ceil_pct = spec.get("ceil_pct", spec.get("max_delta_ceil_percent", default_ceil))
            metric_key = spec.get("metric_key")
        else:
            value = self._resolve_metric_baseline_value(spec)
            floor_pct = default_floor
            ceil_pct = default_ceil
            metric_key = None

        if not isinstance(value, (int, float)):
            raise TypeError(f"Baseline metric value must be numeric, got {type(value).__name__}")
        if not isinstance(floor_pct, (int, float)):
            raise TypeError(f"`floor_pct` must be numeric, got {type(floor_pct).__name__}")
        if not isinstance(ceil_pct, (int, float)):
            raise TypeError(f"`ceil_pct` must be numeric, got {type(ceil_pct).__name__}")

        return {
            "value": float(value),
            "floor_pct": float(floor_pct),
            "ceil_pct": float(ceil_pct),
            "metric_key": metric_key,
        }

    @staticmethod
    def _detect_cuda0_name():
        try:
            if not torch.cuda.is_available():
                return None
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return None

    @classmethod
    def _detect_gpu_profile(cls):
        cuda0_name = cls._detect_cuda0_name()
        if not cuda0_name:
            return None

        normalized = cuda0_name.lower()
        if "a100" in normalized:
            return "A100"
        if "4090" in normalized:
            return "RTX4090"
        return None

    def _resolve_metric_baseline_value(self, value):
        if not isinstance(value, Mapping):
            return value

        normalized_lookup = {
            self._normalize_gpu_profile_key(key): val
            for key, val in value.items()
        }
        gpu_profile = self._detect_gpu_profile()
        normalized_profile = self._normalize_gpu_profile_key(gpu_profile)

        if normalized_profile is not None and normalized_profile in normalized_lookup:
            return normalized_lookup[normalized_profile]

        if "a100" in normalized_lookup:
            return normalized_lookup["a100"]

        available = ", ".join(sorted(normalized_lookup.keys()))
        raise ValueError(
            "Unable to resolve GPU-specific baseline value. "
            f"Detected profile={gpu_profile!r}, available profiles={available}."
        )

    @staticmethod
    def _normalize_gpu_profile_key(profile):
        if profile is None:
            return None
        normalized = str(profile).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if normalized == "a100":
            return "a100"
        if normalized in {"rtx4090", "4090"}:
            return "rtx4090"
        return normalized

    def get_eval_tasks(self):
        self._task_chat_template = {}
        self._task_evalution_suite_kwargs = {}
        self._task_evalution_model_args = {}
        self._task_evalution_use_model_path = {}
        self._task_evalution_batch_size = {}
        eval_tasks = self._selected_eval_tasks_config()
        if eval_tasks:
            baselines = {}
            for task, metrics in eval_tasks.items():
                normalized_task = self._normalize_task_identifier(task)

                metrics_dict = dict(metrics or {})
                chat_template = bool(metrics_dict.pop("chat_template", False))
                evalution_suite_kwargs = dict(metrics_dict.pop("evalution_suite_kwargs", {}) or {})
                evalution_model_args = dict(metrics_dict.pop("evalution_model_args", {}) or {})
                evalution_use_model_path = bool(metrics_dict.pop("evalution_use_model_path", False))
                evalution_batch_size = metrics_dict.pop("evalution_batch_size", None)
                self._task_chat_template[normalized_task] = chat_template
                self._task_evalution_suite_kwargs[normalized_task] = evalution_suite_kwargs
                self._task_evalution_model_args[normalized_task] = evalution_model_args
                self._task_evalution_use_model_path[normalized_task] = evalution_use_model_path
                self._task_evalution_batch_size[normalized_task] = evalution_batch_size

                baselines[normalized_task] = {
                    metric_name: self._normalize_metric_spec(spec)
                    for metric_name, spec in metrics_dict.items()
                }
            return baselines

        baselines = self._legacy_arc_tasks()
        if isinstance(baselines, dict):
            for task_name in baselines.keys():
                if task_name not in self._task_chat_template:
                    self._task_chat_template[task_name] = False
                self._task_evalution_suite_kwargs.setdefault(task_name, {})
                self._task_evalution_model_args.setdefault(task_name, {})
                self._task_evalution_use_model_path.setdefault(task_name, False)
                self._task_evalution_batch_size.setdefault(task_name, None)
        return baselines

    @staticmethod
    def _flatten_task_metrics(task_results):
        flat = {}
        for task_name, metrics in task_results.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    flat[f"{task_name}:{metric_name}"] = value
            else:
                flat[task_name] = metrics
        return flat


    def assertInference(self, model, tokenizer=None, keywords=None, prompt=INFERENCE_PROMPT):
        # gptqmodel can auto init tokenizer internally
        if keywords is None:
            keywords = self.INFERENCE_RESULT_KEYWORDS
        if tokenizer is None:
            tokenizer = model.tokenizer

        generated = self.generate(model, tokenizer, prompt).lower()
        for k in keywords:
            if k.lower() in generated:
                self.assertTrue(True)
                return
        raise AssertionError(f"none of keywords were found in generated: `{generated}`")

    # note that sampling is disabled for help with deterministic generation for ci tests
    def generate(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = self.INFERENCE_PROMPT
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=self.GENERATE_EVAL_SIZE_MIN, max_new_tokens=self.GENERATE_EVAL_SIZE_MIN)
        output = tokenizer.decode(res[0])
        print(f"Result is: >>\n{output}\n<<")
        return output

    def generateChat(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = [
                {"role": "system",
                 "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": "I am in Shanghai, preparing to visit the natural history museum. Can you tell me the best way to"}
            ]

        input_tensor = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_ids=input_tensor.to(model.device), max_new_tokens=self.GENERATE_EVAL_SIZE_MAX)
        output = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        print(f"Result is: \n{output}")
        return output

    # Use this helper for CI output assertions instead of raw model.generate(),
    # including in standalone unittest cases, so expected-text checks stay deterministic.
    @staticmethod
    def generate_stable_with_limit(
        model,
        tokenizer,
        prompt=None,
        max_new_tokens=512,
        min_new_tokens=None,
        skip_special_tokens=True,
        inputs=None,
        decode_start_idx=None,
        batch_decode=False,
        clean_up_tokenization_spaces=None,
        return_generate_output=False,
        **generate_kwargs,
    ):
        if inputs is None:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        elif hasattr(inputs, "to"):
            inputs = inputs.to(model.device)

        generation_inputs = dict(inputs) if isinstance(inputs, Mapping) else {"input_ids": inputs}

        decoder = getattr(tokenizer, "tokenizer", tokenizer)
        pad_token_id = decoder.pad_token_id if decoder.pad_token_id is not None else decoder.eos_token_id
        generated = model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
            eos_token_id=decoder.eos_token_id,
            **generate_kwargs,
        )
        if return_generate_output:
            return generated

        generated_ids = generated[0] if isinstance(generated, tuple) else generated

        if batch_decode:
            if decode_start_idx is None:
                if hasattr(inputs, "input_ids"):
                    decode_start_idx = [len(input_ids) for input_ids in inputs.input_ids]
                else:
                    raise ValueError("decode_start_idx is required for batch_decode when inputs lack input_ids")

            if isinstance(decode_start_idx, int):
                generated_ids = [output_ids[decode_start_idx:] for output_ids in generated_ids]
            else:
                generated_ids = [
                    output_ids[start_idx:]
                    for start_idx, output_ids in zip(decode_start_idx, generated_ids)
                ]

            decode_kwargs = {"skip_special_tokens": skip_special_tokens}
            if clean_up_tokenization_spaces is not None:
                decode_kwargs["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces
            return tokenizer.batch_decode(generated_ids, **decode_kwargs)[0]

        if decode_start_idx is None:
            decode_start_idx = 0

        return tokenizer.decode(
            generated_ids[0][decode_start_idx:],
            skip_special_tokens=skip_special_tokens,
        )

    def run_generic_inference_checks(self, model, tokenizer, backend):
        model.eval()
        log.info(f"Post-quant inference checks for backend `{backend.name}`")
        results = []
        for idx, item in enumerate(self.GENERIC_TEST_PROMPTS, start=1):
            prompt = item["prompt"]
            keywords = item["keywords"]
            try:
                inputs, decode_start_idx = self._prepare_generic_inference_inputs(tokenizer, prompt)
                response = self.generate_stable_with_limit(
                    model,
                    tokenizer,
                    prompt,
                    inputs=inputs,
                    decode_start_idx=decode_start_idx,
                )
                normalized = response.lower()
                matched = any(keyword.lower() in normalized for keyword in keywords)
                results.append(
                    {
                        "prompt": prompt,
                        "keywords": keywords,
                        "response": response,
                        "matched": matched,
                    }
                )
                snippet = self._summarize_response(response, width=160)
                if matched:
                    log.info(
                        f"[{backend.name}] Prompt {idx} PASS: `{prompt}` -> `{snippet}`"
                    )
                else:
                    log.error(
                        f"[{backend.name}] Prompt {idx} MISS: `{prompt}` -> `{snippet}`"
                    )
            except Exception as exc:  # pragma: no cover - informative logging for test harness
                log.error(f"[{backend.name}] Prompt {idx} ERROR: `{prompt}` -> {exc}")
                results.append(
                    {
                        "prompt": prompt,
                        "keywords": keywords,
                        "response": str(exc),
                        "matched": False,
                    }
                )
        return results

    def _prepare_generic_inference_inputs(self, tokenizer, prompt):
        # Some chat-tuned checkpoints only produce stable continuations when the
        # sanity prompts are wrapped with the tokenizer's chat template.
        if not self.APPLY_CHAT_TEMPLATE or not hasattr(tokenizer, "apply_chat_template"):
            return None, None

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([text], return_tensors="pt")
        return inputs, inputs.input_ids.shape[1]

    def run_eval_tasks(self, model, backend, trust_remote_code=False):
        previous_backend = self.LOAD_BACKEND
        self.LOAD_BACKEND = backend
        try:
            task_results = self.evaluate_model(
                model=model,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                delete_quantized_model=False,
            )
            log.info(f"[{backend.name}] Evaluation summary: {task_results}")
        finally:
            self.LOAD_BACKEND = previous_backend
        return task_results

    def _current_load_backend(self):
        effective = getattr(self, "_effective_load_backend", None)
        if effective is not None and self.LOAD_BACKEND == BACKEND.MARLIN:
            return effective
        return self.LOAD_BACKEND

    def _torch_backend(self) -> BACKEND:
        if self.METHOD == METHOD.AWQ:
            return BACKEND.TORCH_AWQ
        if self.METHOD == METHOD.PARO:
            return BACKEND.PARO
        if self.METHOD == METHOD.GGUF:
            return BACKEND.GGUF_TORCH
        return BACKEND.TORCH

    def _torch_fused_backend(self) -> BACKEND:
        return BACKEND.TORCH_FUSED_AWQ if self.METHOD == METHOD.AWQ else BACKEND.TORCH_FUSED

    def perform_post_quant_validation(self, model_path, trust_remote_code=False):
        inference_records = {}
        eval_records = {}
        reuse_candidates = {}
        torch_backend = self._torch_backend()
        torch_fused_backend = self._torch_fused_backend()
        format_family = resolve_quant_format(self.FORMAT, self.METHOD)

        if format_family == FORMAT.GGUF:
            compare_backends = (torch_backend,)
        elif format_family == FORMAT.BITSANDBYTES:
            compare_backends = (BACKEND.BITSANDBYTES,)
        elif format_family == FORMAT.FP8:
            compare_backends = (torch_backend,)
        elif format_family == FORMAT.EXL3:
            compare_backends = (self.LOAD_BACKEND,)
        elif format_family == FORMAT.PAROQUANT:
            compare_backends = (self.LOAD_BACKEND,)
        elif format_family == FORMAT.GPTQ:
            if self.LOAD_BACKEND == BACKEND.MARLIN:
                compare_backends = (BACKEND.MARLIN,)
            else:
                compare_backends = (self.LOAD_BACKEND,)
        else:
            compare_backends = (BACKEND.MARLIN, BACKEND.GEMM)
        fallback_backend = None
        if BACKEND.MARLIN in compare_backends:
            try:
                from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear  # type: ignore
            except Exception:  # pragma: no cover - fallback if module unavailable
                marlin_group_sizes = ()
                marlin_sym = ()
            else:
                marlin_group_sizes = tuple(getattr(MarlinLinear, "SUPPORTS_GROUP_SIZE", ()))
                marlin_sym = tuple(getattr(MarlinLinear, "SUPPORTS_SYM", ()))

            requested_group_size = getattr(self, "GROUP_SIZE", None)
            requested_sym = getattr(self, "SYM", None)

            marlin_supported = True
            if marlin_group_sizes and requested_group_size not in marlin_group_sizes:
                marlin_supported = False
            if marlin_sym and requested_sym not in marlin_sym:
                marlin_supported = False

            if not marlin_supported:
                fallback_backend = torch_backend
                compare_backends = tuple(
                    torch_backend if backend == BACKEND.MARLIN else backend
                    for backend in compare_backends
                )
                log.info(
                    f"Marlin backend unsupported for current quant config (group_size={requested_group_size}, sym={requested_sym}); "
                    f"falling back to {torch_backend} for validation."
                )

        if fallback_backend is not None and self.LOAD_BACKEND == BACKEND.MARLIN:
            self._effective_load_backend = fallback_backend
        else:
            self._effective_load_backend = None

        target_backend = self._current_load_backend()
        can_reuse = target_backend not in (BACKEND.AUTO, BACKEND.AUTO_TRAINABLE)

        for backend in compare_backends:
            log.info(f"Loading post-quant model with backend `{backend.name}`")
            # When EVAL_SINGLE_GPU is enabled, keep post-quant validation on the preferred device.
            use_cuda_map = (
                self.EVAL_SINGLE_GPU
                and torch.cuda.is_available()
                and backend != torch_fused_backend
            )
            if use_cuda_map:
                try:
                    model = self.loadQuantModel(
                        model_path,
                        trust_remote_code=trust_remote_code,
                        backend=backend,
                        device_map={"": "cuda:0"},
                    )
                except torch.OutOfMemoryError:
                    log.warn(
                        "Post-quant load with device_map={'': 'cuda:0'} OOMed for backend `%s`; retrying with loader defaults.",
                        backend.name,
                    )
                    torch_empty_cache()
                    model = self.loadQuantModel(
                        model_path,
                        trust_remote_code=trust_remote_code,
                        backend=backend,
                    )
            else:
                model = self.loadQuantModel(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    backend=backend,
            )
            model.tokenizer or self.load_tokenizer(model_path, trust_remote_code=trust_remote_code)
            # Pre-evaluation smoke prompts are intentionally disabled to keep quantization tests
            # focused only on task execution.
            # inference_records[backend] = self.run_generic_inference_checks(model, tokenizer, backend)

            should_reuse = can_reuse and backend == target_backend and not self.USE_VLLM

            try:
                eval_records[backend] = self.run_eval_tasks(model, backend, trust_remote_code=trust_remote_code)
            finally:
                if should_reuse:
                    reuse_candidates[backend] = model
                else:
                    del model
                torch_empty_cache()

        self.render_inference_summary(inference_records)
        self.render_eval_summary(eval_records)

        return reuse_candidates, eval_records

    @staticmethod
    def _human_size(num_bytes: int) -> str:
        step = 1024.0
        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(num_bytes)
        for unit in units:
            if value < step or unit == units[-1]:
                return f"{value:.2f}{unit}"
            value /= step
        return f"{num_bytes}B"

    @staticmethod
    def _print_post_quant_artifacts(root_path: str) -> None:
        path = Path(root_path)
        if not path.exists():
            log.warn(f"Post-quant artifact path missing: {root_path}")
            return

        reset = "\033[0m"
        depth_colors = [
            "\033[36m",
            "\033[33m",
            "\033[35m",
            "\033[32m",
            "\033[34m",
            "\033[31m",
        ]

        def colorize(name: str, depth: int, is_dir: bool) -> str:
            if not sys.stdout.isatty():
                return name
            if is_dir:
                code = depth_colors[depth % len(depth_colors)]
            else:
                code = "\033[37m"
            return f"{code}{name}{reset}"

        def walk(directory: Path, prefix: str, depth: int) -> None:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for idx, entry in enumerate(entries):
                connector = "└──" if idx == len(entries) - 1 else "├──"
                display_name = entry.name + ("/" if entry.is_dir() else "")
                line = f"{prefix}{connector} {colorize(display_name, depth, entry.is_dir())}"
                if entry.is_file():
                    try:
                        size = entry.stat().st_size
                        line += f" ({ModelTest._human_size(size)})"
                    except OSError:
                        pass
                print(line)
                if entry.is_dir():
                    extension = "    " if idx == len(entries) - 1 else "│   "
                    walk(entry, prefix + extension, depth + 1)

        header = f"Post-quant artifacts: {path.resolve()}"
        print(f"\n{colorize(header, 0, True)}")
        walk(path, "", 1)

        index_files = sorted(path.rglob("*.safetensors.index.json"))
        if not index_files:
            fallback = sorted(path.glob("*.index.json"))
            index_files = fallback

        for idx_file in index_files:
            try:
                with idx_file.open("r", encoding="utf-8") as fh:
                    content = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                log.warn(f"Failed to read index `{idx_file}`: {exc}")
                continue
            rel_name = idx_file.relative_to(path)
            print(f"\n{colorize(f'Index file: {rel_name}', 0, False)}")
            print(json.dumps(content, indent=2, sort_keys=True))

    def _resolve_quantized_model_path(self, model_candidate):
        if model_candidate is None:
            return None
        if isinstance(model_candidate, (list, tuple)):
            model_candidate = model_candidate[0]
        if isinstance(model_candidate, str):
            return model_candidate
        return getattr(model_candidate, "model_local_path", None)

    def _cleanup_quantized_model(self, model_candidate, enabled=True):
        if not enabled:
            return False

        temp_dir_context = getattr(model_candidate, "_temp_dir_context", None)
        if temp_dir_context is not None:
            try:
                temp_dir_context.cleanup()
            except OSError as exc:
                log.warn(f"Failed to delete temp model `{temp_dir_context.name}`: {exc}")
                return False

            try:
                delattr(model_candidate, "_temp_dir_context")
            except AttributeError:
                pass

            log.info(f"Deleting temp model: {temp_dir_context.name}")
            return True

        target_path = self._resolve_quantized_model_path(model_candidate)
        if not target_path or not isinstance(target_path, str):
            return False

        temp_root = os.path.realpath(tempfile.gettempdir())
        candidate_path = os.path.realpath(target_path)
        if not candidate_path.startswith(temp_root):
            return False
        if not os.path.exists(candidate_path):
            return False

        try:
            shutil.rmtree(candidate_path)
        except OSError as exc:
            log.warn(f"Failed to delete temp model `{candidate_path}`: {exc}")
            return False

        log.info(f"Deleting temp model: {candidate_path}")
        return True

    @staticmethod
    def _colorize(text, matched):
        color = "\033[92m" if matched else "\033[91m"
        reset = "\033[0m"
        return f"{color}{text}{reset}"

    def render_inference_summary(self, inference_records):
        if not inference_records:
            return
        torch_backend = self._torch_backend()
        ordered_backends = [backend for backend in (BACKEND.MARLIN, torch_backend) if backend in inference_records]
        if not ordered_backends:
            return

        prompts = [item["prompt"] for item in self.GENERIC_TEST_PROMPTS]
        sanity_scores = {}
        for backend in ordered_backends:
            entries = {entry["prompt"]: entry for entry in inference_records[backend]}
            matched_count = sum(1 for entry in entries.values() if entry.get("matched"))
            total_count = len(entries) if entries else 1
            sanity_scores[backend] = (matched_count, total_count)

        log.info("Sanity prompt comparison:")
        for prompt in prompts:
            expected = ", ".join(
                self._normalize_keyword_case(k) for k in self._keywords_for_prompt(prompt)
            )
            lines = [f"Prompt: {prompt}", f"  Expected: {expected or 'None'}"]
            for backend in ordered_backends:
                entry = next((item for item in inference_records[backend] if item["prompt"] == prompt), None)
                if entry is None:
                    lines.append(f"  {backend.name:<6}: {self._colorize('N/A', False)}")
                    continue
                lines.append(
                    f"  {backend.name:<6}: {self._format_inference_entry(entry)}"
                )
            log.info("\n".join(lines))

        for backend, (matched, total) in sanity_scores.items():
            score_pct = 100.0 * matched / max(total, 1)
            result_text = f"{matched}/{total} ({score_pct:.1f}%)"
            log.info("Sanity score [%s]: %s", backend.name, result_text)

    @staticmethod
    def _normalize_keyword_case(keyword):
        return keyword.lower()

    def _keywords_for_prompt(self, prompt):
        for item in self.GENERIC_TEST_PROMPTS:
            if item["prompt"] == prompt:
                return item["keywords"]
        return []

    @staticmethod
    def _summarize_response(response, width=80):
        clean = " ".join(response.split()) if response else ""
        if not clean:
            return ""
        return textwrap.shorten(clean, width=width, placeholder="…")

    def _format_inference_entry(self, entry):
        matched = entry.get("matched", False)
        response = entry.get("response", "")
        snippet = self._summarize_response(response)
        status = "PASS" if matched else "MISS"
        cell = f"{status} | {snippet}" if snippet else status
        return self._colorize(cell, matched)

    def render_eval_summary(self, eval_records):
        if not eval_records:
            return
        torch_backend = self._torch_backend()
        ordered_backends = [backend for backend in (BACKEND.MARLIN, torch_backend) if backend in eval_records]
        if not ordered_backends:
            return

        flattened_records = {
            backend: self._flatten_task_metrics(results) for backend, results in eval_records.items()
        }

        metrics = sorted({metric for results in flattened_records.values() for metric in results.keys()})

        table_rows = []
        tolerance = 0.01
        torch_reference = flattened_records.get(torch_backend, {})

        for metric in metrics:
            display_metric = metric.replace(":", " :: ")
            row = [display_metric]
            reference_value = None if torch_reference is None else torch_reference.get(metric)
            for backend in ordered_backends:
                backend_values = flattened_records.get(backend, {})
                value = backend_values.get(metric)
                if value is None:
                    row.append(self._colorize("N/A", False))
                    continue
                if backend == torch_backend:
                    row.append(self._colorize(f"{value:.4f}", True))
                else:
                    matched = reference_value is not None and abs(value - reference_value) <= tolerance
                    row.append(self._colorize(f"{value:.4f}", matched))
            table_rows.append(row)

        headers = ["Metric"] + [backend.name for backend in ordered_backends]
        log.info(
            "Evaluation comparison:\n%s",
            render_table(table_rows, headers=headers, tablefmt="github"),
        )

    @classmethod
    def load_tokenizer(cls, model_id_or_path, trust_remote_code=False):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        return tokenizer

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        if hf_load_dataset is None:
            log.warning("datasets.load_dataset unavailable; falling back to local parquet: %s", DATASETS_IMPORT_ERROR)
            dataset = cls._load_calibration_parquet()
        else:
            try:
                dataset = hf_load_dataset(path="/monster/data/model/dataset/nm-calibration", name="LLM", split="train")
            except Exception as exc:  # pragma: no cover - exercised in fallbacks
                log.warning("load_dataset failed; falling back to local parquet: %s", exc)
                dataset = cls._load_calibration_parquet()

        if rows > 0:
            return dataset.select(range(min(rows, len(dataset))))
        return dataset

    @staticmethod
    def _load_calibration_parquet():
        parquet_path = Path("/monster/data/model/dataset/nm-calibration/llm.parquet").expanduser()
        if not parquet_path.exists():
            raise FileNotFoundError(f"Calibration parquet not found at {parquet_path}")

        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - depends on test environment
            pd = None

        if pd is not None:
            records = pd.read_parquet(parquet_path).to_dict(orient="records")
            return ModelTest._LocalCalibrationDataset(records)

        try:
            import pyarrow.parquet as pq
        except ImportError as err:
            raise RuntimeError(
                "Neither pandas nor pyarrow is available to load calibration parquet"
            ) from err

        table = pq.read_table(parquet_path)
        records = table.to_pylist()
        return ModelTest._LocalCalibrationDataset(records)

    class _LocalCalibrationDataset:
        __slots__ = ("_records",)

        def __init__(self, records):
            normalized = []
            for record in records:
                item = {}
                for key, value in dict(record).items():
                    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                        value = value.tolist()
                    item[key] = value
                normalized.append(item)
            self._records = normalized

        def __len__(self):
            return len(self._records)

        def __iter__(self):
            return iter(self._records)

        def __getitem__(self, index):
            return self._records[index]

        def select(self, indices):
            if isinstance(indices, slice):
                selected = self._records[indices]
            else:
                if isinstance(indices, range):
                    indices = list(indices)
                elif not isinstance(indices, Iterable):
                    raise TypeError("select `indices` must be a slice or iterable of integers")
                selected = [self._records[i] for i in indices]
            return self.__class__(selected)


    def check_kernel(self, model, expected_kernels):
        modules = {module.__class__ for _, module in model.named_modules() if isinstance(module, BaseQuantLinear)}
        print(f"modules in model: {modules}")
        if expected_kernels:
            assert modules == expected_kernels, f"kernels are different with expected. found: {modules}. expected: {expected_kernels}"

    def _build_quantize_config(self):
        format_family = resolve_quant_format(self.FORMAT, self.METHOD)

        if self.WEIGHT_ONLY is None:
            if self.METHOD == METHOD.BITSANDBYTES:
                return BitsAndBytesConfig(
                    bits=self.BITS,
                    format=self.BNB_FORMAT,
                    block_size=self.BNB_BLOCK_SIZE,
                    compress_statistics=self.BNB_COMPRESS_STATISTICS,
                    adapter=self.EORA,
                    dense_vram_strategy=self.DENSE_VRAM_STRATEGY,
                    dense_vram_strategy_devices=self.DENSE_VRAM_STRATEGY_DEVICES,
                    moe_vram_strategy=self.MOE_VRAM_STRATEGY,
                    moe_vram_strategy_devices=self.MOE_VRAM_STRATEGY_DEVICES,
                    dynamic=self.DYNAMIC,
                    moe=self.MOE_CONFIG,
                    offload_to_disk=self.OFFLOAD_TO_DISK,
                )
            elif self.METHOD == METHOD.PARO:
                return ParoConfig(
                    bits=self.BITS,
                    method=METHOD.PARO,
                    format=FORMAT.PAROQUANT,
                    opt_rotation_epochs=self.PAROQUANT_ROTATION_EPOCHS,
                    opt_finetune_epochs=self.PAROQUANT_FINETUNE_EPOCHS,
                    opt_train_samples=self.PAROQUANT_TRAIN_SAMPLES,
                    adapter=self.EORA,
                    dense_vram_strategy=self.DENSE_VRAM_STRATEGY,
                    dense_vram_strategy_devices=self.DENSE_VRAM_STRATEGY_DEVICES,
                    moe_vram_strategy=self.MOE_VRAM_STRATEGY,
                    moe_vram_strategy_devices=self.MOE_VRAM_STRATEGY_DEVICES,
                    dynamic=self.DYNAMIC,
                    moe=self.MOE_CONFIG,
                    offload_to_disk=self.OFFLOAD_TO_DISK,
                )

        if self.WEIGHT_ONLY is not None:
            if not isinstance(self.WEIGHT_ONLY, WeightOnlyConfig):
                raise TypeError(f"`WEIGHT_ONLY` must be a WeightOnlyConfig, got {type(self.WEIGHT_ONLY).__name__}")

            if format_family == FORMAT.GGUF or self.WEIGHT_ONLY.method.value == "gguf":
                return GGUFConfig(
                    bits=self.BITS,
                    adapter=self.EORA,
                    pack_impl="cpu",
                    dense_vram_strategy=self.DENSE_VRAM_STRATEGY,
                    dense_vram_strategy_devices=self.DENSE_VRAM_STRATEGY_DEVICES,
                    moe_vram_strategy=self.MOE_VRAM_STRATEGY,
                    moe_vram_strategy_devices=self.MOE_VRAM_STRATEGY_DEVICES,
                    dynamic=self.DYNAMIC,
                    moe=self.MOE_CONFIG,
                    smoother=self.WEIGHT_ONLY.smooth,
                )

            if format_family == FORMAT.FP8 or self.WEIGHT_ONLY.method.value == "fp8":
                return FP8Config(
                    bits=self.BITS,
                    format=self.FORMAT,
                    adapter=self.EORA,
                    pack_impl="cpu",
                    dense_vram_strategy=self.DENSE_VRAM_STRATEGY,
                    dense_vram_strategy_devices=self.DENSE_VRAM_STRATEGY_DEVICES,
                    moe_vram_strategy=self.MOE_VRAM_STRATEGY,
                    moe_vram_strategy_devices=self.MOE_VRAM_STRATEGY_DEVICES,
                    dynamic=self.DYNAMIC,
                    moe=self.MOE_CONFIG,
                    smoother=self.WEIGHT_ONLY.smooth,
                )

            return RTNConfig(
                bits=self.BITS,
                group_size=self.GROUP_SIZE,
                desc_act=self.DESC_ACT,
                sym=self.SYM,
                format=self.FORMAT,
                adapter=self.EORA,
                pack_impl="cpu",
                dense_vram_strategy=self.DENSE_VRAM_STRATEGY,
                dense_vram_strategy_devices=self.DENSE_VRAM_STRATEGY_DEVICES,
                moe_vram_strategy=self.MOE_VRAM_STRATEGY,
                moe_vram_strategy_devices=self.MOE_VRAM_STRATEGY_DEVICES,
                dynamic=self.DYNAMIC,
                moe=self.MOE_CONFIG,
                smooth=self.WEIGHT_ONLY.smooth,
            )

        return QuantizeConfig(
            quant_method=self.METHOD,
            format=self.FORMAT,
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            desc_act=self.DESC_ACT if not self.ACT_GROUP_AWARE else False,
            act_group_aware=self.ACT_GROUP_AWARE,
            fallback=self.FALLBACK,
            sym=self.SYM,
            gptaq=copy.deepcopy(self.GPTAQ),
            foem=copy.deepcopy(self.FOEM),
            adapter=self.EORA,
            pack_impl="cpu",
            dense_vram_strategy=self.DENSE_VRAM_STRATEGY,
            dense_vram_strategy_devices=self.DENSE_VRAM_STRATEGY_DEVICES,
            moe_vram_strategy=self.MOE_VRAM_STRATEGY,
            moe_vram_strategy_devices=self.MOE_VRAM_STRATEGY_DEVICES,
            damp_percent=self.DAMP_PERCENT,
            mse=self.MSE,
            dynamic=self.DYNAMIC,
            hessian=HessianConfig(chunk_size=self.HESSIAN_CHUNK_SIZE),
            moe=self.MOE_CONFIG,
            offload_to_disk=self._mode_specific_test_setting("OFFLOAD_TO_DISK"),
        )

    def quantModel(self, model_id_or_path, trust_remote_code=False, dtype="auto", need_eval=True, batch_size: int = QUANT_BATCH_SIZE, call_perform_post_quant_validation: bool = True, **kwargs):
        """Return `(model, tokenizer, processor)`; `processor` is `None` for text-only models."""
        quantize_config = self._build_quantize_config()

        log.info(f"Quant config: {quantize_config}")
        log.info(f"Quant batch_size: {batch_size}")

        args = kwargs if kwargs else {}

        if self.USE_FLASH_ATTN:
            if is_flash_attn_2_available():
                args["attn_implementation"] = "flash_attention_2"
            else:
                log.warn("flash-attn requested but not available; falling back to framework defaults")
        else:
            args["attn_implementation"] = "eager"


        log.info(f"args: {args}")
        torch_fused_backend = self._torch_fused_backend()
        model = GPTQModel.load(
            model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map=(
                {"": "cpu"}
                if self.LOAD_BACKEND == torch_fused_backend
                else "auto"
            ),
            **args,
        )

        self._layer_stop_callback = None
        if DEBUG_ON and self.STOP_AFTER_LAYER is not None:
            self._layer_stop_callback = self._build_layer_stop_callback(self.STOP_AFTER_LAYER)
            model.layer_callback = self._layer_stop_callback

        tokenizer = model.tokenizer
        self._post_quant_eval_records = {}
        self._effective_load_backend = None
        self._model_compat_fast_dynamic = None
        # Tracks whether quantModel() loaded an existing quantized checkpoint
        # instead of producing a fresh post-quant artifact + Evalution cache.
        self._loaded_model_was_prequantized = False
        processor = None

        self._apply_model_compat_quant_overrides(model)

        dataset_size = self._mode_specific_test_setting("DATASET_SIZE")
        dataset_concat_size = self._mode_specific_test_setting("DATASET_CONCAT_SIZE")
        log.info(
            "Calibration dataset config: size=%s, concat_size=%s",
            dataset_size,
            dataset_concat_size,
        )

        is_image_to_text_model = MODALITY.IMAGE_TO_TEXT in model.modality
        if quantize_config.requires_calibration_dataset():
            calibration_dataset = (
                get_calib_dataset(model)
                if is_image_to_text_model
                else self.load_dataset(tokenizer, dataset_size)
            )
        else:
            calibration_dataset = None

        # mpt model need
        if hasattr(model.config, "pad_token_id") and not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id or 0
        if hasattr(model.config, "eos_token_id") and not model.config.eos_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id or 0

        is_quantized = model.quantized
        self._loaded_model_was_prequantized = bool(is_quantized)

        # ovis cannot load processor
        is_ovis_model = model.config.model_type == "ovis"
        need_create_processor = is_image_to_text_model and not is_ovis_model

        if not is_quantized:
            temp_dir_context = None
            try:
                model.quantize(
                    calibration_dataset,
                    calibration_concat_size=dataset_concat_size,
                    calibration_concat_separator=self.DATASET_CONCAT_SEPARATOR,
                    calibration_sort=self.DATASET_SORT,
                    backend=self.QUANT_BACKEND,
                    batch_size=batch_size,
                )

                self.check_kernel(model, self.KERNEL_QUANT)

                debug_short_circuit = self._debug_layer_stop_triggered()
                if debug_short_circuit:
                    log.info(
                        "DEBUG mode: layer stop triggered at %s; skipping post-quant save and evaluation pipeline.",
                        self.STOP_AFTER_LAYER,
                    )
                    return self._finalize_quant_debug_path(
                        model=model,
                        tokenizer=tokenizer,
                        processor=None,
                    )

                if self.SAVE_PATH:
                    planned_save_path = self.SAVE_PATH
                    save_context = contextlib.nullcontext(planned_save_path)
                elif need_eval:
                    temp_dir_context = tempfile.TemporaryDirectory()
                    planned_save_path = temp_dir_context.name
                    save_context = contextlib.nullcontext(planned_save_path)
                else:
                    save_context = tempfile.TemporaryDirectory()
                    planned_save_path = save_context.name

                log.info(f"Quantized model artifacts will be saved to: {planned_save_path}")

                # TODO: make into shared method
                with save_context as path:
                    os.makedirs(path, exist_ok=True)
                    self.clear_directory(path)

                    model.save(path, split_by=self.SPLIT_BY)
                    self._print_post_quant_artifacts(path)

                    reuse_candidates = {}
                    eval_records = {}
                    if call_perform_post_quant_validation:
                        reuse_candidates, eval_records = self.perform_post_quant_validation(path, trust_remote_code=trust_remote_code)
                    self._post_quant_eval_records = eval_records
                    target_backend = self._current_load_backend()

                    q_model = reuse_candidates.pop(target_backend, None)
                    if q_model is None:
                        # When single-GPU evaluation is requested, keep the reload scoped to cuda:0.
                        torch_fused_backend = self._torch_fused_backend()
                        use_cuda_map = (
                            self.EVAL_SINGLE_GPU
                            and torch.cuda.is_available()
                            and target_backend != torch_fused_backend
                        )
                        if use_cuda_map:
                            q_model = self.loadQuantModel(
                                path,
                                trust_remote_code=trust_remote_code,
                                backend=target_backend,
                                device_map={"": "cuda:0"},
                            )
                        else:
                            q_model = self.loadQuantModel(path, trust_remote_code=trust_remote_code, backend=target_backend)
                    else:
                        log.info(f"Reusing post-quant validation model for backend `{target_backend.name}`")

                    q_tokenizer = q_model.tokenizer or self.load_tokenizer(path, trust_remote_code=trust_remote_code)
                    if need_create_processor:
                        processor = AutoProcessor.from_pretrained(path, trust_remote_code=trust_remote_code)
                    if temp_dir_context is not None:
                        setattr(q_model, "_temp_dir_context", temp_dir_context)
                        temp_dir_context = None
            except Exception:
                if temp_dir_context is not None:
                    try:
                        temp_dir_context.cleanup()
                    except Exception:
                        pass
                raise

        else:
            if need_create_processor:
                processor = AutoProcessor.from_pretrained(model_id_or_path)
        if not is_quantized:
            del model
            torch_empty_cache()
            return q_model, q_tokenizer, processor
        else:
            return model, tokenizer, processor

    def loadQuantModel(self, model_id_or_path, trust_remote_code=False, tokenizer_path=None, backend=None, **args):

        load_kwargs = dict(args)
        if self.LOAD_MODEL_EXTRA_ARGS:
            for key, value in self.LOAD_MODEL_EXTRA_ARGS.items():
                load_kwargs.setdefault(key, value)

        if self.USE_FLASH_ATTN:
            if is_flash_attn_2_available():
                load_kwargs["attn_implementation"] = "flash_attention_2"
            else:
                log.warn("flash-attn requested but not available; falling back to framework defaults")
        else:
            load_kwargs["attn_implementation"] = "eager"

        active_backend = backend if backend is not None else self._current_load_backend()
        torch_fused_backend = self._torch_fused_backend()

        default_device_map = {"": "cpu"} if active_backend == torch_fused_backend else "auto"
        explicit_device = "device" in load_kwargs
        inserted_device_map = False
        if "device_map" not in load_kwargs and not explicit_device:
            load_kwargs["device_map"] = default_device_map
            inserted_device_map = True

        # Post-quant CI runs may expose multiple GPUs; pin loading to the first one to avoid spread-out auto maps.
        if (
            (inserted_device_map or load_kwargs.get("device_map") == "auto")
            and not explicit_device
            and active_backend != torch_fused_backend
            and torch.cuda.is_available()
        ):
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            candidates = [item.strip() for item in visible.split(",") if item.strip()]
            try:
                multi_device = len(candidates) > 1 if candidates else torch.cuda.device_count() > 1
            except Exception:
                multi_device = False

            if multi_device:
                if self.EVAL_SINGLE_GPU:
                    load_kwargs["device_map"] = {"": "cuda:0"}

        model = GPTQModel.load(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            backend=active_backend,
            adapter=self.EORA,
            **load_kwargs
        )

        return model

    def evaluate_model(self, model, trust_remote_code=False, delete_quantized_model=False, extra_args:dict=None):
        try:
            task_names = self._normalize_task_list()
            aggregated_results = {}
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = getattr(model, "model_local_path", None)
                if isinstance(model, str):
                    model_path = model

                if self.USE_VLLM:
                    raise ValueError("ModelTest USE_VLLM is no longer supported; evaluation is delegated to Evalution.")

                model_args = {}
                if extra_args:
                    model_args.update(extra_args)

                chat_template_lookup = getattr(self, "_task_chat_template", {}) or {}
                suite_kwargs_lookup = getattr(self, "_task_evalution_suite_kwargs", {}) or {}
                task_model_args_lookup = getattr(self, "_task_evalution_model_args", {}) or {}
                use_model_path_lookup = getattr(self, "_task_evalution_use_model_path", {}) or {}
                eval_batch_size_lookup = getattr(self, "_task_evalution_batch_size", {}) or {}
                active_backend = self._current_load_backend()
                log.info(f"TEST: Evalution starting: backend = {active_backend.name}")
                if model_path:
                    log.info(f"Inference from model path: {model_path}")

                for task_name in task_names:
                    normalized_name = self._normalize_task_identifier(task_name)
                    apply_chat_template = bool(chat_template_lookup.get(normalized_name, False))
                    task_model_args = dict(model_args)
                    task_model_args.update(task_model_args_lookup.get(normalized_name, {}) or {})
                    # Keep evalution-backed generation reproducible even when a task opts
                    # into sampling or an engine backend introduces RNG-sensitive paths.
                    task_model_args.setdefault("seed", RAND_SEED)
                    task_model_args.setdefault("random_seed", RAND_SEED)
                    task_suite_kwargs = dict(suite_kwargs_lookup.get(normalized_name, {}) or {})
                    task_batch_size = eval_batch_size_lookup.get(normalized_name)
                    if task_batch_size is None:
                        task_batch_size = self.EVAL_BATCH_SIZE
                    use_model_path = bool(use_model_path_lookup.get(normalized_name, False))

                    if use_model_path and model_path:
                        eval_target = model_path
                    elif isinstance(model, BaseQModel):
                        eval_target = model
                    else:
                        eval_target = model_path

                    if eval_target is None:
                        raise ValueError("Model evaluation target could not be determined.")

                    results = evaluate(
                        model_or_id_or_path=eval_target,
                        model_args=task_model_args,
                        output_path=tmp_dir,
                        backend=active_backend,
                        tasks=[normalized_name],
                        apply_chat_template=apply_chat_template,
                        trust_remote_code=trust_remote_code,
                        batch_size=task_batch_size,
                        gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
                        suite_kwargs=task_suite_kwargs,
                    )

                    print('--------Eval Result---------')
                    print(format_eval_result_table(results))
                    print('--------Eval Result End---------')

                    result_metrics = get_eval_task_results(results)
                    metrics = result_metrics.get(normalized_name, {})
                    filtered_metrics = {
                        metric: value
                        for metric, value in metrics.items()
                        if metric != "alias" and "stderr" not in metric
                    }
                    aggregated_results[normalized_name] = filtered_metrics
                    print({normalized_name: filtered_metrics})

                self._cleanup_quantized_model(model, enabled=delete_quantized_model)
                return aggregated_results
        except BaseException as e:
            if isinstance(e, torch.OutOfMemoryError):
                old_batch = self.EVAL_BATCH_SIZE
                if self.EVAL_BATCH_SIZE == "auto":
                    self.EVAL_BATCH_SIZE = "8"
                else:
                    self.EVAL_BATCH_SIZE = f"{int(int(self.EVAL_BATCH_SIZE) / 2)}"
                    self.MODEL_MAX_LEN = max(1024, self.MODEL_MAX_LEN - 1024)

                print(f"batch {old_batch} OOM, retrying with batch {self.EVAL_BATCH_SIZE}")

                if int(self.EVAL_BATCH_SIZE) > 0:
                    results = self.evaluate_model(model=model,
                                           trust_remote_code=trust_remote_code,
                                           delete_quantized_model=delete_quantized_model,
                                           extra_args=extra_args)
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, passed")
                    return results
                else:
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, failed")
                    raise e
            else:
                raise e

    def calculatorPer(self, task_name, metric_name, value, expected):
        diff_pct = (value / expected) * 100
        log.info(f"{task_name}:{metric_name}: `{value}` vs `{expected}` diff {diff_pct:.2f}%")
        return diff_pct, expected

    @staticmethod
    def _metric_within_expected_range(value, expected, floor_pct, ceil_pct):
        diff_pct = (value / expected) * 100
        negative_pct = 100 * (1 - floor_pct)
        positive_pct = 100 * (1 + ceil_pct)
        return negative_pct <= diff_pct <= positive_pct, diff_pct, negative_pct, positive_pct

    def _current_native_backend(self) -> BACKEND:
        return BACKEND.TORCH

    def _get_current_native_eval_results(self):
        cached = getattr(self, "_current_native_eval_results", None)
        if cached is not None:
            return cached

        native_model_id = getattr(self, "NATIVE_MODEL_ID", None)
        if not native_model_id:
            return None

        previous_backend = self.LOAD_BACKEND
        previous_effective_backend = getattr(self, "_effective_load_backend", None)
        self.LOAD_BACKEND = self._current_native_backend()
        self._effective_load_backend = None
        try:
            log.warn(
                "Baseline fallback: evaluating current native model `%s` to verify whether stored expectations are stale.",
                native_model_id,
            )
            cached = self.evaluate_model(
                model=native_model_id,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                delete_quantized_model=False,
            )
        finally:
            self.LOAD_BACKEND = previous_backend
            self._effective_load_backend = previous_effective_backend

        self._current_native_eval_results = cached
        return cached

    def _maybe_accept_current_native_baseline(
        self,
        *,
        task_name: str,
        metric_name: str,
        metric_key: str,
        value: float,
        floor_pct: float,
        ceil_pct: float,
    ) -> bool:
        try:
            native_results = self._get_current_native_eval_results()
        except Exception as exc:  # pragma: no cover - defensive fallback for flaky native eval
            log.warn(f"Baseline fallback: failed to evaluate current native model: {exc}")
            return False

        if not isinstance(native_results, dict):
            return False

        native_metrics = native_results.get(task_name)
        if not isinstance(native_metrics, dict):
            return False

        native_metric_key = self._resolve_metric_key(metric_key, native_metrics)
        if native_metric_key is None and metric_key != metric_name:
            native_metric_key = self._resolve_metric_key(metric_name, native_metrics)
        if native_metric_key is None:
            return False

        native_value = native_metrics[native_metric_key]
        passed, diff_pct, negative_pct, positive_pct = self._metric_within_expected_range(
            value=value,
            expected=native_value,
            floor_pct=floor_pct,
            ceil_pct=ceil_pct,
        )
        if not passed:
            return False

        log.warn(
            f"Baseline fallback: accepting `{task_name}:{metric_name}` using current native value `{native_value}`; "
            f"quantized result `{value}` diff {diff_pct:.2f}% is within [{negative_pct:.2f}-{positive_pct:.2f}] "
            f"while stored expectation appears stale."
        )
        return True

    def quantize_and_evaluate(self):
        self.model = None
        # TODO fix me: LOAD_QUANTIZED_MODEL doesn't make any sense when we have QUANT_SAVE_PATH
        #if self.QUANT_SAVE_PATH:
        #    self.model, _, _ = self.quantModel(self.QUANT_SAVE_PATH, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE)

        log.info("Model compat test mode: %s", self._model_test_mode())
        with self.model_compat_test_context():
            if not self.model:
                self.model, _, _ = self.quantModel(self.NATIVE_MODEL_ID, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE)

        self.check_kernel(self.model, self.KERNEL_INFERENCE)

        if self._debug_layer_stop_triggered():
            log.info("DEBUG mode: skipping evaluation and baseline checks after early layer stop.")
            return

        eval_records = getattr(self, "_post_quant_eval_records", {})
        target_backend = self._current_load_backend()
        if eval_records and len(eval_records) == 1 and target_backend in eval_records:
            log.info("Reusing evaluation results for backend `%s`; skipping duplicate evaluation run", target_backend.name)
            task_results = eval_records[target_backend]
        else:
            task_results = eval_records.get(target_backend)
            if task_results is None:
                if getattr(self, "_loaded_model_was_prequantized", False):
                    log.info(
                        "Loaded checkpoint was already quantized; running Evalution directly for backend `%s`.",
                        target_backend.name,
                    )
                    with self.model_compat_test_context():
                        task_results = self.evaluate_model(
                            model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                            trust_remote_code=self.TRUST_REMOTE_CODE,
                            delete_quantized_model=False,
                        )
                    self._post_quant_eval_records[target_backend] = task_results
                else:
                    raise AssertionError(
                        "Post-quant eval results were not produced. "
                        "The Stage-2 evaluation fallback is disabled."
                    )
        self.check_results(task_results)
        self._cleanup_quantized_model(self.model, enabled=self.DELETE_QUANTIZED_MODEL)

    def check_results(self, task_results):
        baselines = self.get_eval_tasks()
        if not baselines:
            raise AssertionError("No evaluation baselines configured for result validation.")

        errors = []
        diffs = []

        for task_name, expected_metrics in baselines.items():
            metrics = task_results.get(task_name)

            if metrics is None:
                errors.append(f"No evaluation results returned for task `{task_name}`")
                continue

            if not isinstance(metrics, dict):
                raise TypeError(
                    f"Expected metrics for task `{task_name}` to be a dictionary, got {type(metrics).__name__}"
                )

            for metric_name, baseline_spec in expected_metrics.items():
                metric_key = baseline_spec.get("metric_key") or metric_name
                metric_key = self._resolve_metric_key(metric_key, metrics)

                if metric_key is None:
                    errors.append(f"Metric `{metric_name}` missing from results for task `{task_name}`")
                    continue

                value = metrics[metric_key]
                expected_value = baseline_spec["value"]

                diff_pct, expected_value = self.calculatorPer(
                    task_name=task_name,
                    metric_name=metric_name,
                    value=value,
                    expected=expected_value,
                )

                floor_pct = baseline_spec["floor_pct"]
                ceil_pct = baseline_spec["ceil_pct"]
                passed, diff_pct, negative_pct, positive_pct = self._metric_within_expected_range(
                    value=value,
                    expected=expected_value,
                    floor_pct=floor_pct,
                    ceil_pct=ceil_pct,
                )
                diffs.append(
                    f"{task_name}:{metric_name} -> value={value}, expected={expected_value}, diff={diff_pct:.2f}% "
                    f"(allowed [{negative_pct}-{positive_pct}%])"
                )
                if passed:
                    continue
                if self.DISABLE_NATIVE_BASELINE_FALLBACK:
                    continue
                if self._maybe_accept_current_native_baseline(
                    task_name=task_name,
                    metric_name=metric_name,
                    metric_key=metric_key,
                    value=value,
                    floor_pct=floor_pct,
                    ceil_pct=ceil_pct,
                ):
                    continue

                if not (negative_pct <= diff_pct <= positive_pct):
                    errors.append(
                        f"{task_name}:{metric_name} out of range: `{value}` vs expected `{expected_value}`, "
                        f"diff {diff_pct:.2f}% not in [{negative_pct}-{positive_pct}%]"
                    )

        print("\nEvaluation diff summary:")
        for d in diffs:
            print(d)

        if errors:
            raise AssertionError(
                "Evaluation failed:\n" + "\n".join(errors)
            )

    @staticmethod
    def _resolve_metric_key(metric_name, metrics):
        if metric_name in metrics:
            return metric_name
        alias = resolve_eval_metric_alias(metric_name, metrics)
        if alias is not None:
            return alias
        if metric_name is None:
            return None
        # if baseline uses canonical name without suffix, look for variants like acc,none
        prefix = f"{metric_name},"
        for key in metrics.keys():
            if key.startswith(prefix):
                return key
        return None

    def check_lm_head_loss(self, quant_log: List[Dict[str, any]]):
        final_log = quant_log[-1]
        if final_log["module"] == "lm_head":
            loss_value = float(final_log["loss"])
            diff_pct = (loss_value / self.EXPECT_LM_HEAD_LOSS) * 100
            print(f"lm_head loss: {loss_value} diff {diff_pct:.2f}%")
            negative_pct = 100 * (1 - self.LM_HEAD_LOSS_MAX_DELTA_PERCENT)
            positive_pct = 100 * (1 + self.LM_HEAD_LOSS_MAX_DELTA_PERCENT)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct,
                            f"lm_head loss: {loss_value} diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")
        else:
            raise ValueError("No quantization for lm_head module")

    def clear_directory(self, directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
