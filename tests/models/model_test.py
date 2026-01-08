# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
# -- do not touch
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

from enum import Enum  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional  # noqa: E402

from logbar import LogBar  # noqa: E402
from tabulate import tabulate  # noqa: E402


sys.path.insert(0, f"{str(Path(__file__).resolve().parent.parent)}/models")  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import textwrap  # noqa: E402
import unittest  # noqa: E402
from collections.abc import Iterable  # noqa: E402

import torch.cuda  # noqa: E402
from datasets import load_dataset  # noqa: E402


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

from gptqmodel import BACKEND, DEBUG_ON, GPTQModel  # noqa: E402
from gptqmodel.looper.module_looper import StopMainLoop  # noqa: E402
from gptqmodel.models.base import BaseQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.quantization.config import (  # noqa: E402
    FailSafe,
    GPTAQConfig,
    HessianConfig,
    QuantizeConfig,
    VramStrategy,
)
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.model import MODALITY  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402


RAND_SEED = 898

log = LogBar.shared()

DEFAULT_FLOOR_PCT = 0.05
DEFAULT_CEIL_PCT = 0.10
DEFAULT_TASK_NAMES = (EVAL.LM_EVAL.ARC_CHALLENGE,)


class ModelTest(unittest.TestCase):
    DEBUG = True # enable extra debug output

    VRAM_STRATEGY = VramStrategy.EXCLUSIVE
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
    DATASET_CONCAT_SIZE = None
    DATASET_CONCAT_SEPARATOR = None
    DATASET_SORT = "desc"
    DELETE_QUANTIZED_MODEL = True
    EVAL_TASKS = None
    EVAL_SINGLE_GPU = True
    LOAD_MODEL_EXTRA_ARGS: Dict[str, Any] = {}

    KERNEL_QUANT = {}  # kernel sets
    KERNEL_INFERENCE = {}  # kernel sets

    # quant config
    FORMAT = FORMAT.GPTQ
    METHOD = METHOD.GPTQ
    BITS = 4
    GROUP_SIZE = 128
    DESC_ACT = False
    SYM = True
    GPTQA = False
    ACT_GROUP_AWARE = True
    FAILSAFE = FailSafe()
    EORA = None
    DAMP_PERCENT = 0.05
    MSE = 0.0
    DYNAMIC = None
    HESSIAN_CHUNK_SIZE = None

    SAVE_PATH = None  # default is temp folder

    USE_FLASH_ATTN = True

    INFERENCE_PROMPT = "The capital city of France is named"
    INFERENCE_RESULT_KEYWORDS = ["paris"]
    GENERATE_EVAL_SIZE_MIN = 128
    GENERATE_EVAL_SIZE_MAX = 128

    LM_HEAD_LOSS_MAX_DELTA_PERCENT = 0.1  # ±10%
    EXPECT_LM_HEAD_LOSS = None
    STOP_AFTER_LAYER: Optional[int] = None

    GENERIC_TEST_PROMPTS = [
        {"prompt": "Which city is the capital city of France?", "keywords": ["paris"]},
        {"prompt": "What is the smallest habitable planet in the milky way?", "keywords": ["earth"]},
        {"prompt": "Who wrote the play Romeo and Juliet?", "keywords": ["shakespeare"]},
        {"prompt": "What gas do plants primarily absorb from the atmosphere during photosynthesis?", "keywords": ["carbon dioxide"]},
        {"prompt": "Name the largest ocean on Earth.", "keywords": ["pacific"]},
    ]

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
        need_create_processor: bool,
        cleanup_callback,
    ):
        if cleanup_callback is not None:
            try:
                cleanup_callback()
            except Exception:
                pass
        if need_create_processor:
            return model, tokenizer, processor
        return model, tokenizer

    def _normalize_task_identifier(self, task):
        if isinstance(task, Enum):
            return task.value
        if task is None:
            raise ValueError("Evaluation task identifier cannot be None")
        return str(task)

    def _normalize_task_list(self):
        task_specs = self.get_eval_tasks()
        task_lookup = getattr(self, "_resolved_task_lookup", {})
        resolved_tasks = []
        if task_specs:
            for normalized_name in task_specs.keys():
                original = task_lookup.get(normalized_name)
                if original is None:
                    original = self._resolve_task_enum(normalized_name)
                    if isinstance(task_lookup, dict):
                        task_lookup[normalized_name] = original
                resolved_tasks.append(original)
        else:
            resolved_tasks = list(DEFAULT_TASK_NAMES)
            self._resolved_task_lookup = {
                self._normalize_task_identifier(task): task for task in resolved_tasks
            }

        normalized = [self._normalize_task_identifier(task) for task in resolved_tasks if task is not None]
        if not normalized:
            raise ValueError("No evaluation tasks configured")
        return normalized

    def _resolve_task_enum(self, task):
        if isinstance(task, Enum):
            return task
        if isinstance(task, str):
            for enum_member in EVAL.get_task_enums():
                if task == enum_member.value or task == enum_member.name:
                    return enum_member
        raise ValueError(f"Unknown evaluation task identifier: {task}")

    def _legacy_arc_tasks(self):
        baselines = {}
        arc_metrics = {}
        if hasattr(self, "NATIVE_ARC_CHALLENGE_ACC"):
            arc_metrics["acc"] = {
                "value": self.NATIVE_ARC_CHALLENGE_ACC,
                "floor_pct": DEFAULT_FLOOR_PCT,
                "ceil_pct": DEFAULT_CEIL_PCT,
            }
        if hasattr(self, "NATIVE_ARC_CHALLENGE_ACC_NORM"):
            arc_metrics["acc_norm"] = {
                "value": self.NATIVE_ARC_CHALLENGE_ACC_NORM,
                "floor_pct": DEFAULT_FLOOR_PCT,
                "ceil_pct": DEFAULT_CEIL_PCT,
            }
        if arc_metrics:
            normalized = self._normalize_task_identifier(EVAL.LM_EVAL.ARC_CHALLENGE)
            baselines[normalized] = arc_metrics
            lookup = getattr(self, "_resolved_task_lookup", None)
            if isinstance(lookup, dict):
                lookup[normalized] = EVAL.LM_EVAL.ARC_CHALLENGE
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
            value = spec["value"]
            floor_pct = spec.get("floor_pct", spec.get("max_delta_floor_percent", default_floor))
            ceil_pct = spec.get("ceil_pct", spec.get("max_delta_ceil_percent", default_ceil))
            metric_key = spec.get("metric_key")
        else:
            value = spec
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

    def get_eval_tasks(self):
        self._resolved_task_lookup = {}
        self._task_chat_template = {}
        if self.EVAL_TASKS:
            baselines = {}
            for task, metrics in self.EVAL_TASKS.items():
                resolved_task = self._resolve_task_enum(task)
                normalized_task = self._normalize_task_identifier(resolved_task)
                self._resolved_task_lookup[normalized_task] = resolved_task

                metrics_dict = dict(metrics or {})
                chat_template = bool(metrics_dict.pop("chat_template", False))
                self._task_chat_template[normalized_task] = chat_template

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

    def generate_with_limit(self, model, tokenizer, prompt, max_new_tokens=512):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(generated[0], skip_special_tokens=True)

    def run_generic_inference_checks(self, model, tokenizer, backend):
        model.eval()
        log.info(f"Post-quant inference checks for backend `{backend.name}`")
        results = []
        for idx, item in enumerate(self.GENERIC_TEST_PROMPTS, start=1):
            prompt = item["prompt"]
            keywords = item["keywords"]
            try:
                response = self.generate_with_limit(model, tokenizer, prompt)
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

    def run_eval_tasks(self, model, backend, trust_remote_code=False):
        previous_backend = self.LOAD_BACKEND
        self.LOAD_BACKEND = backend
        try:
            task_results = self.lm_eval(
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

    def perform_post_quant_validation(self, model_path, trust_remote_code=False):
        inference_records = {}
        eval_records = {}
        reuse_candidates = {}

        if self.FORMAT is FORMAT.GPTQ:
            if self.LOAD_BACKEND == BACKEND.MARLIN:
                compare_backends = (BACKEND.MARLIN,)
            else:
                compare_backends = (self.LOAD_BACKEND,)
        else:
            compare_backends = (BACKEND.MARLIN, BACKEND.GEMM)
        fallback_backend = None
        if BACKEND.MARLIN in compare_backends:
            try:
                from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # type: ignore
            except Exception:  # pragma: no cover - fallback if module unavailable
                marlin_group_sizes = ()
                marlin_sym = ()
            else:
                marlin_group_sizes = tuple(getattr(MarlinQuantLinear, "SUPPORTS_GROUP_SIZE", ()))
                marlin_sym = tuple(getattr(MarlinQuantLinear, "SUPPORTS_SYM", ()))

            requested_group_size = getattr(self, "GROUP_SIZE", None)
            requested_sym = getattr(self, "SYM", None)

            marlin_supported = True
            if marlin_group_sizes and requested_group_size not in marlin_group_sizes:
                marlin_supported = False
            if marlin_sym and requested_sym not in marlin_sym:
                marlin_supported = False

            if not marlin_supported:
                fallback_backend = BACKEND.TORCH
                compare_backends = tuple(
                    BACKEND.TORCH if backend == BACKEND.MARLIN else backend
                    for backend in compare_backends
                )
                log.info(
                    f"Marlin backend unsupported for current quant config (group_size={requested_group_size}, sym={requested_sym}); "
                    "falling back to BACKEND.TORCH for validation."
                )

        if fallback_backend is not None and self.LOAD_BACKEND == BACKEND.MARLIN:
            self._effective_load_backend = fallback_backend
        else:
            self._effective_load_backend = None

        target_backend = self._current_load_backend()
        can_reuse = target_backend not in (BACKEND.AUTO, BACKEND.AUTO_TRAINABLE)

        for backend in compare_backends:
            log.info(f"Loading post-quant model with backend `{backend.name}`")
            # When EVAL_SINGLE_GPU is enabled, pin post-quant loads to the first CUDA device to avoid auto sharding.
            use_cuda_map = (
                self.EVAL_SINGLE_GPU
                and torch.cuda.is_available()
                and backend != BACKEND.TORCH_FUSED
            )
            if use_cuda_map:
                model = self.loadQuantModel(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    backend=backend,
                    device_map={"": "cuda:0"},
                )
            else:
                model = self.loadQuantModel(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    backend=backend,
                )
            tokenizer = model.tokenizer or self.load_tokenizer(model_path, trust_remote_code=trust_remote_code)
            inference_records[backend] = self.run_generic_inference_checks(model, tokenizer, backend)

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

    def _prepare_quant_save_destination(self, need_eval):
        if self.SAVE_PATH:
            return contextlib.nullcontext(self.SAVE_PATH), self.SAVE_PATH, None

        if need_eval:
            tmp_dir = tempfile.mkdtemp()
            return contextlib.nullcontext(tmp_dir), tmp_dir, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)

        tmp_context = tempfile.TemporaryDirectory()
        return tmp_context, tmp_context.name, tmp_context.cleanup

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
        ordered_backends = [backend for backend in (BACKEND.MARLIN, BACKEND.TORCH) if backend in inference_records]
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
        ordered_backends = [backend for backend in (BACKEND.MARLIN, BACKEND.TORCH) if backend in eval_records]
        if not ordered_backends:
            return

        flattened_records = {
            backend: self._flatten_task_metrics(results) for backend, results in eval_records.items()
        }

        metrics = sorted({metric for results in flattened_records.values() for metric in results.keys()})

        table_rows = []
        tolerance = 0.01
        torch_reference = flattened_records.get(BACKEND.TORCH, {})

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
                if backend == BACKEND.TORCH:
                    row.append(self._colorize(f"{value:.4f}", True))
                else:
                    matched = reference_value is not None and abs(value - reference_value) <= tolerance
                    row.append(self._colorize(f"{value:.4f}", matched))
            table_rows.append(row)

        headers = ["Metric"] + [backend.name for backend in ordered_backends]
        log.info("Evaluation comparison:\n%s", tabulate(table_rows, headers=headers, tablefmt="github"))

    def load_tokenizer(self, model_id_or_path, trust_remote_code=False):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        return tokenizer

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        try:
            dataset = load_dataset(path="/monster/data/model/dataset/nm-calibration", name="LLM", split="train")
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

    def quantModel(self, model_id_or_path, trust_remote_code=False, dtype="auto", need_eval=True, batch_size: int = QUANT_BATCH_SIZE, **kwargs):
        quantize_config = QuantizeConfig(
            quant_method=self.METHOD,
            format=self.FORMAT,
            bits=self.BITS,
            group_size=self.GROUP_SIZE,
            desc_act=self.DESC_ACT if not self.ACT_GROUP_AWARE else False,
            act_group_aware=self.ACT_GROUP_AWARE,
            failsafe=self.FAILSAFE,
            sym=self.SYM,
            gptaq=GPTAQConfig() if self.GPTQA else None,
            adapter=self.EORA,
            pack_impl="cpu",
            vram_strategy=self.VRAM_STRATEGY,
            damp_percent=self.DAMP_PERCENT,
            mse=self.MSE,
            dynamic=self.DYNAMIC,
            hessian=HessianConfig(chunk_size=self.HESSIAN_CHUNK_SIZE),
        )

        log.info(f"Quant config: {quantize_config}")
        log.info(f"Quant batch_size: {batch_size}")

        args = kwargs if kwargs else {}

        if self.USE_FLASH_ATTN:
            if is_flash_attn_2_available():
                args["attn_implementation"] = "flash_attention_2"
            else:
                log.warn("flash-attn requested but not available; falling back to framework defaults")


        log.info(f"args: {args}")
        model = GPTQModel.load(
            model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map={"": "cpu"} if self.LOAD_BACKEND == BACKEND.TORCH_FUSED else "auto",
            **args,
        )

        self._layer_stop_callback = None
        if DEBUG_ON and self.STOP_AFTER_LAYER is not None:
            self._layer_stop_callback = self._build_layer_stop_callback(self.STOP_AFTER_LAYER)
            model.layer_callback = self._layer_stop_callback

        tokenizer = model.tokenizer
        self._post_quant_eval_records = {}
        self._effective_load_backend = None
        processor = None

        is_image_to_text_model = MODALITY.IMAGE_TO_TEXT in model.modality
        calibration_dataset = get_calib_dataset(model) if is_image_to_text_model else self.load_dataset(tokenizer, self.DATASET_SIZE)

        # mpt model need
        if not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id or 0
        if not model.config.eos_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id or 0

        is_quantized = model.quantized

        # ovis cannot load processor
        is_ovis_model = model.__class__.__name__ == "OvisGPTQ"
        need_create_processor = is_image_to_text_model and not is_ovis_model

        debug_short_circuit = False
        if not is_quantized:
            save_context = None
            planned_save_path = None
            cleanup_callback = None
            try:
                save_context, planned_save_path, cleanup_callback = self._prepare_quant_save_destination(need_eval)
                log.info(f"Quantized model artifacts will be saved to: {planned_save_path}")
                model.quantize(
                    calibration_dataset,
                    calibration_concat_size=self.DATASET_CONCAT_SIZE,
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
                        need_create_processor=need_create_processor,
                        cleanup_callback=cleanup_callback,
                    )

                # TODO: make into shared method
                with save_context as path:
                    cleanup_callback = None
                    os.makedirs(path, exist_ok=True)
                    self.clear_directory(path)

                    model.save(path)
                    tokenizer.save_pretrained(path)
                    self._print_post_quant_artifacts(path)

                    reuse_candidates, eval_records = self.perform_post_quant_validation(path, trust_remote_code=trust_remote_code)
                    self._post_quant_eval_records = eval_records
                    target_backend = self._current_load_backend()

                    q_model = reuse_candidates.pop(target_backend, None)
                    if q_model is None:
                        # When single-GPU evaluation is requested, keep the reload scoped to cuda:0.
                        use_cuda_map = (
                            self.EVAL_SINGLE_GPU
                            and torch.cuda.is_available()
                            and target_backend != BACKEND.TORCH_FUSED
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
                        processor = AutoProcessor.from_pretrained(path)
            except Exception:
                if cleanup_callback is not None:
                    try:
                        cleanup_callback()
                    except Exception:
                        pass
                raise

        else:
            if need_create_processor:
                processor = AutoProcessor.from_pretrained(model_id_or_path)
        if not is_quantized:
            del model
            torch_empty_cache()
            if need_create_processor:
                return q_model, q_tokenizer, processor
            else:
                return q_model, q_tokenizer
        else:
            if need_create_processor:
                return model, tokenizer, processor
            else:
                return model, tokenizer

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

        active_backend = backend if backend is not None else self._current_load_backend()

        default_device_map = {"": "cpu"} if active_backend == BACKEND.TORCH_FUSED else "auto"
        explicit_device = "device" in load_kwargs
        inserted_device_map = False
        if "device_map" not in load_kwargs and not explicit_device:
            load_kwargs["device_map"] = default_device_map
            inserted_device_map = True

        # Post-quant CI runs may expose multiple GPUs; pin loading to the first one to avoid spread-out auto maps.
        if (
            (inserted_device_map or load_kwargs.get("device_map") == "auto")
            and not explicit_device
            and active_backend != BACKEND.TORCH_FUSED
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

    def lm_eval(self, model, trust_remote_code=False, delete_quantized_model=False, extra_args:dict=None):
        try:
            task_names = self._normalize_task_list()
            aggregated_results = {}
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = getattr(model, "model_local_path", None)
                if isinstance(model, str):
                    model_path = model

                if self.USE_VLLM:
                    tensor_parallel = 1
                    if not self.EVAL_SINGLE_GPU:
                        try:
                            candidate = torch.cuda.device_count()
                        except Exception:
                            candidate = 1
                        tensor_parallel = max(1, candidate)
                    model_args = {
                        "pretrained": model_path,
                        "dtype": "auto", #"float16",
                        "gpu_memory_utilization": 0.8,
                        "tensor_parallel_size": tensor_parallel,
                        "trust_remote_code": trust_remote_code,
                        "max_model_len": self.MODEL_MAX_LEN
                    }
                else:
                    model_args = {}
                if extra_args:
                    model_args.update(extra_args)

                from lm_eval.tasks import TaskManager
                from lm_eval.utils import make_table

                task_groups = EVAL.get_task_groups_from_tasks(task_names)

                chat_template_lookup = getattr(self, "_task_chat_template", {}) or {}

                for framework, tasks in task_groups.items():
                    active_backend = self._current_load_backend()
                    log.info(f"TEST: EVAL starting: backend = {active_backend.name}")
                    if model_path:
                        log.info(f"Inference from model path: {model_path}")

                    if isinstance(model, BaseQModel) and not self.USE_VLLM:
                        eval_target = model
                    else:
                        eval_target = model_path

                    if eval_target is None:
                        raise ValueError("Model evaluation target could not be determined.")

                    resolved_lookup = getattr(self, "_resolved_task_lookup", {})
                    eval_tasks = []
                    for task in tasks:
                        original_task = resolved_lookup.get(task)
                        if original_task is None:
                            original_task = self._resolve_task_enum(task)
                            if isinstance(resolved_lookup, dict):
                                normalized_task = self._normalize_task_identifier(original_task)
                                resolved_lookup[normalized_task] = original_task
                        eval_tasks.append(original_task)

                    grouped_tasks: Dict[bool, List] = {}
                    for task in eval_tasks:
                        normalized_name = self._normalize_task_identifier(task)
                        apply_chat = bool(chat_template_lookup.get(normalized_name, False))
                        grouped_tasks.setdefault(apply_chat, []).append(task)

                    for apply_chat_template, grouped in grouped_tasks.items():
                        results = GPTQModel.eval(
                            model_or_id_or_path=eval_target,
                            llm_backend="vllm" if self.USE_VLLM else "gptqmodel",
                            model_args=model_args,
                            output_path=tmp_dir,
                            backend=active_backend,
                            framework=framework,
                            tasks=grouped,
                            apply_chat_template=apply_chat_template,
                            trust_remote_code=trust_remote_code,
                            batch_size=self.EVAL_BATCH_SIZE,
                            gen_kwargs="temperature=0.0,top_k=50",
                            random_seed=RAND_SEED,
                            task_manager=TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tasks"), include_defaults=False)
                        )

                        print('--------Eval Result---------')
                        print(make_table(results))
                        if "groups" in results:
                            print(make_table(results, "groups"))
                        print('--------Eval Result End---------')

                        for task_name in grouped:
                            normalized_task_name = self._normalize_task_identifier(task_name)
                            metrics = results["results"].get(normalized_task_name, {})
                            filtered_metrics = {
                                metric: value
                                for metric, value in metrics.items()
                                if metric != "alias" and "stderr" not in metric
                            }
                            aggregated_results[normalized_task_name] = filtered_metrics
                            print({normalized_task_name: filtered_metrics})

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
                    self.lm_eval(model=model,
                                 trust_remote_code=trust_remote_code,
                                 delete_quantized_model=delete_quantized_model,
                                 extra_args=extra_args)
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, passed")
                else:
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, failed")
                    raise e
            else:
                raise e

    def calculatorPer(self, task_name, metric_name, value, expected):
        diff_pct = (value / expected) * 100
        log.info(f"{task_name}:{metric_name}: `{value}` vs `{expected}` diff {diff_pct:.2f}%")
        return diff_pct, expected

    def quant_lm_eval(self):
        self.model = None
        # TODO fix me: LOAD_QUANTIZED_MODEL doesn't make any sense when we have QUANT_SAVE_PATH
        #if self.QUANT_SAVE_PATH:
        #    self.model, _ = self.quantModel(self.QUANT_SAVE_PATH, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE)

        if not self.model:
            self.model, _ = self.quantModel(self.NATIVE_MODEL_ID, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE)

        self.check_kernel(self.model, self.KERNEL_INFERENCE)

        if self._debug_layer_stop_triggered():
            log.info("DEBUG mode: skipping lm_eval and baseline checks after early layer stop.")
            return

        eval_records = getattr(self, "_post_quant_eval_records", {})
        target_backend = self._current_load_backend()
        if eval_records and len(eval_records) == 1 and target_backend in eval_records:
            log.info("Reusing evaluation results for backend `%s`; skipping duplicate lm_eval run", target_backend.name)
            task_results = eval_records[target_backend]
        else:
            task_results = self.lm_eval(
                model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                trust_remote_code=self.TRUST_REMOTE_CODE,
                delete_quantized_model=self.DELETE_QUANTIZED_MODEL,
            )
        self.check_results(task_results)
        self._cleanup_quantized_model(self.model, enabled=self.DELETE_QUANTIZED_MODEL)

    def check_results(self, task_results):
        baselines = self.get_eval_tasks()
        if not baselines:
            raise AssertionError("No evaluation baselines configured for result validation.")

        for task_name, expected_metrics in baselines.items():
            metrics = task_results.get(task_name)
            if metrics is None:
                self.fail(f"No evaluation results returned for task `{task_name}`")
            if not isinstance(metrics, dict):
                raise TypeError(f"Expected metrics for task `{task_name}` to be a dictionary, got {type(metrics).__name__}")

            for metric_name, baseline_spec in expected_metrics.items():
                metric_key = baseline_spec.get("metric_key") or metric_name
                metric_key = self._resolve_metric_key(metric_key, metrics)
                if metric_key is None:
                    self.fail(f"Metric `{metric_name}` missing from results for task `{task_name}`")

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
                negative_pct = 100 * (1 - floor_pct)
                positive_pct = 100 * (1 + ceil_pct)
                self.assertTrue(
                    negative_pct <= diff_pct <= positive_pct,
                    f"{task_name}:{metric_name}: `{value}` vs expected `{expected_value}`, "
                    f"diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]",
                )

    @staticmethod
    def _resolve_metric_key(metric_name, metrics):
        if metric_name in metrics:
            return metric_name
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
