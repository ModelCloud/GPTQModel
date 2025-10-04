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

from pathlib import Path  # noqa: E402
from typing import Dict, List  # noqa: E402

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

import torch  # noqa: E402
import torch.cuda  # noqa: E402
from datasets import Dataset, concatenate_datasets, load_dataset  # noqa: E402
from ovis.image_to_test_dataset import get_calib_dataset  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402
from transformers.utils import is_flash_attn_2_available  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.perplexity import Perplexity  # noqa: E402
from gptqmodel.utils.model import MODALITY  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402

import random  # noqa: E402
from collections import Counter  # noqa: E402
import math  # noqa: E402


RAND_SEED = 898

log = LogBar.shared()
ATTN_IMPLEMENTATION_KEY = "attn_implementation"

class ModelTest(unittest.TestCase):
    DEBUG = True # enable extra debug output

    TASK_NAME = EVAL.LM_EVAL.ARC_CHALLENGE
    # sub test can modify
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.15  # -15%
    QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0  # 200%
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = False
    TORCH_DTYPE = "auto"
    EVAL_BATCH_SIZE = "auto"
    QUANT_BATCH_SIZE = 1
    LOAD_BACKEND = BACKEND.AUTO
    QUANT_BACKEND = BACKEND.AUTO
    USE_VLLM = False
    INPUTS_MAX_LENGTH = 2048
    MODEL_MAX_LEN = 4096
    DATASET_SIZE = 256
    DATASET_SORT = "asc"
    DELETE_QUANTIZED_MODEL = True
    MAX_QUANT_LAYERS = None

    # post-quant validation controls
    POST_QUANT_VALIDATION_BACKENDS = None  # default preserves legacy double-backend check

    # calibration noise controls
    CALIB_NOISE_PERCENT = 0.0  # share of calibration samples to synthesize
    CALIB_NOISE_MODE = "none" # "unseen"  # supported: none|random|unseen
    CALIB_NOISE_RANDOM_SEED = 1337
    CALIB_NOISE_MIN_SEQ_LEN = 32
    CALIB_NOISE_MAX_SEQ_LEN = 256
    CALIB_NOISE_GUARD_MAX_FREQ_RATIO = 1.3
    CALIB_NOISE_GUARD_MIN_TTR_FACTOR = 0.95
    CALIB_NOISE_GUARD_MAX_FRACTION = 0.1

    KERNEL_QUANT = {}  # kernel sets
    KERNEL_INFERENCE = {}  # kernel sets

    # quant config
    FORMAT = FORMAT.GPTQ
    METHOD = METHOD.GPTQ
    BITS = 4
    GROUP_SIZE = 128
    DESC_ACT = False
    SYM = True
    V2 = False
    ACT_GROUP_AWARE = True
    FAIL_SAFE = True
    EORA = None

    SAVE_PATH = None  # default is temp folder

    MOCK_QUANTIZATION = False
    ATTN_IMPLEMENTATION = None  # allow forcing a specific attention backend when needed; use "flash_attention_2"

    COMPUTE_PPL = False
    PPL_DATASET_PATH = "wikitext"
    PPL_DATASET_NAME = "wikitext-2-raw-v1"
    PPL_DATASET_SPLIT = "test"
    PPL_DATASET_COLUMN = "text"
    PPL_CTX = 512
    PPL_BATCH = 512
    PPL_MAX_SAMPLES = 32
    PPL_FALLBACK_CTX = 192
    PPL_FALLBACK_MAX_CHUNKS_PER_SAMPLE = 4

    INFERENCE_PROMPT = "The capital city of France is named"
    INFERENCE_RESULT_KEYWORDS = ["paris"]
    GENERATE_EVAL_SIZE_MIN = 128
    GENERATE_EVAL_SIZE_MAX = 128

    LM_HEAD_LOSS_MAX_DELTA_PERCENT = 0.1  # ±10%
    EXPECT_LM_HEAD_LOSS = None

    GENERIC_TEST_PROMPTS = [
        {"prompt": "Which city is the capital city of France?", "keywords": ["paris"]},
        {"prompt": "What is the smallest habitable planet in the milky way?", "keywords": ["earth"]},
        {"prompt": "Who wrote the play Romeo and Juliet?", "keywords": ["shakespeare"]},
        {"prompt": "What gas do plants primarily absorb from the atmosphere during photosynthesis?", "keywords": ["carbon dioxide"]},
        {"prompt": "Name the largest ocean on Earth.", "keywords": ["pacific"]},
    ]


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
        self.assertTrue(False, f"none of keywords were found in generated: `{generated}`")

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

    def _response_matches_keywords(self, response: str, keywords):
        if not response:
            return False

        normalized = response.lower()

        for keyword in keywords:
            if not keyword:
                continue

            needle = keyword.lower()

            if needle.isalpha():
                def _strip_other_alpha(text):
                    return "".join(ch for ch in text if ch.isalpha())

                if needle in normalized:
                    return True

                if _strip_other_alpha(needle) in _strip_other_alpha(normalized):
                    return True
            else:
                if needle in normalized:
                    return True

        return False

    def run_generic_inference_checks(self, model, tokenizer, backend):
        model.eval()
        log.info(f"Post-quant inference checks for backend `{backend.name}`")
        results = []
        for idx, item in enumerate(self.GENERIC_TEST_PROMPTS, start=1):
            prompt = item["prompt"]
            keywords = item["keywords"]
            try:
                response = self.generate_with_limit(model, tokenizer, prompt)
                matched = self._response_matches_keywords(response, keywords)
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

    def run_arc_challenge_eval(self, model, backend, trust_remote_code=False):
        previous_backend = self.LOAD_BACKEND
        self.LOAD_BACKEND = backend
        self._ensure_model_attributes(model)
        try:
            task_results = self.lm_eval(
                model=model,
                apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                trust_remote_code=trust_remote_code,
                delete_quantized_model=False,
            )
            log.info(f"[{backend.name}] ARC summary: {task_results}")
        except AttributeError as exc:
            log.warning(
                "Skipping ARC eval for backend %s due to attribute error: %s",
                backend.name,
                exc,
            )
            task_results = {}
        finally:
            self.LOAD_BACKEND = previous_backend
        return task_results

    @staticmethod
    def _ensure_model_attributes(model):
        inner_model = getattr(model, "model", None)
        if not hasattr(model, "device") and inner_model is not None:
            try:
                model.device = next(inner_model.parameters()).device
            except StopIteration:
                model.device = torch.device("cpu")
        if not hasattr(model, "config") and inner_model is not None:
            setattr(model, "config", getattr(inner_model, "config", None))

    def get_post_quant_validation_backends(self):
        configured = getattr(self, "POST_QUANT_VALIDATION_BACKENDS", None)
        if configured:
            return tuple(configured)

        if self.FORMAT is FORMAT.GPTQ:
            return (BACKEND.MARLIN, BACKEND.TORCH)
        return (BACKEND.MARLIN, BACKEND.GEMM)

    def perform_post_quant_validation(self, model_path, trust_remote_code=False):
        inference_records = {}
        arc_records = {}
        compare_backends = self.get_post_quant_validation_backends()
        executed_backends = []
        for backend in compare_backends:
            log.info(f"Loading post-quant model with backend `{backend.name}`")
            model = self.loadQuantModel(
                model_path,
                trust_remote_code=trust_remote_code,
                backend=backend,
            )
            tokenizer = model.tokenizer or self.load_tokenizer(model_path, trust_remote_code=trust_remote_code)
            inference_records[backend] = self.run_generic_inference_checks(model, tokenizer, backend)
            try:
                arc_records[backend] = self.run_arc_challenge_eval(model, backend, trust_remote_code=trust_remote_code)
            finally:
                del model
                torch_empty_cache()
            executed_backends.append(backend)
        self.render_inference_summary(inference_records, executed_backends)
        self.render_arc_summary(arc_records, executed_backends)

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

    @staticmethod
    def _colorize(text, matched):
        color = "\033[92m" if matched else "\033[91m"
        reset = "\033[0m"
        return f"{color}{text}{reset}"

    def render_inference_summary(self, inference_records, backends_order=None):
        if not inference_records:
            return
        if backends_order:
            ordered_backends = [backend for backend in backends_order if backend in inference_records]
        else:
            ordered_backends = list(inference_records.keys())
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

    def render_arc_summary(self, arc_records, backends_order=None):
        if not arc_records:
            return
        if backends_order:
            ordered_backends = [backend for backend in backends_order if backend in arc_records]
        else:
            ordered_backends = list(arc_records.keys())
        if not ordered_backends:
            return

        metrics = set()
        for results in arc_records.values():
            metrics.update(results.keys())
        metrics = sorted(metrics)

        table_rows = []
        tolerance = 0.01
        torch_reference = arc_records.get(BACKEND.TORCH, {})

        for metric in metrics:
            row = [metric]
            reference_value = torch_reference.get(metric)
            for backend in ordered_backends:
                value = arc_records[backend].get(metric)
                if value is None:
                    row.append(self._colorize("N/A", False))
                    continue
                if backend == BACKEND.TORCH or reference_value is None:
                    row.append(self._colorize(f"{value:.4f}", True))
                else:
                    matched = abs(value - reference_value) <= tolerance
                    row.append(self._colorize(f"{value:.4f}", matched))
            table_rows.append(row)

        headers = ["Metric"] + [backend.name for backend in ordered_backends]
        log.info("ARC challenge comparison:\n%s", tabulate(table_rows, headers=headers, tablefmt="github"))

    def load_tokenizer(self, model_id_or_path, trust_remote_code=False):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        return tokenizer

    @classmethod
    def load_dataset(cls, tokenizer=None, rows: int = 0):
        try:
            dataset = load_dataset(path="/monster/data/_ci_/nm-calibration", name="LLM", split="train")
        except Exception as exc:  # pragma: no cover - exercised in fallbacks
            log.warning("load_dataset failed; falling back to local parquet: %s", exc)
            dataset = cls._load_calibration_parquet()

        if rows > 0:
            dataset = dataset.select(range(min(rows, len(dataset))))

        return cls._apply_calibration_noise(dataset, tokenizer)

    @staticmethod
    def _load_calibration_parquet():
        parquet_path = Path("/monster/data/_ci_/nm-calibration/llm.parquet").expanduser()
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


    @classmethod
    def _apply_calibration_noise(cls, dataset, tokenizer):
        mode = (getattr(cls, "CALIB_NOISE_MODE", "none") or "none").lower()
        share = float(getattr(cls, "CALIB_NOISE_PERCENT", 0.0) or 0.0)

        cls._noise_summary = {
            "mode": mode,
            "share": share,
            "requested": 0,
            "generated": 0,
            "applied": False,
            "reason": "disabled",
        }

        if dataset is None or tokenizer is None or share <= 0.0 or mode == "none":
            return dataset

        records = cls._materialize_records(dataset)
        if not records:
            cls._noise_summary["reason"] = "empty_dataset"
            return dataset

        requested = max(1, int(len(records) * share))
        cls._noise_summary.update({
            "requested": requested,
            "reason": "generated",
        })

        stats = cls._collect_token_stats(records, tokenizer)
        if stats["total_tokens"] == 0:
            cls._noise_summary["reason"] = "no_tokens"
            return dataset

        noise_records, noise_token_sequences = cls._build_noise_records(
            tokenizer=tokenizer,
            stats=stats,
            sample_count=requested,
            mode=mode,
            base_records=records,
        )

        if not noise_records:
            cls._noise_summary["reason"] = "no_samples"
            return dataset

        if not cls._passes_noise_guard(stats, noise_token_sequences):
            cls._noise_summary["reason"] = "guard_block"
            return dataset

        merged = cls._merge_noise(dataset, noise_records, base_records=records)
        cls._noise_summary.update({
            "generated": len(noise_records),
            "applied": True,
            "reason": "ok",
        })
        log.info(
            "Injected %s synthetic calibration samples (mode=%s, avg_len=%s)",
            len(noise_records),
            mode,
            stats["avg_length"],
        )
        return merged

    @staticmethod
    def _materialize_records(dataset):
        if dataset is None:
            return []
        if hasattr(dataset, "to_list"):
            try:
                return dataset.to_list()
            except TypeError:
                pass
        try:
            return [dataset[idx] for idx in range(len(dataset))]
        except Exception:  # pragma: no cover - defensive fallback
            return list(dataset)

    @staticmethod
    def _extract_text(record):
        if isinstance(record, dict):
            if record.get("text"):
                return record["text"]
            if record.get("messages"):
                parts = [msg.get("content", "") for msg in record["messages"]]
                return "\n".join(part for part in parts if part)
        return ""

    @classmethod
    def _collect_token_stats(cls, records, tokenizer):
        counts = Counter()
        total_tokens = 0
        for record in records:
            text = cls._extract_text(record)
            if not text:
                continue
            encoded = tokenizer(text, add_special_tokens=False).get("input_ids", [])
            counts.update(encoded)
            total_tokens += len(encoded)

        unique_tokens = len(counts)
        avg_length = max(1, round(total_tokens / max(len(records), 1)))
        type_token_ratio = (unique_tokens / total_tokens) if total_tokens else 0.0
        max_freq = max(counts.values()) if counts else 0

        return {
            "counts": counts,
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "avg_length": avg_length,
            "type_token_ratio": type_token_ratio,
            "max_freq": max_freq,
        }

    @classmethod
    def _build_noise_records(cls, tokenizer, stats, sample_count, mode, base_records):
        if sample_count <= 0:
            return [], []

        if mode == "structured":
            return cls._build_structured_noise_records(
                tokenizer=tokenizer,
                base_records=base_records,
                sample_count=sample_count,
                stats=stats,
            )

        rng = random.Random(cls.CALIB_NOISE_RANDOM_SEED)

        try:
            vocab_values = list(tokenizer.get_vocab().values())
        except Exception:  # pragma: no cover - tokenizer fallback
            vocab_values = list(range(getattr(tokenizer, "vocab_size", 0)))

        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        vocab_ids = [tok for tok in vocab_values if tok not in special_ids]
        if not vocab_ids:
            return [], []

        existing_ids = set(stats["counts"].keys())
        if mode == "unseen":
            vocab_ids = [tok for tok in vocab_ids if tok not in existing_ids]
            if not vocab_ids:
                log.warning("No unseen tokens available for noise generation; skipping")
                return [], []

        seq_min = max(1, int(getattr(cls, "CALIB_NOISE_MIN_SEQ_LEN", 32)))
        seq_max = max(seq_min, int(getattr(cls, "CALIB_NOISE_MAX_SEQ_LEN", 256)))
        avg_target = min(seq_max, max(seq_min, stats["avg_length"]))
        std_dev = max(1, int(avg_target * 0.2))

        records = []
        token_sequences = []
        max_iterations = max(sample_count * 3, 32)

        for _ in range(max_iterations):
            if len(records) >= sample_count:
                break

            length = int(rng.gauss(avg_target, std_dev))
            length = max(seq_min, min(seq_max, length))
            if length <= 1:
                continue

            if mode == "unseen":
                length = min(length, len(vocab_ids))
                if length <= 1:
                    continue
                token_ids = rng.sample(vocab_ids, k=length)
            else:
                token_ids = rng.choices(vocab_ids, k=length)

            split = max(1, min(length - 1, length // 2))
            prompt_tokens = token_ids[:split]
            completion_tokens = token_ids[split:]

            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True).strip()
            completion_text = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

            if not prompt_text:
                prompt_text = " ".join(tokenizer.convert_ids_to_tokens(prompt_tokens))
            if not completion_text:
                completion_text = " ".join(tokenizer.convert_ids_to_tokens(completion_tokens))

            combined_text = (
                "### Instruction:\n"
                + prompt_text
                + "\n\n### Response:\n"
                + completion_text
            ).strip()

            encoded = tokenizer(combined_text, add_special_tokens=False)
            seq_ids = encoded.get("input_ids", [])
            if not seq_ids:
                continue

            records.append(
                {
                    "text": combined_text,
                    "messages": [
                        {"role": "user", "content": prompt_text},
                        {"role": "assistant", "content": completion_text},
                    ],
                }
            )
            token_sequences.append(seq_ids)

        return records, token_sequences

    @classmethod
    def _build_structured_noise_records(cls, tokenizer, base_records, sample_count, stats):
        if not base_records:
            log.warning("Structured noise requested but no base records; skipping")
            return [], []

        rng = random.Random(cls.CALIB_NOISE_RANDOM_SEED + 17)
        replacement_pool = list(stats["counts"].keys())
        if not replacement_pool:
            try:
                replacement_pool = list(tokenizer.get_vocab().values())
            except Exception:  # pragma: no cover - tokenizer fallback
                replacement_pool = list(range(getattr(tokenizer, "vocab_size", 0)))

        records = []
        token_sequences = []
        max_attempts = max(sample_count * 6, 48)

        for _ in range(max_attempts):
            if len(records) >= sample_count:
                break

            base_record = rng.choice(base_records)
            user_text, assistant_text = cls._extract_message_pair(base_record)
            if not user_text and not assistant_text:
                base_text = cls._extract_text(base_record)
                user_text, assistant_text = cls._split_instruction_response(base_text)

            if not user_text and not assistant_text:
                continue

            pert_user = cls._perturb_text(user_text, tokenizer, rng, replacement_pool, stats)
            pert_assistant = cls._perturb_text(assistant_text, tokenizer, rng, replacement_pool, stats)

            pert_user = pert_user or user_text
            pert_assistant = pert_assistant or assistant_text or pert_user

            if not pert_user or not pert_assistant:
                continue

            combined_text = (
                "### Instruction:\n"
                + pert_user.strip()
                + "\n\n### Response:\n"
                + pert_assistant.strip()
            ).strip()

            encoded = tokenizer(combined_text, add_special_tokens=False)
            seq_ids = encoded.get("input_ids", [])
            if len(seq_ids) < max(4, cls.CALIB_NOISE_MIN_SEQ_LEN // 2):
                continue

            records.append(
                {
                    "text": combined_text,
                    "messages": [
                        {"role": "user", "content": pert_user.strip()},
                        {"role": "assistant", "content": pert_assistant.strip()},
                    ],
                }
            )
            token_sequences.append(seq_ids)

        return records, token_sequences

    @staticmethod
    def _extract_message_pair(record):
        if not isinstance(record, dict):
            return "", ""

        messages = record.get("messages")
        if not isinstance(messages, list):
            return "", ""

        user_text = ""
        assistant_text = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if role == "user" and not user_text:
                user_text = content
            elif role == "assistant" and not assistant_text:
                assistant_text = content
            if user_text and assistant_text:
                break

        return user_text, assistant_text

    @staticmethod
    def _split_instruction_response(text):
        if not text:
            return "", ""

        instruction = ""
        response = ""

        if "### Response" in text:
            parts = text.split("### Response", 1)
            instruction = parts[0].replace("### Instruction:", "").strip()
            response = parts[1].replace(":", "", 1).strip()
        elif "Response:" in text:
            parts = text.split("Response:", 1)
            instruction = parts[0].replace("Instruction:", "").strip()
            response = parts[1].strip()
        else:
            split_parts = text.split("\n\n", 1)
            if len(split_parts) == 2:
                instruction, response = split_parts[0].strip(), split_parts[1].strip()
            else:
                mid = len(text) // 2
                instruction, response = text[:mid].strip(), text[mid:].strip()

        return instruction, response

    @classmethod
    def _perturb_text(cls, text, tokenizer, rng, replacement_pool, stats):
        if not text:
            return ""

        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = list(encoded.get("input_ids", []))
        if len(token_ids) <= 2:
            return text.strip()

        operations = ["shuffle", "drop", "replace"]
        operation = rng.choice(operations)
        tokens = list(token_ids)

        if operation == "shuffle" and len(tokens) > 8:
            chunk_size = max(1, len(tokens) // rng.randint(3, 6))
            segments = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
            rng.shuffle(segments)
            tokens = [tok for segment in segments for tok in segment]
        elif operation == "drop" and len(tokens) > 6:
            max_drop = max(1, min(len(tokens) // 4, 24))
            span = rng.randint(1, max_drop)
            if len(tokens) > span:
                start = rng.randint(0, len(tokens) - span)
                del tokens[start:start + span]
        else:  # replace
            span = max(1, min(len(tokens) // 5, 16))
            if len(tokens) > span:
                start = rng.randint(0, len(tokens) - span)
                replacement = cls._sample_replacement_tokens(rng, replacement_pool, span, stats)
                tokens[start:start + span] = replacement

        if not tokens:
            tokens = token_ids

        new_text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return new_text or text.strip()

    @classmethod
    def _sample_replacement_tokens(cls, rng, candidates, length, stats):
        if not candidates:
            candidates = list(stats["counts"].keys())
        if not candidates:
            candidates = list(range(1, max(512, length * 4)))
        return [rng.choice(candidates) for _ in range(length)]

    @classmethod
    def _passes_noise_guard(cls, stats, noise_sequences):
        if not noise_sequences:
            return False

        base_counts = stats["counts"].copy()
        base_total = stats["total_tokens"]
        base_max = stats["max_freq"]
        base_ttr = stats["type_token_ratio"]

        noise_counts = Counter()
        noise_token_total = 0
        for seq in noise_sequences:
            noise_counts.update(seq)
            noise_token_total += len(seq)

        if noise_token_total == 0:
            return False

        combined_counts = base_counts + noise_counts
        combined_max = max(combined_counts.values()) if combined_counts else 0
        if base_max == 0:
            base_max = combined_max or 1

        max_freq_ratio = combined_max / max(base_max, 1)
        if max_freq_ratio > getattr(cls, "CALIB_NOISE_GUARD_MAX_FREQ_RATIO", 1.3):
            log.info(
                "Noise guard triggered by max frequency ratio %.4f", max_freq_ratio
            )
            return False

        combined_total = base_total + noise_token_total
        combined_unique = len(combined_counts)
        combined_ttr = (combined_unique / combined_total) if combined_total else 0.0
        min_ttr = base_ttr * getattr(cls, "CALIB_NOISE_GUARD_MIN_TTR_FACTOR", 0.95)
        if base_ttr and combined_ttr < min_ttr:
            log.info(
                "Noise guard triggered by type-token ratio %.4f < %.4f",
                combined_ttr,
                min_ttr,
            )
            return False

        noise_fraction = noise_token_total / combined_total if combined_total else 1.0
        if noise_fraction > getattr(cls, "CALIB_NOISE_GUARD_MAX_FRACTION", 0.1):
            log.info(
                "Noise guard triggered by noise fraction %.4f", noise_fraction
            )
            return False

        return True

    @classmethod
    def _merge_noise(cls, dataset, noise_records, base_records=None):
        if dataset is None:
            return cls._LocalCalibrationDataset(noise_records)

        try:
            noise_dataset = Dataset.from_list(noise_records)
            return concatenate_datasets([dataset, noise_dataset])
        except Exception as exc:  # pragma: no cover - fall back to python dataset
            log.warning("Falling back to in-memory dataset for noise merge: %s", exc)
            if base_records is None:
                base_records = cls._materialize_records(dataset)
            combined = list(base_records) + list(noise_records)
            return cls._LocalCalibrationDataset(combined)


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
            fail_safe=self.FAIL_SAFE,
            sym=self.SYM,
            v2=self.V2,
            adapter=self.EORA,
            mock_quantization=self.MOCK_QUANTIZATION,
            offload_to_disk=getattr(self, "OFFLOAD_TO_DISK", True),
        )

        log.info(f"Quant config: {quantize_config}")
        log.info(f"Quant batch_size: {batch_size}")

        args = kwargs if kwargs else {}
        if (
            self.ATTN_IMPLEMENTATION is not None
            and ATTN_IMPLEMENTATION_KEY not in args
        ):
            args[ATTN_IMPLEMENTATION_KEY] = self.ATTN_IMPLEMENTATION

        log.info(f"args: {args}")
        model = GPTQModel.load(
            model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            device_map={"": "cpu"} if self.LOAD_BACKEND == BACKEND.TORCH_FUSED else "auto",
            debug=self.DEBUG,
            **args,
        )

        tokenizer = model.tokenizer

        is_image_to_text_model = MODALITY.IMAGE_TO_TEXT in model.modality
        calibration_dataset = get_calib_dataset(model) if is_image_to_text_model else self.load_dataset(tokenizer, self.DATASET_SIZE)

        noise_summary = getattr(self, "_noise_summary", None)
        if noise_summary and noise_summary.get("mode") != "none":
            log.info(f"Calibration noise summary: {noise_summary}")

        # mpt model need
        model_cfg = getattr(model, "config", None)
        if model_cfg is None:
            inner = getattr(model, "model", None)
            if inner is not None:
                model_cfg = getattr(inner, "config", None)
        if model_cfg is None:
            model_cfg = getattr(model, "model_config", None)
        if model_cfg is not None:
            if not getattr(model_cfg, "pad_token_id", None):
                model_cfg.pad_token_id = tokenizer.pad_token_id or 0
            if not getattr(model_cfg, "eos_token_id", None):
                model_cfg.eos_token_id = tokenizer.eos_token_id or 0

        is_quantized = model.quantized

        # ovis cannot load processor
        is_ovis_model = model.__class__.__name__ == "OvisGPTQ"
        need_create_processor = is_image_to_text_model and not is_ovis_model
        if not is_quantized:
            prev_max_layers = os.environ.get("GPTQMODEL_MAX_QUANT_LAYERS")
            max_layers_limit = getattr(self, "MAX_QUANT_LAYERS", None)
            if max_layers_limit is not None:
                os.environ["GPTQMODEL_MAX_QUANT_LAYERS"] = str(max_layers_limit)
            try:
                model.quantize(calibration_dataset, calibration_sort=self.DATASET_SORT, backend=self.QUANT_BACKEND, batch_size=batch_size)
            finally:
                if max_layers_limit is not None:
                    if prev_max_layers is None:
                        os.environ.pop("GPTQMODEL_MAX_QUANT_LAYERS", None)
                    else:
                        os.environ["GPTQMODEL_MAX_QUANT_LAYERS"] = prev_max_layers

            self.check_kernel(model, self.KERNEL_QUANT)

            if self.MOCK_QUANTIZATION:
                if need_create_processor:
                    processor = AutoProcessor.from_pretrained(model_id_or_path)
                    return model, tokenizer, processor
                return model, tokenizer

            # TODO: make into shared method
            with (contextlib.nullcontext(self.SAVE_PATH) if self.SAVE_PATH else contextlib.nullcontext(tempfile.mkdtemp()) if need_eval else tempfile.TemporaryDirectory()) as path:
                os.makedirs(path, exist_ok=True)
                self.clear_directory(path)

                model.save(path)
                tokenizer.save_pretrained(path)
                self._print_post_quant_artifacts(path)
                log.info(f"Quantized Model saved to tmp dir: {path}")
                self.perform_post_quant_validation(path, trust_remote_code=trust_remote_code)
                q_model = self.loadQuantModel(path, trust_remote_code=trust_remote_code)
                q_tokenizer = q_model.tokenizer
                if need_create_processor:
                    processor = AutoProcessor.from_pretrained(path)

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
        if (
            self.ATTN_IMPLEMENTATION is not None
            and ATTN_IMPLEMENTATION_KEY not in load_kwargs
        ):
            load_kwargs[ATTN_IMPLEMENTATION_KEY] = self.ATTN_IMPLEMENTATION
        elif ATTN_IMPLEMENTATION_KEY not in load_kwargs and is_flash_attn_2_available():
            load_kwargs[ATTN_IMPLEMENTATION_KEY] = "flash_attention_2"

        active_backend = backend if backend is not None else self.LOAD_BACKEND

        import os
        print("[DEBUG] loadQuantModel", model_id_or_path, trust_remote_code, active_backend)
        if trust_remote_code:
            prev_env = os.environ.get("TRANSFORMERS_TRUST_REMOTE_CODE")
            os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
        model = GPTQModel.load(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            backend=active_backend,
            device_map={"": "cpu"} if active_backend == BACKEND.TORCH_FUSED else "auto",
            debug=self.DEBUG,
            adapter=self.EORA,
            **load_kwargs
        )
        if trust_remote_code:
            if prev_env is None:
                os.environ.pop("TRANSFORMERS_TRUST_REMOTE_CODE", None)
            else:
                os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = prev_env

        return model

    def lm_eval(self, model, apply_chat_template=False, trust_remote_code=False, delete_quantized_model=False, extra_args:dict=None):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if self.USE_VLLM:
                    model_args = {
                        "pretrained": model.model_local_path,
                        "dtype": "auto", #"float16",
                        "gpu_memory_utilization": 0.8,
                        "tensor_parallel_size": 1,
                        "trust_remote_code": trust_remote_code,
                        "max_model_len": self.MODEL_MAX_LEN
                    }
                else:
                    model_args = {}
                if extra_args:
                    model_args.update(extra_args)

                from lm_eval.tasks import TaskManager
                from lm_eval.utils import make_table

                task_groups = EVAL.get_task_groups_from_tasks(self.TASK_NAME)

                for framework, tasks in task_groups.items():
                    log.info(f"TEST: EVAL starting: backend = {self.LOAD_BACKEND}")
                    log.info(f"Inference from model path: {model.model_local_path}")
                    results = GPTQModel.eval(
                        model_or_id_or_path=model.model_local_path,
                        llm_backend="vllm" if self.USE_VLLM else "gptqmodel",
                        model_args=model_args,
                        output_path=tmp_dir,
                        backend=self.LOAD_BACKEND,
                        framework=framework,
                        tasks=tasks,
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
                    task_results = {
                        metric: value for metric, value in results['results'].get(self.TASK_NAME.value, {}).items()
                        if metric != 'alias' and 'stderr' not in metric
                    }
                    print(task_results)

                # only delete tmp folders
                if delete_quantized_model and model.model_local_path.startswith("/tmp") and os.path.exists(
                        model.model_local_path):
                    log.info(f"Deleting temp model: {model.model_local_path}")
                    shutil.rmtree(model.model_local_path)
                return task_results
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
                                 apply_chat_template=apply_chat_template,
                                 trust_remote_code=trust_remote_code,
                                 delete_quantized_model=delete_quantized_model)
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, passed")
                else:
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, failed")
                    raise e
            else:
                raise e

    def calculatorPer(self, filter, value):
        if "norm" in filter:
            expected = self.NATIVE_ARC_CHALLENGE_ACC_NORM
        else:
            expected = self.NATIVE_ARC_CHALLENGE_ACC

        diff_pct = (value / expected) * 100
        log.info(f"{filter}: `{value}` vs `{expected}` diff {diff_pct:.2f}%")

        return diff_pct, expected

    def quant_lm_eval(self):
        self.model = None
        # TODO fix me: LOAD_QUANTIZED_MODEL doesn't make any sense when we have QUANT_SAVE_PATH
        #if self.QUANT_SAVE_PATH:
        #    self.model, _ = self.quantModel(self.QUANT_SAVE_PATH, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE)

        if not self.model:
            self.model, _ = self.quantModel(self.NATIVE_MODEL_ID, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, dtype=self.TORCH_DTYPE)

        self.check_kernel(self.model, self.KERNEL_INFERENCE)

        if self.MOCK_QUANTIZATION:
            task_results = {
                "acc,none": self.NATIVE_ARC_CHALLENGE_ACC,
                "acc_norm,none": self.NATIVE_ARC_CHALLENGE_ACC_NORM,
            }
        else:
            task_results = self.lm_eval(model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                                        apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                                        trust_remote_code=self.TRUST_REMOTE_CODE,
                                        delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
        self.check_results(task_results)
        self.last_task_results = task_results
        self._maybe_compute_perplexity(self.model)

    def check_results(self, task_results):
        for filter, value in task_results.items():
            diff_pct, expected = self.calculatorPer(filter=filter, value=value)
            negative_pct = 100 * (1 - self.QUANT_ARC_MAX_DELTA_FLOOR_PERCENT)
            positive_pct = 100 * (1 + self.QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct, f"{filter}: `{value}` vs expected `{expected}`, diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")

    def _maybe_compute_perplexity(self, model):
        self.perplexity_scores = []
        self.perplexity_avg = None
        self.perplexity_error = None

        if not self.COMPUTE_PPL or model is None:
            return None

        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            log.warning("Model has no tokenizer; skipping perplexity computation")
            return None

        try:
            ppl_runner = Perplexity(
                model=model,
                tokenizer=tokenizer,
                dataset_path=self.PPL_DATASET_PATH,
                dataset_name=self.PPL_DATASET_NAME,
                split=self.PPL_DATASET_SPLIT,
                text_column=self.PPL_DATASET_COLUMN,
            )
            scores = ppl_runner.calculate(n_ctx=self.PPL_CTX, n_batch=self.PPL_BATCH)
            if scores:
                self.perplexity_scores = scores
                self.perplexity_avg = sum(scores) / len(scores)
                log.info(
                    "Perplexity average: %.4f computed over %s windows",
                    self.perplexity_avg,
                    len(scores),
                )
            else:
                log.warning("Perplexity calculation returned no scores")
            return self.perplexity_avg
        except Exception as exc:  # pragma: no cover - diagnostics only
            self.perplexity_error = str(exc)
            log.error(f"Perplexity computation failed: {exc}")
            return self._compute_perplexity_fallback(model, tokenizer)

    def _compute_perplexity_fallback(self, model, tokenizer):
        max_samples = getattr(self, "PPL_MAX_SAMPLES", 32)
        dataset = None
        try:
            dataset = load_dataset(
                self.PPL_DATASET_PATH,
                self.PPL_DATASET_NAME,
                split=self.PPL_DATASET_SPLIT,
            )
        except Exception as exc:  # pragma: no cover - dataset missing
            log.error(f"Fallback perplexity dataset load failed: {exc}")
            return None

        sample_count = min(max_samples, len(dataset))
        if sample_count == 0:
            log.warning("Fallback perplexity has no samples to evaluate")
            return None

        max_context = getattr(model.config, "max_position_embeddings", self.PPL_CTX)
        fallback_ctx = min(
            getattr(self, "PPL_FALLBACK_CTX", self.PPL_CTX),
            self.PPL_CTX,
            max_context,
        )
        fallback_ctx = max(32, fallback_ctx)
        max_chunks_per_sample = max(1, getattr(self, "PPL_FALLBACK_MAX_CHUNKS_PER_SAMPLE", 1))
        rng = random.Random(self.CALIB_NOISE_RANDOM_SEED + 202)

        total_tokens = 0
        total_neg_log_likelihood = 0.0

        for entry in dataset.select(range(sample_count)):
            text = entry.get(self.PPL_DATASET_COLUMN)
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False).get("input_ids", [])
            if len(token_ids) <= 1:
                continue

            offsets = self._select_fallback_offsets(
                length=len(token_ids),
                chunk_size=fallback_ctx + 1,
                max_chunks=max_chunks_per_sample,
                rng=rng,
            )

            for offset in offsets:
                chunk = token_ids[offset: offset + fallback_ctx + 1]
                if len(chunk) <= 1:
                    continue

                input_tensor = torch.tensor(
                    chunk[:-1], dtype=torch.long, device=model.device
                ).unsqueeze(0)
                labels = torch.tensor(
                    chunk[1:], dtype=torch.long, device=model.device
                ).unsqueeze(0)
                attention_mask = torch.ones_like(input_tensor, dtype=torch.long)

                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_tensor,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                token_count = labels.numel()
                total_tokens += token_count
                total_neg_log_likelihood += outputs.loss.item() * token_count

        if total_tokens == 0:
            log.warning("Fallback perplexity produced zero tokens")
            return None

        average_loss = total_neg_log_likelihood / total_tokens
        ppl = math.exp(average_loss)
        self.perplexity_scores = [ppl]
        self.perplexity_avg = ppl
        log.info(
            "Fallback perplexity average: %.4f computed over %s tokens",
            ppl,
            total_tokens,
        )
        return ppl

    @staticmethod
    def _select_fallback_offsets(length, chunk_size, max_chunks, rng):
        if length <= chunk_size or max_chunks <= 1:
            return [0]

        limit = max(0, length - chunk_size)
        offsets = set()
        attempts = 0
        max_attempts = max_chunks * 6

        while len(offsets) < max_chunks and attempts < max_attempts:
            offsets.add(rng.randint(0, limit))
            attempts += 1

        if not offsets:
            return [0]
        return sorted(offsets)

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
