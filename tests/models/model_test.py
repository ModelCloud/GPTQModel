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
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from collections.abc import Iterable  # noqa: E402

import torch.cuda  # noqa: E402
from datasets import load_dataset  # noqa: E402
from ovis.image_to_test_dataset import get_calib_dataset  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.model import MODALITY  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402


RAND_SEED = 898

log = LogBar.shared()

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

    USE_FLASH_ATTN = True

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
                if matched:
                    log.info(f"[{backend.name}] Prompt {idx} PASS: `{prompt}`")
                else:
                    snippet = response.replace("\n", " ")[:200]
                    log.error(f"[{backend.name}] Prompt {idx} MISS: `{prompt}` -> `{snippet}`")
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
        try:
            task_results = self.lm_eval(
                model=model,
                apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                trust_remote_code=trust_remote_code,
                delete_quantized_model=False,
            )
            log.info(f"[{backend.name}] ARC summary: {task_results}")
        finally:
            self.LOAD_BACKEND = previous_backend
        return task_results

    def perform_post_quant_validation(self, model_path, trust_remote_code=False):
        inference_records = {}
        arc_records = {}
        for backend in (BACKEND.MARLIN, BACKEND.TORCH):
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
        self.render_inference_summary(inference_records)
        self.render_arc_summary(arc_records)

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
        table_rows = []
        sanity_scores = {}
        for backend in ordered_backends:
            entries = {entry["prompt"]: entry for entry in inference_records[backend]}
            matched_count = sum(1 for entry in entries.values() if entry.get("matched"))
            total_count = len(entries) if entries else 1
            sanity_scores[backend] = (matched_count, total_count)

        for prompt in prompts:
            row = [prompt, ", ".join(self._normalize_keyword_case(k) for k in self._keywords_for_prompt(prompt))]
            for backend in ordered_backends:
                entry = next((item for item in inference_records[backend] if item["prompt"] == prompt), None)
                if entry is None:
                    row.append(self._colorize("N/A", False))
                    continue
                matched = entry["matched"]
                snippet = entry["response"].replace("\n", " ")[:80]
                row.append(self._colorize(f"{'PASS' if matched else 'MISS'} | {snippet}", matched))
            table_rows.append(row)

        headers = ["Prompt", "Expected Keywords"] + [backend.name for backend in ordered_backends]
        log.info("Sanity prompt comparison:\n%s", tabulate(table_rows, headers=headers, tablefmt="github"))

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

    def render_arc_summary(self, arc_records):
        if not arc_records:
            return
        ordered_backends = [backend for backend in (BACKEND.MARLIN, BACKEND.TORCH) if backend in arc_records]
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
                if backend == BACKEND.TORCH:
                    row.append(self._colorize(f"{value:.4f}", True))
                else:
                    matched = reference_value is not None and abs(value - reference_value) <= tolerance
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
            return dataset.select(range(min(rows, len(dataset))))
        return dataset

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
        )

        log.info(f"Quant config: {quantize_config}")
        log.info(f"Quant batch_size: {batch_size}")

        args = kwargs if kwargs else {}

        if self.USE_FLASH_ATTN:
            args["attn_implementation"] = "flash_attention_2"


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

        # mpt model need
        if not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id or 0
        if not model.config.eos_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id or 0

        is_quantized = model.quantized

        # ovis cannot load processor
        is_ovis_model = model.__class__.__name__ == "OvisGPTQ"
        need_create_processor = is_image_to_text_model and not is_ovis_model
        if not is_quantized:
            model.quantize(calibration_dataset, calibration_sort=self.DATASET_SORT, backend=self.QUANT_BACKEND, batch_size=batch_size)

            self.check_kernel(model, self.KERNEL_QUANT)

            # TODO: make into shared method
            with (contextlib.nullcontext(self.SAVE_PATH) if self.SAVE_PATH else contextlib.nullcontext(tempfile.mkdtemp()) if need_eval else tempfile.TemporaryDirectory()) as path:
                os.makedirs(path, exist_ok=True)
                self.clear_directory(path)

                model.save(path)
                tokenizer.save_pretrained(path)
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

        if self.USE_FLASH_ATTN:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        active_backend = backend if backend is not None else self.LOAD_BACKEND

        model = GPTQModel.load(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            backend=active_backend,
            device_map={"": "cpu"} if active_backend == BACKEND.TORCH_FUSED else "auto",
            debug=self.DEBUG,
            adapter=self.EORA,
            **load_kwargs
        )

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

        task_results = self.lm_eval(model=self.SAVE_PATH if self.SAVE_PATH else self.model,
                                    apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                                    trust_remote_code=self.TRUST_REMOTE_CODE,
                                    delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
        self.check_results(task_results)

    def check_results(self, task_results):
        for filter, value in task_results.items():
            diff_pct, expected = self.calculatorPer(filter=filter, value=value)
            negative_pct = 100 * (1 - self.QUANT_ARC_MAX_DELTA_FLOOR_PERCENT)
            positive_pct = 100 * (1 + self.QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct, f"{filter}: `{value}` vs expected `{expected}`, diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")

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
